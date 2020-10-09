# -*- coding: utf-8 -*-
import csv
import os, sys, getopt
import math
import tkinter as tk
from tkinter import filedialog
import subprocess
from subprocess import check_output
import multiprocessing
import time
import pandas as pd
import porespy

import cv2
import numpy

#remove root windows
root = tk.Tk()
root.withdraw()

def programInfo():
    print("##########################################################")
    print("# Class to get PSD / CLDs of a single or multiple images #")
    print("#                                                        #")
    print("# © 2020 Florian Kleiner                                 #")
    print("#   Bauhaus-Universität Weimar                           #")
    print("#   Finger-Institut für Baustoffkunde                    #")
    print("#                                                        #")
    print("##########################################################")
    print()

home_dir = os.path.dirname(os.path.realpath(__file__))

ts_path = os.path.dirname( home_dir ) + os.sep + 'tiff_scaling' + os.sep
ts_file = 'set_tiff_scaling'
if ( os.path.isdir( ts_path ) and os.path.isfile( ts_path + ts_file + '.py' ) or os.path.isfile( home_dir + ts_file + '.py' ) ):
    if ( os.path.isdir( ts_path ) ): sys.path.insert( 1, ts_path )
    import extract_tiff_scaling as es
else:
    programInfo()
    print( 'missing ' + ts_path + ts_file + '.py!' )
    print( 'download from https://github.com/kleinerELM/tiff_scaling' )
    sys.exit()

def singeFileProcess(analyze_color_name='black', verbose=False):

    print( "Please select the directory with the source image tiles.")#, end="\r" )
    filepath = filedialog.askopenfilename( title='Please select the reference image', filetypes=[("Tiff images", "*.tif;*.tiff")] )
    file_name, file_extension = os.path.splitext( filepath )
    file_name = os.path.basename(file_name)

    return size_distribution(os.path.dirname(filepath), file_name, file_extension, analyze_color_name=analyze_color_name, verbose=verbose)

class size_distribution():
    progressFactor = 500
    image_count = 0
    tile_width = 0
    tile_height = 0
    tile_area = 0

    scaling = es.getEmptyScaling()
    pore_diameter_limit = 100 #nm

    black = 0
    white = 255

    folder = ''
    output_folder = ''

    def set_color_to_be_analyzed( self, color_name ):
        if color_name == "white":
            self.analyze_Color = self.white
            self.ignore_Color = self.black
        else:
            self.analyze_Color = self.black
            self.ignore_Color = self.white

    def getPoreDiameter( self, area ):
        return ( math.sqrt( area /math.pi )*2 )

    def getPoreArea( self, diameter ):
        radius = diameter/2
        return 4/3*(math.pi*(radius**3))

    def getPoreVolume( self, diameter=None, area=None ):
        if diameter == None: diameter = getPoreDiameter( area )
        radius = diameter/2
        return 4/3*(math.pi*(radius**3))

    def getPoreSurface( self, diameter=None, area=None ):
        if diameter == None: diameter = getPoreDiameter( area )
        radius = diameter/2
        return (4*math.pi*(radius**2))

    def get_scaled_width(self):
        return self.tile_width*self.scaling['x']

    def get_scaled_height(self):
        return self.tile_height*self.scaling['y']

    def check_tile_dimension(self, height, width):
        if self.tile_width + self.tile_height == 0:
            self.tile_width = width
            self.tile_height = height
            self.tile_area = height*width
            print('size: {} x {} {} / {} x {} px '.format(self.get_scaled_width(),self.get_scaled_height(),self.scaling['unit'],width,height))
            return True
        else:
            if self.tile_width == width and self.tile_height == height:
                return True
            else:
                print( 'Tile sizes do not match! (w: {} != {} | h: {} != {})'.format(width, self.tile_width, height, self.tile_height) )
                exit()
        return False

    def load_binary_image(self):
        self.scaling = es.autodetectScaling( self.filename, self.folder )
        self.img = cv2.imread( self.folder + os.sep + self.filename, cv2.IMREAD_GRAYSCALE )

        if self.img is None:
            print( 'Error loading {}'.format(self.filename))
            exit()

        #binarizes an image
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        min_val = self.black #int( (self.ignore_Color+1)/2 ) if self.ignore_Color > self.analyze_Color else self.analyze_Color
        max_val = int( (self.white+1)/2 ) #int( (self.analyze_Color+1)/2 ) if self.analyze_Color > self.ignore_Color else self.ignore_Color
        _, self.thresh_img = cv2.threshold(self.img, min_val, max_val, cv2.THRESH_BINARY)

    def get_file(self, file_name, file_extension, process_position=None, verbose=False ):
        processID = ' '
        if process_position != None: processID = " #{}: ".format(process_position)
        self.filename = file_name + file_extension
        self.load_binary_image()

        if verbose: print( '{}Analysing {}'.format(processID, self.filename) )

        height, width = self.img.shape[:2]

        if self.check_tile_dimension(height, width):
            new_row = {'filename':file_name, 'height':height, 'width':width }
        self.analyzed_files_DF = self.analyzed_files_DF.append(new_row, ignore_index=True)

    def process_result_row( self, file_index, diameter=0, area=0 ):
        result_row = False
        if area > 0 or diameter > 0:
            if area > 0:
                diameter = self.getPoreDiameter(area)
            else:
                area = self.getPoreArea(diameter)
            result_row = {
                'file_index':   file_index,
                'diameter':     diameter,
                'area' :        area,
                'surface':      self.getPoreSurface(diameter=diameter),
                'volume':       self.getPoreVolume(diameter=diameter)
            }
        return result_row

    def processPSD( self, img, processID = ' ', verbose=False ):
        processID = ' '

        contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #print(len(self.analyzed_files_DF))
        #print(self.analyzed_files_DF.iloc[-1])
        areas = []
        contourCount = len(contours)
        file_index = int(len(self.analyzed_files_DF)-1)
        emptyArea = 0

        startTime = int(round(time.time() * 1000))
        for i in range(0, contourCount):
            area = cv2.contourArea(contours[i])
            result_row = self.process_result_row( file_index, area=area )
            if result_row != False:
                self.psd_df = self.psd_df.append(result_row, ignore_index=True)
                if i % self.progressFactor == 0 and verbose: print( "{}...processing particle #{}/{}".format(processID, i, contourCount ))#, end="\r")
            else:
                emptyArea += 1
        if verbose:
            print( '{}Analysed particles: {}, ignored 0-values: {}, finished in {} ms'.format(processID, contourCount-emptyArea, emptyArea, int(round(time.time() * 1000)) - startTime) )
            #self.psd_df.to_csv( self.output_folder + file_name + '_pores.csv', index=False )
            #print(self.psd_df)

    def process_directional_CLD( self, img, direction, processID = ' ', verbose=False):
        file_index = int(len(self.analyzed_files_DF)-1)

        ignoreBorder = True
        minLength = 1

        #process variables
        lineCount = 0
        lastValue = -1      # var to save the last value
        lastChangedPos = -1   # var to save the last position
        usedImageArea = 0
        fullPoreArea = 0

        if verbose: print( '  processing {} cord length distribution'.format(direction) )#, end='' )
        startTime = int(round(time.time() * 1000))
        if ( direction == 'horizontal' ):
            for y in range(self.tile_height):
                if y % self.progressFactor == 0 and verbose: print('  ... line {} of {}'.format(y, self.tile_height))#, end="\r")
                for x in range(self.tile_width):
                    value = img[y,x]#.getpixel((x,y))
                    borderReached = (x == self.tile_width-1)
                    if ( value != self.ignore_Color ): fullPoreArea +=1
                    if ( value != lastValue or borderReached ):
                        isBorder = ( lastChangedPos < 0 or borderReached )
                        length = x - lastChangedPos
                        if ( (value != lastValue and value == self.ignore_Color) or ( lastChangedPos > 0 and value != self.ignore_Color and isBorder and not ignoreBorder ) ): # if ignore_Color appears, a completed void line is detected
                            if ( length != self.tile_width and length > minLength ):
                                lineCount += 1
                                usedImageArea += length
                                result_row = self.process_result_row( file_index, diameter=length )
                                self.cld_df = self.cld_df.append(result_row, ignore_index=True)
                        if ( borderReached ):
                            lastChangedPos = -1
                            lastValue = self.ignore_Color
                        else:
                            lastChangedPos = x
                    lastValue = value

        elif ( direction == 'vertical' ):
            for x in range(self.tile_width):
                if x % self.progressFactor == 0 and verbose: print('  ... line {} of {}'.format(x, self.tile_width))#, end="\r")
                for y in range(self.tile_height):
                    value = img[y,x]#.getpixel((x,y))
                    borderReached = (y == self.tile_height-1)
                    if ( value != self.ignore_Color ): fullPoreArea +=1
                    if ( value != lastValue or borderReached ):
                        isBorder = ( lastChangedPos < 0 or borderReached )
                        length = y - lastChangedPos
                        if ( (value != lastValue and value == self.ignore_Color) or ( lastChangedPos > 0 and value != self.ignore_Color and isBorder and not ignoreBorder ) ): # if ignore_Color appears, a completed void line is detected
                            if ( length != self.tile_height and length > minLength ):
                                lineCount += 1
                                usedImageArea += length
                                result_row = self.process_result_row( file_index, diameter=length )
                                self.cld_df = self.cld_df.append(result_row, ignore_index=True)
                        if ( borderReached ):
                            lastChangedPos = -1
                            lastValue = self.ignore_Color
                        else:
                            lastChangedPos = y
                    lastValue = value
        else:
            print( "  unknown direction '{}'".format(direction) )

        if verbose:
            print( "   {} lines measured in {} ms".format(lineCount, int(round(time.time() * 1000)) - startTime), end='' )
            print( "   {:.2f} of {:.2f} area-% were taken into account".format(100/self.tile_area*usedImageArea, 100/self.tile_area*fullPoreArea) )

    def processCLD( self, img, processID=' ', process_position=None, verbose=False ):
        self.process_directional_CLD(img, 'horizontal', verbose=verbose)
        self.process_directional_CLD(img, 'vertical', verbose=verbose)

    def get_porespy_CLD(self, img):
        print('processing CLD using porespy')
        startTime = int(round(time.time() * 1000))
        chords_x = porespy.filters.apply_chords(img, axis=0, spacing=1, trim_edges=True)
        cld_x = porespy.metrics.chord_length_distribution(chords_x, bins=100, log=True)

        chords_y = porespy.filters.apply_chords(img, axis=1, spacing=1, trim_edges=True)
        cld_y = porespy.metrics.chord_length_distribution(chords_y, bins=100, log=True)
        #print(len(chords_x), len(chords_y))
        #print(chords_x, chords_y)
        print('finished in {} ms'.format(int(round(time.time() * 1000)) - startTime))
        #fig, (ax0, ax1) = plt.subplots( ncols=2, nrows=1, figsize=(20,10) )
        #ax0.bar(cld_x.bin_centers,cld_x.relfreq,width=cld_x.bin_widths,edgecolor='k')
        #ax1.bar(cld_y.bin_centers,cld_y.relfreq,width=cld_y.bin_widths,edgecolor='k')

    def get_histogram_bins(self, max_value, step, as_pixel=False):
        bin_count = round( max_value/self.scaling['x']+2 )

        if as_pixel:
            bins = [i for i in range(0,bin_count,step)]
        else:
            bins = [round(i*self.scaling['x'],4) for i in range(0,bin_count,step)]

        return bins

    def get_histogram_list(self, series, max_value=100, step=1, power=1):
        scale = self.scaling['x']
        #if power > 1:
            #bins = numpy.power(bins, power)
        bins = self.get_histogram_bins(max_value, step, as_pixel=True)

        histogram = numpy.histogram(list(series), bins=bins)

        return histogram[0], numpy.round( numpy.array(bins, dtype=numpy.float32)*scale,4)

    def clean_histogram(self, histogram, bins, power=1):
        if power > 1:
            pow_bins = []#bins
            maxval = numpy.amax(bins)
            cleaned_hist = []
            pos = 0
            deleted = 0
            for value in histogram:
                if value > 0:
                    pow_bins.append(bins[pos])
                    cleaned_hist.append(value)
                pos += 1
            #print(len(pow_bins), len(cleaned_hist))
            #print((pow_bins), (cleaned_hist))
        else:
            pow_bins = bins
            cleaned_hist = histogram
        return cleaned_hist, [bins[0]]+pow_bins

    def get_basic_values(self, column, df):
        col_max = df[column].max
        col_mean = df[column].mean
        col_median = df[column].median
        return col_max, col_mean, col_median

    def get_sum_list(self, value_list):
        sums = [0]
        for i in value_list:
            sums.append(i+sums[-1])
        return sums

    def get_values_above(self, series, value):
        return numpy.where( list(series) > value )

    def __init__(self, folder, file_name, file_extension, analyze_color_name='black', verbose=False):
        self.set_color_to_be_analyzed( analyze_color_name )

        self.folder = folder
        self.output_folder = self.folder + os.sep + 'CSV' + os.sep
        self.target_folder = os.path.abspath(os.path.join(self.folder, os.pardir))

        self.analyzed_files_DF = pd.DataFrame(columns = ['filename' , 'height', 'width'])
        dataframe_columns = ['file_index', 'diameter', 'area', 'surface', 'volume']
        self.psd_df = pd.DataFrame(columns = dataframe_columns)
        self.cld_df = pd.DataFrame(columns = dataframe_columns)

        self.get_file(file_name, file_extension, verbose=verbose)

        print('-'*50)
        self.processPSD(self.thresh_img, verbose=verbose)
        print('-'*50)
        self.get_porespy_CLD(self.thresh_img)
        print('-'*50)
        self.process_directional_CLD(self.thresh_img, 'horizontal', verbose=verbose)
        print('-'*50)
        self.process_directional_CLD(self.thresh_img, 'vertical', verbose=verbose)
        #self.processCLD(img, verbose=verbose)
        print('-'*50)

        # make sure the image_shuffle_count does not exceed the image count!
        #if self.image_shuffle_count > self.image_count:
        #    self.image_shuffle_count = self.image_count

        #self.reprocess_mean_and_stdev(verbose=verbose)


### actual program start
if __name__ == '__main__':
    programInfo()

    psd = singeFileProcess( verbose=True)

    print( "Script DONE!" )