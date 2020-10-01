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

def singeFileProcess(verbose=False):
    print( "Please select the directory with the source image tiles.", end="\r" )
    filepath = filedialog.askopenfilename( title='Please select the reference image', filetypes=[("Tiff images", "*.tif;*.tiff")] )
    file_name, file_extension = os.path.splitext( filepath )
    file_name = os.path.basename(file_name)

    return size_distribution(os.path.dirname(filepath), file_name, file_extension, verbose=verbose)

class size_distribution():

    image_count = 0
    tile_width = 0
    tile_height = 0

    scaling = es.getEmptyScaling()
    
    materialColor = 255
    poreColor = 0

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

    def check_tile_dimension(self, height, width):
        if self.tile_width + self.tile_height == 0:
            self.tile_width = width
            self.tile_height = height
            return True
        else:
            if self.tile_width == width and self.tile_height == height:
                return True
            else:
                print( 'Tile sizes do not match! (w: {} != {} | h: {} != {})'.format(width, self.tile_width, height, self.tile_height) )
                exit()
        return False

    def load_binary_image(self, filename):
        self.scaling = es.autodetectScaling( filename, self.folder )
        img = cv2.imread( self.folder + os.sep + filename, cv2.IMREAD_GRAYSCALE )

        height, width = img.shape[:2]

        if img is None:
            print( 'Error loading {}'.format(filename))
            exit()

        #binarizes an image
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        min_val = int( (self.materialColor+1)/2 ) if self.materialColor > self.poreColor else self.poreColor
        max_val = int( (self.poreColor+1)/2 ) if self.poreColor > self.materialColor else self.materialColor
        _, img = cv2.threshold(img, min_val, max_val, cv2.THRESH_BINARY)
        return img

    def processPSD( self, file_name, file_extension, process_position=None, verbose=False ):
        processID = ' '
        if process_position != None: processID = " #{}: ".format(process_position)

        img = self.load_binary_image(file_name + file_extension)

        print( '{}Analysing {}'.format(processID, file_name + file_extension) )

        height, width = img.shape[:2]
        
        if self.check_tile_dimension(height, width):
            new_row = {'filename':file_name, 'height':height, 'width':width }

        contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        self.analyzed_files_DF = self.analyzed_files_DF.append(new_row, ignore_index=True)
        #print(len(self.analyzed_files_DF))
        #print(self.analyzed_files_DF.iloc[-1])
        areas = []
        contourCount = len(contours)
        file_index = int(len(self.analyzed_files_DF)-1)
        emptyArea = 0
        progressFactor = 500
        for i in range(0, contourCount):
            area = cv2.contourArea(contours[i])
            if area > 0:
                diameter = self.getPoreDiameter(area)
                result_row = {  
                    'file_index':   file_index,
                    'diameter':     diameter,
                    'area' :        area,
                    'surface':      self.getPoreSurface(diameter=diameter),
                    'volume':       self.getPoreVolume(diameter=diameter)
                }
                self.psd_df = self.psd_df.append(result_row, ignore_index=True)
                if i % progressFactor == 0: print( "{}...processing particle #{}/{}".format(processID, i, contourCount ))#, end="\r")
            else:
                emptyArea += 1
        print( '{}Analysed particles: {}, ignored 0-values: {}'.format(processID, contourCount-emptyArea, emptyArea) )
        #self.psd_df.to_csv( self.output_folder + file_name + '_pores.csv', index=False )
        print(self.psd_df)

    
    def processDirectionalCLD( img, directory, direction ):
        pass
        """
        global materialColor
        global ignoreBorder
        global minLength

        lineCount = 0
        lastValue = -1      # var to save the last value
        lastChangedPos = -1   # var to save the last position
        width, height = im.size
        imageArea = width * height
        usedImageArea = 0
        fullPoreArea = 0
        pixelWidth = self.scaling['x'] # TODO Check unit?
        pixelHeight = self.scaling['y'] # TODO Check unit?

        minLength = 1
        resultCSV = ''
        print( '  processing {} cord length distribution'.format(direction) )#, end='' )
        startTime = int(round(time.time() * 1000))
        if ( direction == 'horizontal' ):
            for y in range(height):
                if ( y % 100 == 0 ): print('  ... line {} of {}'.format(y, height), end="\r")
                for x in range(width):
                    value = im.getpixel((x,y))
                    borderReached = (x == width-1)
                    if ( value != materialColor ): fullPoreArea +=1
                    if ( value != lastValue or borderReached ):
                        isBorder = ( lastChangedPos < 0 or borderReached )
                        length = x - lastChangedPos
                        if ( (value != lastValue and value == materialColor) or ( lastChangedPos > 0 and value != materialColor and isBorder and not ignoreBorder ) ): # if materialColor appears, a completed void line is detected
                            if ( length != width and length > minLength ):
                                lineCount += 1
                                usedImageArea += length
                                #resultCSV += str(x) +"	" + str(y)  +"	" + str(lastChangedPos)+"	" + str( lineCount ) + "	" + str( pixelWidth * length ) + "\n"#+ "	" + str( length/imageArea ) + "\n"
                                resultCSV += str( lineCount ) + "	" + str( pixelWidth * length ) + "\n"#+ "	" + str( length/imageArea ) + "\n"
                        if ( borderReached ):
                            lastChangedPos = -1
                            lastValue = materialColor
                        else:
                            lastChangedPos = x
                    lastValue = value

        elif ( direction == 'vertical' ):
            for x in range(width):
                if ( x % 100 == 0  ): print('  ... line {} of {}'.format(x, width), end="\r")
                for y in range(height):
                    value = im.getpixel((x,y))
                    borderReached = (y == height-1)
                    if ( value != materialColor ): fullPoreArea +=1
                    if ( value != lastValue or borderReached ):
                        isBorder = ( lastChangedPos < 0 or borderReached )
                        length = y - lastChangedPos
                        if ( (value != lastValue and value == materialColor) or ( lastChangedPos > 0 and value != materialColor and isBorder and not ignoreBorder ) ): # if materialColor appears, a completed void line is detected
                            if ( length != width and length > minLength ):
                                lineCount += 1
                                usedImageArea += length
                                resultCSV += str( lineCount ) + "	" + str( pixelHeight * length ) + "\n"
                        if ( borderReached ):
                            lastChangedPos = -1
                            lastValue = materialColor
                        else:
                            lastChangedPos = y
                    lastValue = value
        else:
            print( "  unknown direction '" + str( direction )+ "'" )
        if ( resultCSV != '' ):
            print( "   {} lines measured in {} ms".format(lineCount, int(round(time.time() * 1000)) - startTime), end='' )
            print( "   {:.2f} of {:.2f} area-% were taken into account".format(100/imageArea*usedImageArea, 100/imageArea*fullPoreArea) )
            #print( str( usedImageArea ) + '  ' + str( fullPoreArea ) )
            headerLine = "lineCount" + "	" + "length [nm]" + "\n"#+ "	" + "volume fraction" + "\n"
            resultFile = open(directory + os.sep + outputDirName + os.sep + filename + "." + direction + ".csv","w") 
            resultFile.write( headerLine + resultCSV ) 
            resultFile.close() #to change file access modes 
        return resultCSV
        """

    def processCLD( self, file_name, file_extension, process_position=None, verbose=False ):
        processID = ' '
        if process_position != None: processID = " #{}: ".format(process_position)

        img = self.load_binary_image(file_name + file_extension)
        """global globMaskPagePos
        global outputDirName
        global sumResultCSV
        
        scaling = es.autodetectScaling( filename, directory )
        pageCnt = 0
        
        im = Image.open( directory + os.sep + filename )
        # check page count in image
        for i in enumerate(ImageSequence.Iterator(im)):
            pageCnt +=1
        if ( pageCnt - 1 < globMaskPagePos ) :
            print( '  WARNING: The image has only {} page(s)! Trying to use page 1 as mask.'.format(pageCnt))
            maskPagePos = 0
        else:
            maskPagePos = globMaskPagePos

        # run analysis
        for i, page in enumerate(ImageSequence.Iterator(im)):
            if ( i == maskPagePos ): 
                sumResultCSV += processDirectionalCLD(im, scaling, directory, 'horizontal')
                sumResultCSV += processDirectionalCLD(im, scaling, directory, 'vertical')
                
                img = imageio.imread( directory + os.sep + filename )
                chords_x = ps.filters.apply_chords(img, axis=0, spacing=1, trim_edges=True)
                chords_y = ps.filters.apply_chords(img, axis=1, spacing=1, trim_edges=True)
                #chords_z = ps.filters.apply_chords(img, axis=2, spacing=1, trim_edges=True)
                
                cld_x = ps.metrics.chord_length_distribution( chords_x, bins=100, log=True )
                cld_y = ps.metrics.chord_length_distribution( chords_y, bins=100, log=True )
                #cld_z = ps.metrics.chord_length_distribution( chords_z, bins=100, log=True )

                fig, (ax0, ax1) = plt.subplots( ncols=2, nrows=1, figsize=(20,10) )
                ax0.bar(cld_x.bin_centers,cld_x.relfreq,width=cld_x.bin_widths,edgecolor='k')
                ax1.bar(cld_y.bin_centers,cld_y.relfreq,width=cld_y.bin_widths,edgecolor='k')
                #ax2.bar(cld_z.bin_centers,cld_z.relfreq,width=cld_z.bin_widths,edgecolor='k')
                
                plt.savefig(directory + os.sep + filename + 'line_plot.svg')  
        im.close()
        """
        print()

    def __init__(self, folder, file_name, file_extension, verbose=False ):
        self.folder = folder
        self.output_folder = self.folder + os.sep + 'CSV' + os.sep
        self.target_folder = os.path.abspath(os.path.join(self.folder, os.pardir))
        self.analyzed_files_DF = pd.DataFrame(columns = ['filename' , 'height', 'width'])
        self.psd_df = pd.DataFrame(columns = ['file_index', 'diameter', 'area', 'surface', 'volume'])
        self.processPSD(file_name, file_extension)
        # make sure the image_shuffle_count does not exceed the image count!
        #if self.image_shuffle_count > self.image_count:
        #    self.image_shuffle_count = self.image_count

        #self.reprocess_mean_and_stdev(verbose=verbose)

    
### actual program start
if __name__ == '__main__':
    programInfo()

    psd = singeFileProcess( verbose=True)
    
    print( "Script DONE!" )