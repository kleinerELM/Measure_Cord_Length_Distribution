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

from matplotlib import pyplot as plt
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

def File_Process(analyze_color_name='white',analyze_color_BGR=[0,0,255], min_cld_length=1, color_id=0, force_reprocessing=False, verbose=False):
    print( "Please select the image to analyse.")
    filepath = filedialog.askopenfilename( title='Please select the image to analyse.', filetypes=[("Tiff images", "*.tif")] )
    file_name, file_extension = os.path.splitext( filepath )
    file_name = os.path.basename(file_name)

    return size_distribution(os.path.dirname(filepath), file_name, file_extension, analyze_color_name=analyze_color_name, analyze_color_BGR=analyze_color_BGR, min_cld_length=min_cld_length, color_id=color_id, force_reprocessing=force_reprocessing, verbose=verbose)

def Folder_Process(analyze_color_name='white',analyze_color_BGR=[0,0,255], force_use_color=True, min_cld_length=1, force_reprocessing=False, verbose=False):
    print( "Please select the directory with the source image tiles.")#, end="\r" )
    workingDirectory = filedialog.askdirectory(title='Please select the image directory')

    count = 0
    fileList = []

    cld_df = None
    psd_df = None

    ## count files
    if os.path.isdir( workingDirectory ) :
        startTime = int(round(time.time() * 1000))
        for file in os.listdir(workingDirectory):
            if ( file.endswith(".tif") or file.endswith(".TIF")):
                fileList.append( file )
                count +=  1

        print( 'Found {} files...'.format(count) )

        for file in fileList:
            if ( file.endswith(".tif") or file.endswith(".TIF")):
                file_name, file_extension = os.path.splitext( file )
                sd = size_distribution(workingDirectory, file_name, file_extension, analyze_color_name=analyze_color_name, analyze_color_BGR=analyze_color_BGR, force_use_color=force_use_color, min_cld_length=min_cld_length, force_reprocessing=force_reprocessing, verbose=verbose)

                if not isinstance(cld_df, pd.DataFrame):
                    cld_df = sd.cld_df.copy()
                    psd_df = sd.psd_df.copy()
                    print(sd.cld_df)
                else:
                    cld_df = pd.concat([cld_df, sd.cld_df], ignore_index=True)
                    psd_df = pd.concat([psd_df, sd.psd_df], ignore_index=True)
                break
        cld_df.to_csv(workingDirectory + 'CLD_complete.csv')
        psd_df.to_csv(workingDirectory + 'PSD_complete.csv')
        print('finished processing of {} files in {:.2f} s'.format(count, (int(round(time.time() * 1000)) - startTime)/1000))
        print('found {} pores and measured {} lines'.format(len(psd_df), len(cld_df)))

#https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html
def smooth_histogram(x, window_len=7, window='hanning'):
    if window_len<3:
        return x

    if window_len % 2 == 0:
        window_len += 1
        print("Warning: 'window_len' has to be an odd number! Changing window size to {}".format(window_len))

    if isinstance(x, list):
        x = numpy.array(x)

    if x.ndim != 1:
        raise ValueError("smooth() only accepts 1 dimension arrays or lists.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        window = 'hanning'
        print("Warning: 'window' has to be one of the following: of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'. Switching to '{}'.".format(window))

    prepend_x = x[int((window_len-1)/2):0:-1] #x[window_len-1:0:-1]
    append_x = x[-2:-int(2+(window_len-1)/2):-1] #x[-2:-window_len-1
    s = numpy.r_[prepend_x, x, append_x]

    if window == 'flat': #moving average
        w = numpy.ones(window_len, 'd')
    else:
        w = eval('numpy.' + window + '(window_len)')

    y = numpy.convolve(w/w.sum(), s, mode='valid')

    return y

def get_bin_center(bins):
    bin_center = []
    for l, r in zip(bins[1:], bins[:-1]):
        bin_center.append((r-l)/2+l)
    return bin_center

class size_distribution():
    coreCount = multiprocessing.cpu_count()
    processCount = (coreCount - 1) if coreCount > 1 else 1

    progressFactor = 500
    image_count = 0
    tile_width = 0
    tile_height = 0
    tile_area = 0
    scaling = es.getEmptyScaling()

    channel_count = 1
    black = 0
    white = 255
    available_colors = []
    force_use_color = False

    min_cld_length = 1
    cld_result = []
    psd_result = []

    histograms_CLD = []
    histograms_PSD = []
    #result_dataframe_columns = ['file_index', 'diameter', 'area', 'surface', 'volume']

    folder = ''
    output_folder = ''

    def process_histograms(self):
        column = 'diameter'
        max_value = 10000
        hist, bins = self.get_histogram_list( self.cld_result[column], max_value)
        """
        first_value_id = next(i for i,v in enumerate(hist) if v > 0)
        x_min = bins[first_value_id]-psd.scaling['x']
        if column=='area':
            x_axis_type='log'
            x_min = psd.getPoreArea( x_min )
            bins = psd.getPoreArea( bins )
        elif column=='volume':
            x_axis_type='log'
            x_min = psd.getPoreVolume( x_min )
            bins = psd.getPoreVolume( bins )
        elif column=='surface':
            x_axis_type='log'
            x_min = psd.getPoreSurface( x_min )
            bins = psd.getPoreSurface( bins )
        else:
            x_axis_type='linear'
            x_min = bins[first_value_id]-psd.scaling['x']
        """

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

    def getPoreVolume( self, diameter=None ):
        radius = diameter/2
        return 4/3*(math.pi*(radius**3))

    def getPoreSurface( self, diameter=None ):
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
            print('size: {:.2f} x {:.2f} {} / {} x {} px '.format(self.get_scaled_width(),self.get_scaled_height(),self.scaling['unit'],width,height))
            return True
        else:
            if self.tile_width == width and self.tile_height == height:
                return True
            else:
                print( 'Tile sizes do not match! (w: {} != {} | h: {} != {})'.format(width, self.tile_width, height, self.tile_height) )
                exit()
        return False

    # plot the raw image next to the thresholded image
    def plot_images(self):
        fig = plt.figure(figsize=(12,8), dpi= 100)
        fig.add_subplot(1, 2, 1)

        plot_axis_scaling = [0,self.get_scaled_width(),0,self.get_scaled_height()]

        if self.channel_count == 3:
            plt.imshow( numpy.array(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)), extent=plot_axis_scaling );
        else:
            plt.imshow( self.img, extent=plot_axis_scaling, cmap='gray' );
        plt.title('original: {} in {}'.format(self.filename, self.scaling['unit']), fontsize=8);

        fig.add_subplot(1, 2, 2)
        plt.imshow( self.thresh_img, extent=plot_axis_scaling, cmap='gray' );
        plt.title('binarized: {} in {}'.format(self.filename, self.scaling['unit']), fontsize=8);

    def get_available_colors(self):
        if len(self.available_colors) < 1:
            for x in self.img[0]:
                if list(x) not in self.available_colors:
                    self.available_colors.append(list(x))
        return self.available_colors

    def get_image_channel_count(self):
        if len(self.img.shape) >= 2:
            self.channel_count = self.img.shape[-1]
        return self.channel_count

    def load_binary_image(self, color=None):
        self.scaling = es.autodetectScaling( self.filename, self.folder )
        self.img = cv2.imread( self.folder + os.sep + self.filename, -1 )
        if self.img is None:
            print( 'Error loading {}'.format(self.filename))
            exit()

        channel_count = self.get_image_channel_count()

        if channel_count == 3:
            #self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
            self.load_binary_image_from_color(color=self.analyze_color_BGR, color_id=self.color_id)
        else:
            #self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
            self.load_binary_image_from_binary()

    def load_binary_image_from_binary(self):
        #binarizes an image
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        min_val = self.black #int( (self.ignore_Color+1)/2 ) if self.ignore_Color > self.analyze_Color else self.analyze_Color
        max_val = int( (self.white+1)/2 ) #int( (self.analyze_Color+1)/2 ) if self.analyze_Color > self.ignore_Color else self.ignore_Color
        _, self.thresh_img = cv2.threshold(self.img, min_val, max_val, cv2.THRESH_BINARY)

    # Directly define a color to select or define a color_id.
    # The color id is not constant! The color order is defined by appearance within the image!
    def load_binary_image_from_color(self, color=None, color_id=0):
        available_colors = self.get_available_colors()
        if not color==None:
            color_found = False
            for avaiable_color in available_colors:
                if list(avaiable_color) == list(color):
                    color_found = True
            if color_found or self.force_use_color:
                color = numpy.array(color)
            else:
                print('  Color R{} G{} B{} does not exist in the loaded image! Available colors:'.format(color[0], color[1], color[2]))
                print(available_colors)
                color = numpy.array(available_colors[color_id])
        else:
            print('  Using color B{} G{} R{}! Available colors:'.format(available_colors[color_id][0], available_colors[color_id][1], available_colors[color_id][2]))
            print(available_colors)
            color = numpy.array(available_colors[color_id])

        #binarizes an image
        min_val = color
        max_val = color
        self.thresh_img = cv2.inRange(self.img, min_val, max_val)

    def get_file(self, file_name, file_extension, verbose=False ):
        processID = ' '
        self.filename = file_name + file_extension
        self.load_binary_image()

        if verbose: print( 'Analysing {}'.format(self.filename) )

        height, width = self.img.shape[:2]

        if self.check_tile_dimension(height, width):
            new_row = {'filename':file_name, 'height':height, 'width':width }
        self.analyzed_files_DF = self.analyzed_files_DF.append(new_row, ignore_index=True)

    def get_column_unit(self, column):
        power = 1
        unit = self.scaling['unit']
        if column == 'area' or column == 'surface':
            unit += '²'
            power = 2
        elif column == 'volume':
            unit += '³'
            power = 3
        return unit, power

    def process_result_row( self, file_index=0, diameter=0, area=0 ):
        result_row = False
        if area > 0 or diameter > 0:
            if area > 0:
                diameter = self.getPoreDiameter(area)
            else:
                area = self.getPoreArea(diameter)
            result_row = {
                #'file_index':   file_index,
                'diameter':     diameter,
                'area' :        area,
                'surface':      self.getPoreSurface(diameter=diameter),
                'volume':       self.getPoreVolume(diameter=diameter)
            }
        return result_row

    def processPSD( self, img, processID = ' ', start_pos = None, end_pos = None, verbose=False ):
        processID = ' '

        contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        startTime = int(round(time.time() * 1000))
        processCount = self.processCount if self.processCount <=3 else 3
        pool = multiprocessing.Pool(processCount)
        start_pos = 0
        start_pos = 0
        for pos in range(processCount):
            end_pos = int(len(contours)/processCount * (pos+1))
            pool.apply_async(self.processPSD_subprocess, args=(contours, ' PID{:02d} '.format(pos+1) , start_pos, end_pos, verbose), callback = self.append_PSD_result)
            start_pos = end_pos
        pool.close()
        pool.join()

        self.psd_df = pd.DataFrame(self.psd_result)
        if verbose:
            print( '{}Analysed particles: {}, ignored 0-values: {}, finished in {} ms'.format(processID, len(self.psd_result), len(contours)-len(self.psd_result), int(round(time.time() * 1000)) - startTime) )

    def processPSD_subprocess(self, contours, processID, start_pos, end_pos, verbose=False):
        #if verbose: print( '{}processing pore size distribution (pore {}-{})'.format(processID, start_pos, end_pos) )
        #file_index = int(len(self.analyzed_files_DF)-1)
        result_list = []
        for i in range(start_pos, end_pos):
            result_row = self.process_result_row( area=cv2.contourArea(contours[i]) )
            if result_row != False:
                result_list.append( result_row )
        return result_list

    def process_directional_CLD( self, img, direction, processID = ' ', start_pos = None, end_pos = None, verbose=False):
        #file_index = int(len(self.analyzed_files_DF)-1)

        ignoreBorder = True
        minLength = 1

        #process variables
        lineCount = 0
        lastValue = -1        # var to save the last value
        lastChangedPos = -1   # var to save the last position
        usedImageArea = 0
        fullPoreArea = 0

        if ( direction == 'horizontal' or direction == 'vertical' ):
            result = []

            max_a = self.tile_height if direction == 'horizontal' else self.tile_width

            if end_pos == None: end_pos = max_a
            if start_pos == None: start_pos = 0
            range_a = range(start_pos, end_pos)

            max_b = self.tile_width if direction == 'horizontal' else self.tile_height
            #if verbose: print( '  {}processing {} cord length distribution (lines {}-{})'.format(processID, direction, start_pos, end_pos) )

            if verbose: startTime = int(round(time.time() * 1000))
            pixel_cnt = 0
            for a in range_a:
                #if a % self.progressFactor == 0 and verbose: print('{} line {} of {}'.format(processID, a, max_a))#, end="\r")
                for b in range(max_b):
                    x = b if direction == 'horizontal' else a
                    y = a if direction == 'horizontal' else b
                    value = img[y,x]

                    borderReached = (b == max_b-1)
                    if ( value != self.ignore_Color ): fullPoreArea +=1
                    if ( value != lastValue or borderReached ):
                        isBorder = ( lastChangedPos < 0 or borderReached )
                        length = b - lastChangedPos
                        # if ignore_Color appears, a completed void line is detected
                        if ( (value != lastValue and value == self.ignore_Color)
                              or ( lastChangedPos > 0 and value != self.ignore_Color and isBorder and not ignoreBorder ) ):

                            if ( length < max_b and length > self.min_cld_length ):
                                lineCount += 1
                                usedImageArea += length
                                result_row = self.process_result_row( diameter=length )
                                result.append(result_row)
                                #self.cld_df = self.cld_df.append(result_row, ignore_index=True)
                        if ( borderReached ):
                            lastChangedPos = -1
                            lastValue = self.ignore_Color
                        else:
                            lastChangedPos = b
                    lastValue = value

            if verbose:
                print( "  {} {} lines measured. {:.2f} of {:.2f} area-% were taken into account".format(processID, lineCount, 100/self.tile_area*usedImageArea, 100/self.tile_area*fullPoreArea))#, end='' )
            return result
        else:
            print( "  unknown direction '{}'".format(direction) )

    def append_CLD_result(self, result_list):
        self.cld_result += result_list

    def append_PSD_result(self, result_list):
        self.psd_result += result_list

    def merge_result(self, df, result_list):
        result_df = pd.DataFrame(result_list)
        df.append(result_df, ignore_index=True)

    def get_porespy_CLD(self, img):
        print('processing CLD using porespy')
        startTime = int(round(time.time() * 1000))
        chords_x = porespy.filters.apply_chords(img, axis=0, spacing=1, trim_edges=True)
        cld_x = porespy.metrics.chord_length_distribution(chords_x, bins=100, log=True)

        chords_y = porespy.filters.apply_chords(img, axis=1, spacing=1, trim_edges=True)
        cld_y = porespy.metrics.chord_length_distribution(chords_y, bins=100, log=True)

        print('finished in {} ms'.format(int(round(time.time() * 1000)) - startTime))

    def get_histogram_bins(self, max_value, step, as_pixel=False, verbose=False):
        bin_count = round( max_value/self.scaling['x']+2 )

        if as_pixel:
            if verbose: print('  bins are scaled as px!')
            bins = [i for i in range(0,bin_count,step)]
        else:
            bins = [round(i*self.scaling['x'],4) for i in range(0,bin_count,step)]

        return bins

    def get_histogram_list(self, df, column, max_value=100, step=1, power=1):

        px_bins = self.get_histogram_bins(max_value, step, as_pixel=True)

        # scale the bins
        bins = numpy.round( numpy.array(px_bins, dtype=numpy.float32)*self.scaling['x'], 4)
        if column=='area':
            bins = self.getPoreArea( bins )
        elif column=='volume':
            bins = self.getPoreVolume( bins )
        elif column=='surface':
            bins = self.getPoreSurface( bins )

        histogram = numpy.histogram(list(df[column]), bins=bins)

        return histogram[0], bins

    def get_basic_values(self, column, df):
        max_vals = df.max(axis=0)
        mean_vals = df.mean(axis=0)
        median_vals = df.median(axis=0)
        return max_vals[column], mean_vals[column], median_vals[column]

    def get_sum_list(self, value_list):
        sums = [0]
        for i in value_list:
            sums.append(i+sums[-1])
        return sums

    def get_values_above(self, series, value):
        return numpy.where( list(series) > value )

    def __init__(
            self,
            folder,                     # folder name without trailing path seperator / or \
            file_name,                  # file name without file extension
            file_extension,             # file extension including the dot ".TIF"
            analyze_color_name='white', # color name of the binarized image, whi is used as phase to analyse (black or white)
            analyze_color_BGR=None,     # if a segmentation results in a multiphase image, a bgr color can be defined to create a binary image
            force_use_color=False,      # use the BGR color even if it does not exist
            color_id=0,                 # if the id of the color is known instead of the bgr color (usable to identify color values)
            min_cld_length=1,           # minimal length of a pore for the chord length distibution
            force_reprocessing=False,   # force the reprocessing, even if a CSV of a PSD or CLD exists
            verbose=False               # print some debug information
        ):
        self.cld_result = []
        self.psd_result = []

        self.set_color_to_be_analyzed( analyze_color_name )

        self.folder = folder
        self.output_folder = self.folder + os.sep + 'CSV' + os.sep
        if not os.path.isdir( self.output_folder ): os.mkdir(self.output_folder)
        self.target_folder = os.path.abspath(os.path.join(self.folder, os.pardir))

        self.analyzed_files_DF = pd.DataFrame(columns = ['filename' , 'height', 'width'])

        psd_csv_path = self.output_folder + file_name + '_psd.csv'
        cld_csv_path = self.output_folder + file_name + '_cld.csv'

        print('-'*50)
        self.color_id = color_id
        self.analyze_color_BGR = analyze_color_BGR
        self.force_use_color = force_use_color
        self.get_file(file_name, file_extension, verbose=verbose)

        # pore size distribution
        if not os.path.isfile( psd_csv_path ) or force_reprocessing:
            #self.psd_df = pd.DataFrame(columns = self.result_dataframe_columns)
            self.processPSD(self.thresh_img, verbose=verbose)
            self.psd_df.to_csv(psd_csv_path)
        else:
            self.psd_df = pd.read_csv( psd_csv_path, index_col=0 )
            print('loaded psd csv')

        print('-'*50)
        # chord length distribution
        if not os.path.isfile( cld_csv_path ) or force_reprocessing:
            self.min_cld_length = min_cld_length
            #self.cld_df = pd.DataFrame(columns = self.result_dataframe_columns)

            # There is a time penalty using to many threads. The loading time of the images and libaries seem to hav a huge imact on the multithreading performance.
            # Therefore, the horizontal and vertical CLD is calculated parallel in few threads instead a single direction on all available threads!
            # Up to two cores will idle in this time.
            processes_per_direction = int(self.processCount/2)
            process_range = range(processes_per_direction) if processes_per_direction > 1 else range(1)
            pool_size = processes_per_direction*2 if processes_per_direction > 1 else 1

            print( 'processing cld using {} threads'.format(pool_size) )
            startTime = int(round(time.time() * 1000))
            pool = multiprocessing.Pool( pool_size )
            h_start_pos = 0
            v_start_pos = 0
            pid = 0
            for pos in process_range:
                pid += 1
                h_end_pos = int(self.tile_height/processes_per_direction * (pos+1))
                pool.apply_async(self.process_directional_CLD, args=(self.thresh_img, 'horizontal', ' PID{:02d} '.format(pid) , h_start_pos, h_end_pos, verbose), callback = self.append_CLD_result)
                h_start_pos = h_end_pos

                pid += 1
                v_end_pos = int(self.tile_width/processes_per_direction * (pos+1))
                pool.apply_async(self.process_directional_CLD, args=(self.thresh_img, 'vertical', ' PID{:02d} '.format(pid) , v_start_pos, v_end_pos, verbose), callback = self.append_CLD_result)
                v_start_pos = v_end_pos
            pool.close()
            pool.join()
            print( 'finished processing {} lines in {} ms'.format(len(self.cld_result), int(round(time.time() * 1000)) - startTime) )

            print('-'*50)

            self.cld_df = pd.DataFrame(self.cld_result)

            #self.process_directional_CLD(self.thresh_img, 'horizontal', verbose=verbose)
            #print('-'*50)
            #self.process_directional_CLD(self.thresh_img, 'vertical', verbose=verbose)
            #print('-'*50)

            self.cld_df.to_csv(cld_csv_path)
        else:
            self.cld_df = pd.read_csv( cld_csv_path, index_col=0 )
            print('loaded cld csv')

        if self.scaling['unit'] == 'px': print('scaled as PX!!!')

        #self.get_porespy_CLD(self.thresh_img)
        #print('-'*50)

### actual program start
if __name__ == '__main__':
    programInfo()

    #psd = singeFileProcess( verbose=True)
    Folder_Process(force_reprocessing=False, verbose=True)

    print( "Script DONE!" )