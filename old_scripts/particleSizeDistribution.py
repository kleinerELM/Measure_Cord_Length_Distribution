import csv
import os, sys, getopt
import subprocess
import math
import tkinter as tk
import mmap
from tkinter import filedialog
from subprocess import check_output
import time
import multiprocessing

import cv2
import numpy

def programInfo():
    print("#########################################################")
    print("# A Script to process the Particle Size Distribution    #")
    print("#                                                       #")
    print("# © 2020 Florian Kleiner                                #")
    print("#   Bauhaus-Universität Weimar                          #")
    print("#   F. A. Finger-Institut für Baustoffkunde             #")
    print("#                                                       #")
    print("#########################################################")
    print()

#### process given command line arguments
def processArguments():
    global outputDirName
    global showDebuggingOutput
    global pixelWidth
    global ignoreBorder
    global minLength
    global materialColor
    global maskPagePos
    argv = sys.argv[1:]
    usage = sys.argv[0] + " [-h] [-o] [-s] [-p] [-d]"
    try:
        opts, args = getopt.getopt(argv,"hs:p:o:d",[])
    except getopt.GetoptError:
        print( usage )
    for opt, arg in opts:
        if opt == '-h':
            print( 'usage: ' + usage )
            print( '-h,                  : show this help' )
            print( '-o,                  : setting output directory name [{}]'.format(outputDirName) )
            print( '-s,                  : set pixel size [{} nm per pixel]'.format(pixelWidth) )
            print( '-p,                  : set page position of the mask in a TIFF [{}]'.format(maskPagePos + 1) )
            print( '-d                   : show debug output' )
            print( '' )
            sys.exit()
        elif opt in ("-o"):
            outputDirName = arg
            print( 'changed output directory to ' + outputDirName )
        elif opt in ("-s"):
            pixelWidth = float( arg )
        elif opt in ("-p"):
            maskPagePos = int( arg ) -1
            if ( maskPagePos < 0 ): 
                maskPagePos = 0
        elif opt in ("-d"):
            print( 'show debugging output' )
            showDebuggingOutput = True
    # print information for the main settings:
    print( 'Settings:')
    print( ' - pixel size is set to ' + str( pixelWidth ) + ' nm per pixel')
    if ( ignoreBorder ) : print( ' - areas touching a border will be ignored' )
    else : print( ' - areas touching a border will be included (may be flawed!)' )
    print( ' - ignoring areas smaller than ' + str( minLength ) + ' pixel')
    if ( materialColor == 0 ) : colorName = 'white'
    else: colorName = 'black'
    print( ' - calculating the Cord Length Distribution of ' + colorName + ' areas')
    if ( maskPagePos == 0 ) : print( ' - expecting a normal b/w TIFF or a multi page TIFF, where the mask is on page 1' )
    else : print( ' - expecting a multi page Tiff where the mask is on page ' + str( maskPagePos + 1 ) )
    print( '' )

def getPoreDiameter( area ):
    return ( math.sqrt( area /math.pi )*2 )

def getPoreVolume( area ):
    radius = getPoreDiameter( area )/2
    return 4/3*(math.pi*(radius**3))

def getPoreSurface( area ):
    radius = getPoreDiameter( area )/2
    return (4*math.pi*(radius**2))

def writeCSV( workingDirectory, outputDirName, filename, sumResultCSV, CSVdelimiter, showDebuggingOutput ):
    csvFileName = "." + os.sep + outputDirName + os.sep + "sumPSD" + filename + ".csv"
    if ( showDebuggingOutput ): print( " saving " + csvFileName )
    if ( sumResultCSV != '' ):
        headerLine = "poreNr" + CSVdelimiter + "area [nm^2]" + CSVdelimiter + "diameter [nm]" + CSVdelimiter + "volume [nm^3]" + CSVdelimiter + "surface [nm^2]\n"
        resultFile = open(workingDirectory + os.sep + csvFileName,"w") 
        resultFile.write( headerLine + sumResultCSV ) 
        resultFile.close() #to change file access modes 

def processPSD( workingDirectory, filename, position, maskPagePos, outputDirName, CSVdelimiter, showDebuggingOutput ):
    sumResultCSV = ''
    processID = " #{}: ".format(position)

    frame = cv2.imread( workingDirectory + os.sep + filename )

    if frame is None:
        print( '{}Error loading {}'.format(processID, filename))
        exit()
    else:
        print( '{}Analysing {}'.format(processID, filename) )

    #binarizes an image
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, frame = cv2.threshold(frame, 127, 255, cv2.THRESH_BINARY)
    #if ( showDebuggingOutput ) : print( "Trying to find contours" )
    #frame, contours, hierarchy = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    areas = []
    contourCount = len(contours)
    for i in range(0, contourCount):
        areas.append(cv2.contourArea(contours[i]))

    diameters = []
    cleanedAreas = []
    volumes = []
    surfaces = []
    pos = 0
    progressFactor = 50000
    for i in range(0, contourCount):
        if ( areas[i] > 0 ):
            diameters.append( getPoreDiameter( areas[i] ) )
            cleanedAreas.append( areas[i] )
            volumes.append( getPoreVolume( areas[i] ) )
            surfaces.append( getPoreSurface( areas[i] ) )
            if ( showDebuggingOutput ) : 
                print( '{}Area {}:{}'.format(processID, pos+1, areas[i]) )
            else: 
                if ( pos % progressFactor == 0 ): print( "{}...processing particle #{}/{}".format(processID, pos, contourCount ))#, end="\r")
            sumResultCSV += str( pos+1 ) + CSVdelimiter + str( cleanedAreas[pos] ) + CSVdelimiter + str( diameters[pos] ) + CSVdelimiter + str( volumes[pos] ) + CSVdelimiter + str( surfaces[pos] ) + "\n"
            pos += 1
    print( '{}Analysed particles: {}, ignored 0-values: {}'.format(processID, pos, contourCount-pos) )
    
    writeCSV( workingDirectory, outputDirName, filename, sumResultCSV, CSVdelimiter, showDebuggingOutput )
    
    return [diameters, cleanedAreas, volumes, surfaces]

result_list = []
def log_result(result):
    global result_list
    # This is called whenever foo_pool(i) returns a result.
    # result_list is modified only by the main process, not the pool workers.
    result_list.append(result)

if __name__ == '__main__':

    # remove root windows
    root = tk.Tk()
    root.withdraw()

    #### main definitions
    home_dir = os.path.dirname(os.path.realpath(__file__))
    coreCount = multiprocessing.cpu_count()
    processCount = (coreCount - 1) if coreCount > 1 else 1

    outputDirName = "resultsPSD"
    showDebuggingOutput = False

    materialColor = 255#0
    poreColor = 0#255
    maskPagePos = 1     # second image in tiff stack is the masked image
    ignoreBorder = True#False#True
    minLength = 1
    sumResultCSV = ''
    pixelWidth = 2.9141 #nm
    CSVdelimiter = "	" # tab

    ### actual program start
    programInfo()
    processArguments()
    workingDirectory = filedialog.askdirectory(title='Please select the image / working directory')
    if ( showDebuggingOutput ) : 
        print( 'Found {} CPU cores. Using max. {} processes at once.'.format(coreCount, processCount) )
        print( "I am living in '{}'".format(home_dir) )
        print( "Selected working directory: {}".format( workingDirectory ), end='\n\n' )

    count = 0
    fileList = []
    ## count files
    if os.path.isdir( workingDirectory ) :
        for file in os.listdir(workingDirectory):
            if ( file.endswith(".tif") or file.endswith(".TIF")):
                fileList.append( file )
                count +=  1

    print( "{} Tiffs found!".format(count) )
    ## run actual code
    if ( count > 0 ):
        if not os.path.exists( workingDirectory + os.sep + outputDirName ):
            os.makedirs( workingDirectory + os.sep + outputDirName )
        pool = multiprocessing.Pool(processCount)
        pos = 0
        for file in fileList:
            filename = os.fsdecode(file)
            pos += 1
            pool.apply_async(processPSD, args=( workingDirectory, filename, pos, maskPagePos, outputDirName, CSVdelimiter, showDebuggingOutput ), callback = log_result)

        pool.close()
        pool.join()


        print( "generating full csv" )
        sumResultCSV = ''
        pos = 0
        for result in result_list:
            sumResultCSV += str( pos+1 )
            for item in result:
                sumResultCSV += CSVdelimiter + str( item )
            sumResultCSV += "\n"
            pos += 1

        writeCSV( workingDirectory, outputDirName, '', sumResultCSV, CSVdelimiter, showDebuggingOutput )

    print( "Results can be found in directory:" )
    print( "  {}/{}/\n".format(workingDirectory, outputDirName) )
    print( "Script DONE!" )