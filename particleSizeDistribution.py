import csv
import os, sys, getopt
import subprocess
import math
import tkinter as tk
import mmap
from tkinter import filedialog
from subprocess import check_output
import time

import cv2
import numpy

#remove root windows
root = tk.Tk()
root.withdraw()

print("#########################################################")
print("# A Script to process the Particle Size Distribution    #")
print("#                                                       #")
print("# © 2020 Florian Kleiner                                #")
print("#   Bauhaus-Universität Weimar                          #")
print("#   F. A. Finger-Institut für Baustoffkunde             #")
print("#                                                       #")
print("#########################################################")
print()

#### directory definitions
home_dir = os.path.dirname(os.path.realpath(__file__))

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
            print( '-o,                  : setting output directory name [' + outputDirName + ']' )
            print( '-s,                  : set pixel size [' + str( pixelWidth ) + ' nm per pixel]' )
            print( '-p,                  : set page position of the mask in a TIFF [' + str( maskPagePos + 1 ) + ']' )
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

def processPSD( directory, filename ):
    global maskPagePos
    global outputDirName
    global sumResultCSV
    global CSVdelimiter

    frame = cv2.imread( directory + "/" + filename )

    if frame is None:
        print('Error loading image')
        exit()

    #binarizes an image
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, frame = cv2.threshold(frame, 127, 255, cv2.THRESH_BINARY)
    if ( showDebuggingOutput ) : print( "Trying to find contours" )
    #frame, contours, hierarchy = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    areas = []
    for i in range(0, len(contours)):
        areas.append(cv2.contourArea(contours[i]))

    #mass_centres_x = []
    #mass_centres_y = []
    #for i in range(0, len(contours)):
    #    M = cv2.moments(contours[i], 0)
    #    mass_centres_x.append(int(M['m10']/M['m00']))
    #    mass_centres_y.append(int(M['m01']/M['m00']))

    #if ( not showDebuggingOutput ) : print( '[', end = '' )
    pos = 0
    progressFactor = 1000
    for i in range(0, len(contours)):
        if ( areas[i] > 0 ):
            pos += 1
            diameter = getPoreDiameter( areas[i] )
            volume = getPoreVolume( areas[i] )
            surface = getPoreSurface( areas[i] )
            if ( showDebuggingOutput ) : 
                print( '  Area ' + str( pos ) + ':'  + str( areas[i] ) )
            else: 
                if ( pos % progressFactor == 0 ):
                    print("  ...processing particle #" + str( pos ), end="\r")
            sumResultCSV += str( pos ) + CSVdelimiter + str( areas[i] ) + CSVdelimiter + str( diameter ) + CSVdelimiter + str( volume ) + CSVdelimiter + str( surface ) + "\n"
    #if ( not showDebuggingOutput ) : print( ']' )

    print( '  Num particles: ' + str( pos ) + ', ignored 0-values: ' + str( len(contours)-pos ) )

    #for i in range(0, len(contours)):
    #    print( 'Centre' + str( (i + 1) ) + ':' + str( mass_centres_x[i]) + str( mass_centres_y[i] ) )
    

### actual program start
processArguments()
if ( showDebuggingOutput ) : print( "I am living in '" + home_dir + "'" )
workingDirectory = filedialog.askdirectory(title='Please select the image / working directory')
if ( showDebuggingOutput ) : print( "Selected working directory: " + workingDirectory )

count = 0
position = 0
## count files
if os.path.isdir( workingDirectory ) :
    for file in os.listdir(workingDirectory):
        if ( file.endswith(".tif") or file.endswith(".TIF")):
            count = count + 1
print( str(count) + " Tiffs found!" )
## run actual code
if os.path.isdir( workingDirectory ) :
    if not os.path.exists( workingDirectory + "/" + outputDirName ):
        os.makedirs( workingDirectory + "/" + outputDirName )
    ## processing files
    for file in os.listdir(workingDirectory):
        if ( file.endswith(".tif") or file.endswith(".TIF")):
            filename = os.fsdecode(file)
            position = position + 1
            print( " Analysing " + filename + " (" + str(position) + "/" + str(count) + ") :" )
            processPSD( workingDirectory, filename )


if ( sumResultCSV != '' ):
    headerLine = "poreNr" + CSVdelimiter + "area [nm^2]" + CSVdelimiter + "diameter [nm]" + CSVdelimiter + "volume [nm^3]" + CSVdelimiter + "surface [nm^2]\n"
    resultFile = open(workingDirectory + "/" + outputDirName + "/sumPSD.csv","w") 
    resultFile.write( headerLine + sumResultCSV ) 
    resultFile.close() #to change file access modes 

print( "Results can be found in directory:" )
print( "  " +  workingDirectory + "/" + outputDirName + "/\n" )
print( "Script DONE!" )