#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, getopt, subprocess, csv, time, math, mmap
import tkinter as tk
from PIL import Image, ImageSequence
from tkinter import filedialog
from subprocess import check_output
import porespy as ps
import matplotlib.pyplot as plt
import imageio

#remove root windows
root = tk.Tk()
root.withdraw()

print("#########################################################")
print("# A Script to process the Cord Length Distribution of a #")
print("# masked image                                          #")
print("#                                                       #")
print("# © 2020 Florian Kleiner                                #")
print("#   Bauhaus-Universität Weimar                          #")
print("#   F. A. Finger-Institut für Baustoffkunde             #")
print("#                                                       #")
print("#########################################################")
print()

#### directory definitions
home_dir = os.path.dirname(os.path.realpath(__file__))

outputDirName = "resultsCLD"
showDebuggingOutput = False

materialColor = 255#0
poreColor = 0#255
globMaskPagePos = 1     # second image in tiff stack is the masked image
ignoreBorder = True#False#True
minLength = 1
sumResultCSV = ''
#pixelWidth = 2.9141 #nm

#### process given command line arguments
def processArguments():
    global outputDirName
    global showDebuggingOutput
    global ignoreBorder
    global minLength
    global materialColor
    global poreColor
    global globMaskPagePos
    argv = sys.argv[1:]
    usage = sys.argv[0] + " [-h] [-o] [-p] [-w] [-d]"
    colorName = 'black' if ( poreColor == 0 ) else 'white'
    altColor = 'black' if ( materialColor == 0 ) else 'white'
    outputDirNameOld = outputDirName
    doExit = False
    try:
        opts, args = getopt.getopt(argv,"hwp:o:d",[])
    except getopt.GetoptError:
        print( usage )
    for opt, arg in opts:
        if opt == '-h':
            print( 'usage: ' + usage )
            print( '-h,                  : show this help' )
            print( '-o,                  : setting output directory name [' + outputDirName + ']' )
            print( '-w,                  : change expected pore color from ' + colorName + ' to ' + altColor )
            print( '-p,                  : set page position of the mask in a TIFF [' + str( globMaskPagePos + 1 ) + ']' )
            print( '-d                   : show debug output' )
            print( '' )
            doExit = True
        elif opt in ("-o"):
            outputDirName = arg
        elif opt in ("-w"):
            poreColor = 255
            materialColor = 0
        elif opt in ("-p"):
            globMaskPagePos = int( arg ) -1
            if ( globMaskPagePos < 0 ): 
                globMaskPagePos = 0
        elif opt in ("-d"):
            print( 'show debugging output' )
            showDebuggingOutput = True
    # print information for the main settings:
    print( 'Settings:')
    if ( outputDirNameOld != outputDirName ): print( ' - changed output directory to "' + outputDirName + '"' )
    #print( ' - pixel size is set to ' + str( pixelWidth ) + ' nm per pixel')
    if ( ignoreBorder ) : print( ' - areas touching a border will be ignored' )
    else : print( ' - areas touching a border will be included (may be flawed!)' )
    print( ' - ignoring areas smaller than ' + str( minLength ) + ' pixel')
    colorName = 'black' if ( poreColor == 0 ) else 'white'
    print( ' - calculating the Cord Length Distribution of ' + colorName + ' areas')
    if ( globMaskPagePos == 0 ) : print( ' - expecting a normal b/w TIFF or a multi page TIFF, where the mask is on page 1' )
    else : print( ' - expecting a multi page Tiff where the mask is on page ' + str( globMaskPagePos + 1 ) )
    print( '' )
    if doExit: sys.exit()

def processDirectionalCLD( im, scaling, directory, direction ):
    global outputDirName
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
    pixelWidth = scaling['x'] # TODO Check unit?
    pixelHeight = scaling['y'] # TODO Check unit?

    minLength = 1
    resultCSV = ''
    print( '  processing ' + str( direction ) + ' cord length distribution' )#, end='' )
    startTime = int(round(time.time() * 1000))
    if ( direction == 'horizontal' ):
        for y in range(height):
            if ( y % 100 == 0 ): print('  ... line ' + str( y ) + ' of ' + str( height ), end="\r")
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
            if ( x % 100 == 0  ): print('  ... line ' + str( x ) + ' of ' + str( width ), end="\r")
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
        print( "   " + str( lineCount ) + " lines measured in " + str( int(round(time.time() * 1000)) -startTime ) + " ms", end='' )
        print( "   " + str( round(100 / imageArea * usedImageArea, 2) ) + ' of ' + str( round(100 / imageArea * fullPoreArea, 2) ) + " area-% were taken into account"   )
        #print( str( usedImageArea ) + '  ' + str( fullPoreArea ) )
        headerLine = "lineCount" + "	" + "length [nm]" + "\n"#+ "	" + "volume fraction" + "\n"
        resultFile = open(directory + "/" + outputDirName + "/" + filename + "." + direction + ".csv","w") 
        resultFile.write( headerLine + resultCSV ) 
        resultFile.close() #to change file access modes 
    return resultCSV

def getPoreArea( diameter ):
    radius = diameter/2
    return 4/3*(math.pi*(radius**3))

def getPoreVolume( diameter ):
    radius = diameter/2
    return 4/3*(math.pi*(radius**3))

def getPoreSurface( diameter ):
    radius = diameter/2
    return (4*math.pi*(radius**2))

def processCLD( directory, filename ):
    global globMaskPagePos
    global outputDirName
    global sumResultCSV
    
    scaling = getImageJScaling( filename, directory )
    pageCnt = 0
    
    im = Image.open( directory + "/" + filename )
    # check page count in image
    for i in enumerate(ImageSequence.Iterator(im)):
        pageCnt +=1
    if ( pageCnt - 1 < globMaskPagePos ) :
        print( '  WARNING: The image has only ' + str( pageCnt ) + ' page(s)! Trying to use page 1 as mask.')
        maskPagePos = 0
    else:
        maskPagePos = globMaskPagePos

    # run analysis
    for i, page in enumerate(ImageSequence.Iterator(im)):
        if ( i == maskPagePos ): 
            sumResultCSV += processDirectionalCLD(im, scaling, directory, 'horizontal')
            sumResultCSV += processDirectionalCLD(im, scaling, directory, 'vertical')
            
            img = imageio.imread( directory + "/" + filename )
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
            
            plt.savefig(directory + "/" + filename + 'line_plot.svg')  
    im.close()
    print()

def getImageJScaling( filename, workingDirectory ):
    global showDebuggingOutput

    scaling = { 'x' : 1, 'y' : 1, 'unit' : 'px', 'editor':None}
    with Image.open( workingDirectory + '/' + filename ) as img:
        #print(filename + ':')
        if ( 282 in img.tag ) and ( 283 in img.tag ):
            x_tag = img.tag[282][0]
            y_tag = img.tag[283][0]
            scaling['x'] = int( x_tag[1] )/ int( x_tag[0] )
            scaling['y'] = int( y_tag[1] )/ int( y_tag[0] )
            #print( x_tag ) #x
            #print( y_tag ) #y
        if 270 in img.tag:            
            # getimagej definitions
            IJSettingString = img.tag[270][0].split('\n')
            IJSettingsArray = {}
            for val in IJSettingString:
                if ( val != '' ):
                    setting = val.split('=')
                    IJSettingsArray[setting[0]] = setting[1]
            #print(IJSettingsArray)
            if ( 'ImageJ' in IJSettingsArray ):
                if ( showDebuggingOutput ): print( '  Image edited using ImageJ ' + IJSettingsArray['ImageJ'] )
                scaling['editor'] = 'ImageJ ' + IJSettingsArray['ImageJ']
            if ( 'unit' in IJSettingsArray ):
                scaling['unit'] = IJSettingsArray['unit']
                print( '  scaling: ' + str( round( scaling['x'], 4) ) + ' x ' + str( round( scaling['y'], 4) ) + ' ' + scaling['unit'] )
            else :
                print( '  unitless scaling: ' + str( round( scaling['x'], 4) ) + ' x ' + str( round( scaling['y'], 4) ) )
    return scaling

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
            processCLD( workingDirectory, filename )


if ( sumResultCSV != '' ):
    headerLine = "lineCount" + "	" + "length [nm]\n"
    resultFile = open(workingDirectory + "/" + outputDirName + "/sumCLD.csv","w") 
    resultFile.write( headerLine + sumResultCSV ) 
    resultFile.close() #to change file access modes 

print( "Results can be found in directory:" )
print( "  " +  workingDirectory + "/" + outputDirName + "/\n" )
print( "Script DONE!" )