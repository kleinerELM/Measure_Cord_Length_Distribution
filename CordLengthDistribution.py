import csv
import os, sys, getopt
import subprocess
import math
import tkinter as tk
import mmap
from PIL import Image, ImageSequence
from tkinter import filedialog
from subprocess import check_output
import time

#remove root windows
root = tk.Tk()
root.withdraw()

print("#########################################################")
print("# A Script to process the Cord Length Distribution of a #")
print("# masked image                                          #")
print("#                                                       #")
print("# © 2019 Florian Kleiner                                #")
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
maskPagePos = 1     # second image in tiff stack is the masked image
ignoreBorder = True#False#True
minLength = 1
sumResultCSV = ''
pixelWidth = 2.9141 #nm

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

def processDirectionalCLD( im, directory, direction ):
    global outputDirName
    global materialColor
    global ignoreBorder
    global minLength
    global pixelWidth

    lineCount = 0
    lastValue = -1      # var to save the last value
    lastChangedPos = -1   # var to save the last position
    width, height = im.size
    imageArea = width * height
    usedImageArea = 0
    fullPoreArea = 0

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
            if ( x % 100 == 0  ): print('  ... line ' + str( x ) + ' of ' + str( height ), end="\r")
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
                            resultCSV += str( lineCount ) + "	" + str( pixelWidth * length ) + "\n"# + "	" + str( length/imageArea ) + "\n"
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
    global maskPagePos
    global outputDirName
    global sumResultCSV
    im = Image.open( directory + "/" + filename ) 
    for i, page in enumerate(ImageSequence.Iterator(im)):
        if ( i == maskPagePos ): 
            sumResultCSV += processDirectionalCLD(im, directory, 'horizontal')
            sumResultCSV += processDirectionalCLD(im, directory, 'vertical')
    im.close()
    print()
    

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