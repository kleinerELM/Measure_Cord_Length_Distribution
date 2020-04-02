#########################################################
# Automated diagram generation from CSV-files
#
# © 2020 Florian Kleiner
#   Bauhaus-Universität Weimar
#   Finger-Institut für Baustoffkunde
#
# programmed using python 3.7, gnuplot 5.2
#
#########################################################


import csv
import os, sys, getopt
import subprocess
import math
import tkinter as tk
import mmap
import re #regular expressions
import statistics
from PIL import Image
from tkinter import filedialog
from subprocess import check_output

print("#################################################################")
print("# Automated diagram generation of a pore size distribution      #")
print("# created by CSV-files from \"supported_manual_segmentation.ijm\" #")
print("#                                                               #")
print("# © 2020 Florian Kleiner                                        #")
print("#   Bauhaus-Universität Weimar                                  #")
print("#   Finger-Institut für Baustoffkunde                           #")
print("#                                                               #")
print("#################################################################")
print()

#### directory definitions
home_dir = os.path.dirname(os.path.realpath(__file__))

#### global var definitions
root = tk.Tk()
root.withdraw()
runGnuPlot_Script = True
showDebuggingOutput = False
#poreSizeRangeArray = [ 
#    0, 1, 2, 4, 8, 16, 31.5, 63, 125, 250, 500, 
#    1000, 2000, 4000, 8000, 16000, 31500, 63000, 125000, 250000, 500000, 1000000, 
#    2000000, 4000000, 8000000, 16000000, 31500000, 63000000, 125000000, 250000000, 500000000, 1000000000,
#    2000000000, 4000000000, 8000000000, 16000000000, 31500000000, 63000000000, 125000000000, 250000000000, 500000000000, 1000000000000,
#    2000000000000, 4000000000000, 8000000000000, 16000000000000, 31500000000000, 63000000000000, 125000000000000, 250000000000000, 500000000000000, 1000000000000000,
#    2000000000000000, 4000000000000000, 8000000000000000, 16000000000000000, 31500000000000000, 63000000000000000, 125000000000000000, 250000000000000000, 500000000000000000, 1000000000000000000  ]
#poreSizeRangeArray = [i for i in range(9,1800,9)]
#poreSizeRangeArray = [i for i in range(9,(300*9),18)]
#for i in range(100):
#    poreSizeRangeArray[i] = 0
#for i in range(100, 1000, 2):
#    poreSizeRangeArray[i] = 0
# prepare result arrays
poreCountSumArray = []
resultCSVTable = []
summaryResultCSVTable = []
circularitySumArray = []
summaryPosition = 0
gnuplotBefehl = ''
gnuplotLineBefehl = ''
gnuplotPlotID = 1
gnuplotPlotLineID = 1
showBinning = False
xUnit = 'nm'
yUnit = 1 # 0 = area-%, 1 = count
unitArray = [ 'nm', 'µm', 'mm', 'm' ]
unitFactorArray = [ 1, 1000, 1000000, 1000000000 ]
calcPoreDia = 0
calcPoreDiaNames = [ 'diameter', 'area', 'volume', 'surface' ]
calcPoreDiaUnits = []
roundPrecisionArray = [ 1, 3, 6, 9 ]
rangeFactor = 1
roundPrecision = 2
minVal = 0
maxVal = 0
minCircularity = float( 0 )
poreDiameterLimit = 100 #nm
ignoreLastLine = True
fileCount = 0
singeFile = -1
pixelSize = 2.9141

def getRangeFactor( unit ):
    global unitArray
    global unitFactorArray
    global rangeFactor
    global roundPrecision

    for i in range(len(unitArray)):
        if ( unitArray[ i ] == unit ):
            rangeFactor = unitFactorArray[i]**3
            roundPrecision = roundPrecisionArray[i]**3

getRangeFactor( xUnit )
roundPrecision = 2


def processArguments():
    argv = sys.argv[1:]
    usage = sys.argv[0] + " [-h] [-x <unit>] [-y <unit type>] [-s <column nr>] [-g] [-d] [-c]"
    
    global unitArray
    global minCircularity
    global yUnit
    global xUnit
    global runGnuPlot_Script
    global showDebuggingOutput
    global showBinning
    global singeFile
    global poreDiameterLimit
    global calcPoreDia
    global calcPoreDiaNames
    global calcPoreDiaUnits
    global pixelSize

    availableUnits = ''
    for i in range(len(unitArray)):
        availableUnits = availableUnits + ' ' + unitArray[ i ]

    try:
        opts, args = getopt.getopt(argv,"hgl:s:x:y:c:m:dbip:",["noGnuPlot=","setXUnit=","setYUnit=","calcPoreValue="])
    except getopt.GetoptError:
        print( usage )
    for opt, arg in opts:
        if opt == '-h':
            print( 'usage: ' + usage )
            print( 'initial values are shown in brackets' )
            print( '-h,                  : show this help' )
            print( '-g, --noGnuPlot      : skip GnuPlot processing [off]' )
            print( '-l,                  : set pore diameter limit [' + str( poreDiameterLimit ) + ' nm]' )
            print( '-s,                  : set this option to process a single file and' )
            print( '                       specify the column containing the diameter of a pore [off]' )
            print( '-x, --setXUnit       : set unit (Available: ' + availableUnits + ') [nm]')
            print( '-y, --setYUnit       : set output type (0: area-%, 1: count) [' + str( yUnit ) + ']' )
            print( '-c, --calcPoreValue  : calculating Y value type [' + str( calcPoreDia ) + ": " + calcPoreDiaNames[ calcPoreDia ] + ']:' )
            print( '                       0: using pore area A (measurement)' )
            print( '                       1: using pore diameter d (d=sqrt( A/Pi )*2)' )
            print( '                       2: using pore volume V (V= )' )
            print( '                       3: using pore surface S (S= )' )
            print( '-m                   : min circularity [' + str( minCircularity ) + ']' )
            print( '-d                   : show debug output [off]' )
            print( '-b                   : show binning results [off]' )
            print( '-i                   : export last bin/line [off]' )
            print( '-p                   : pixelSize [' + str( pixelSize ) +  ' nm]' )
            print( '' )
            sys.exit()
        elif opt in ("-g", "-noGnuPlot"):
            runGnuPlot_Script = False
        elif opt in ("-x", "-setXUnit"):
            if ( arg in unitArray ):
                xUnit = arg
                rangeFactor = getRangeFactor( xUnit )
                print( 'X-Unit changed to ' + xUnit )
            else:
                print( 'Unit is unknown! (Fallback: nm) Known units:' + availableUnits )
        elif opt in ("-y", "-setYUnit"):
            yUnit = int( arg )
        elif opt in ("-c", "-calcPoreDia"):
            if ( int( arg ) < len(calcPoreDiaNames) ): calcPoreDia = int( arg )
        elif opt in ("-b"):
            showBinning = True
        elif opt in ("-d"):
            showDebuggingOutput = True
        elif opt in ("-m"):
            minCircularity = float( arg )
        elif opt in ("-i"):
            ignoreLastLine = False
        elif opt in ("-s"):
            singeFile = int( arg )
        elif opt in ("-l"):
            poreDiameterLimit = int( arg )
        elif opt in ("-p"):
            pixelSize = float( arg )

    print( 'Use -h to show help.' )
    print( '' )
    print( 'Start parameters:' )
    if ( runGnuPlot_Script ): print( '  GnuPlot processing deactivated' )
    print( '  X-Unit is set to ' + xUnit )
    calcPoreDiaUnits = [ xUnit, xUnit+'²', xUnit+'³', xUnit+'²']
    if ( yUnit == 0 ): print( '  Y-Unit is set to area-%' )
    else: print( '  Y-Unit is set to count' )
    print( '  Using pore ' + calcPoreDiaNames[calcPoreDia] )
    if ( showBinning ): print( '  Show binning results' )
    if ( showDebuggingOutput ): print( '  Show debugging output' )
    if ( minCircularity > 0 ): print( '  Min circularity: ' + str( minCircularity ) )
    print( "  Pore diameter limit is set to " + str( poreDiameterLimit ) + " " + xUnit )
    print( '' )

def combineCSV( directory ):
    global fileCount
    outputFileName = 'combined.csv'
    outputFilePath = directory + "/" + outputFileName
    firstLine = True
    i = 0
    fileCount = 0
    if ( os.path.isfile( outputFilePath ) ): os.remove(outputFilePath )
    fout=open( outputFilePath,"a")
    print( "Combining all CSVs:" ) 
    for file in os.listdir(directory):
        if ( file.endswith(".csv") or file.endswith(".CSV")):
            filename = os.fsdecode(file)
            if ( 'result_' not in filename and filename != outputFileName ):
                print( "  Adding \"" + filename + "\":" )
                i = 0
                for line in open( directory + "/" + filename, 'r' ):
                    if ( i == 0 ):
                        if ( firstLine ): 
                            fout.write(line)
                            firstLine = False
                    else:
                        fout.write(line)
                    i += 1
                fileCount += 1
            else:
                if ( showDebuggingOutput ) : print( '  Ignoring result file: ' + filename )
    print( '' )
    print( '  Combined '+str( i )+' lines in '+str( fileCount )+' files' )
    fout.close()
    # if no file found, remove the created CSV
    if (fileCount == 0): os.remove(outputFilePath )

def getUnit():
    global xUnit
    global calcPoreDia
    unit = xUnit# + '²'
    return unit

def getLimit():
    global calcPoreDia
    global poreDiameterLimit
    limit = poreDiameterLimit
    return limit

def getPoreRadius( area ):
    return ( math.sqrt( area /math.pi ) )

def getPoreDiameter( area ):
    return ( getPoreRadius( area )*2 )

def getPoreVolume( area ):
    return 4/3*(math.pi*( getPoreRadius( area )**3))

def getPoreSurface( area ):
    return (math.pi*( getPoreDiameter( area )**2))

def initMeanResultCSV( directory ):
    global poreSizeRangeArray
    global calcPoreDiaNames
    global calcPoreDiaUnits
    limit = getLimit()
    unit = getUnit()

    headlinePart = ""
    commas = ""
    headlineUnits = "[ "
    for j in range(len(calcPoreDiaNames)):
        headlinePart += "," + calcPoreDiaNames[j]
        headlineUnits += calcPoreDiaUnits[j] + " "
        if ( j > 0 ): commas += ","
    headlineUnits += "]"

    result_file = '/result_mean.csv'
    csv_result_file = open(directory + result_file, 'w')
    # headline 1
    csv_result_file.write( 
        "specimen," + 
        "mean" + commas + "," +
        "mean (diameter limit: " + str( round( limit,1 ) ) + " " + xUnit + ")" + commas + "," +
        "median" + commas + "," +
        "median (diameter limit: " + str( round( limit,1 ) ) + " " + xUnit + ")" + commas + "\n" )
    # headline 2
    csv_result_file.write( 
        headlineUnits + 
        headlinePart +
        headlinePart +
        headlinePart +
        headlinePart +
        "\n" )
    #csv_result_file.write( "specimen, Mean, Mean (limit: " + str( round( limit,1 ) ) + " " + unit + "), Median, Median (limit: " + str( round( limit,1 ) ) + " " + unit + ") \n" )
    csv_result_file.close()

    for j in range(len(calcPoreDiaNames)):
        csv_result_file = open(directory + '/result_' + calcPoreDiaNames[j] + '.csv', 'w')
        resultline = "specimen"
        for i in range(len(poreSizeRangeArray)):
            resultline += ", " + str( poreSizeRangeArray[i] )
        csv_result_file.write( resultline + "\n" )
        csv_result_file.close()

def processCSV( directory, filename ):
    # pore analysis
    resultLine = ""
    global resultCSVTable
    global rangeFactor
    global yUnit
    global minVal
    global maxVal
    global summaryResultCSVTable
    global summaryPosition
    global minCircularity
    global poreCountSumArray
    global circularitySumArray
    global gnuplotPlotID
    global gnuplotLineBefehl
    global gnuplotBefehl
    global showBinning
    global roundPrecision
    global calcPoreDiaNames
    global calcPoreDiaUnits
    
    with open(directory +'/'+ filename, 'r') as csv_file:
        #im = Image.open( directory + outputDir_Pores + filename + "-masked.tif")
        #width, height = im.size
        #im.close()
        #imageArea = width * pixelSize * height * pixelSize
        #print( " image area: " + str( width * height ) + " px² | " + str( imageArea ) + " nm²" )
        
        delimiterChar = ',' if ( singeFile < 0 ) else "\t"
        csv_reader = csv.reader(csv_file, delimiter=delimiterChar)

        lineNr = 0
        poreDiameterSum = 0
        poreAreaSum     = 0
        poreVolumeSum   = 0
        poreSurfaceSum  = 0
        poreAreaLimited         = 0
        poreDiameterSumLimited = 0
        poreVolumeSumLimited   = 0
        poreSurfaceSumLimited  = 0

        poreDiameterList = []
        poreAreaList     = []
        poreVolumeList   = []
        poreSurfaceList  = []
        poreDiameterListLimited = []
        poreAreaListLimited     = []
        poreVolumeListLimited   = []
        poreSurfaceListLimited  = []

        poreSizePercentArray = []
        poreCountArray = []
        poreDiameterArray = []
        poreAreaArray     = []
        poreVolumeArray   = []
        poreSurfaceArray  = []
        
        poreDiameterListByBin = []
        poreAreaListByBin = []
        poreVolumeListByBin = []
        poreSurfaceListByBin = []

        #initiate arrays
        resultCSVTable = []
        resultCSVTable.append( "#bucket" )
        resultCSVTable.append( "#scale [nm/px]" )
        resultCSVTable[0] += ",fullSum"
        resultCSVTable[1] += ",-"
        for i in range(len(poreSizeRangeArray)):
            resultCSVTable.append( str( poreSizeRangeArray[i] ) )

        # define vars which are depending on user input for calcPoreDia 
        limit = getLimit()
        unit = getUnit()
        
        poreType = 'pore diameter'
        # initiate empty arrays
        poreCountSumArray = []
        poreSizeSumPercentArray = []
        circularitySumArray = []
        for val in poreSizeRangeArray:
            poreCountSumArray.append( 0 )
            poreCountArray.append( 0 )
            circularitySumArray.append( float(0) )
            poreDiameterArray.append( float( 0 ) )
            poreAreaArray.append( float( 0 ) )
            poreVolumeArray.append( float( 0 ) )
            poreSurfaceArray.append( float( 0 ) )
            poreDiameterListByBin.append( [] )
            poreAreaListByBin.append( [] )
            poreVolumeListByBin.append( [] )
            poreSurfaceListByBin.append( [] )

        result_file = '/result_' + filename
        csv_result_file = open(directory + result_file, 'w')
        resultline = ("size, " +              # 1: actual pore size bin
                      "count, " +             # 2: pore count in the actual bin
                      "size percent, " +      # 3: % pore size/full pore size
                      "size percent sum, " +  # 4: % summed pore size/full pore size
                      "sum diameter, " +      # 5: summed pore diameter for the actual bin
                      "sum area, " +          # 6: summed pore area for the actual bin
                      "sum volume, " +        # 7: summed pore volume for the actual bin
                      "sum surface, " +       # 8: summed pore surface for the actual bin
                      "circularity, " +        # 9: mean circularity
                      "stdev of pore diameter, " + # : stdev of pore diameter
                      "stdev of pore area, " +     # : stdev of pore area
                      "stdev of pore volume, " +   # : stdev of pore volume
                      "stdev of pore surface\n")   # : stdev of pore surface
        csv_result_file.write( resultline )
        csv_result_file.close()
        elementCount = 0
        elementCountLimited = 0

        print( '  counting rows', end="\r" )
        rowCountCSV = sum(1 for row in csv_reader)
        #print( '  found ' + str( rowCountCSV ) + ' rows', end="\r" )
        csv_file.seek(0)
        # read every line in csv
        for line in csv_reader:
            
            if ( lineNr % 1000 == 0) : print('  ... position ' + str( lineNr ) + ' of ' + str( rowCountCSV ), end="\r")
            if ( lineNr > 0 and line != "" ): # ignore first two lines (headline) and empty lines
                if ( singeFile > 0 ):
                    diameter = float( line[ singeFile ] )
                    area = math.pi*(diameter/2)**2
                    circularity = 1
                    # print( "circularity processing does not work!" )
                else:
                    area = float( line[1] )
                    circularity = float( line[2] )
                    diameter = getPoreDiameter( area )

                volume = getPoreVolume( area )
                surface = getPoreSurface( area )

                poreDiameterList.append( diameter )
                poreAreaList.append( area )
                poreVolumeList.append( volume )
                poreSurfaceList.append( surface )
                # fill array with pore sizes for mean/median calculations
                if ( diameter <= limit ): 
                    elementCountLimited += 1
                    poreDiameterArray[i] = diameter
                    poreAreaArray[i]     = area
                    poreVolumeArray[i]   = volume
                    poreSurfaceArray[i]  = surface
                    poreDiameterSumLimited += diameter
                    poreAreaLimited         += area
                    poreVolumeSumLimited   += volume
                    poreSurfaceSumLimited  += surface
                    poreDiameterListLimited.append( diameter )
                    poreAreaListLimited.append(     area )
                    poreVolumeListLimited.append(   volume )
                    poreSurfaceListLimited.append(  surface )
                noBinFound = True
                #search the correct size range and insert into corresponding result arrays
                #if ( showDebuggingOutput and lineNr < 5 ): print( str(round(diameter,4)) + "------------------" )
                poreSizeRangeArrayLength = len(poreSizeRangeArray)
                for i in range(len(poreSizeRangeArray)):
                    #if ( showDebuggingOutput and lineNr < 5 ): print( poreSizeRangeArray[i] )
                    if ( i > 0 ):
                        if ( minCircularity < circularity ):
                            binFound = False
                            if (len(poreSizeRangeArray)-1 == i and poreSizeRangeArray[i] < round(diameter,4) ):
                                if ( showDebuggingOutput and lineNr < 60) : print( '  A: ' + str(poreSizeRangeArray[i])  + ' < ' + str(round(diameter,4)) + ' ('+str(round( area, 1))+') ')
                                binFound = True
                            elif ( poreSizeRangeArray[i-1] < round(diameter,4) and poreSizeRangeArray[i] >= round(diameter,4) ):
                                if ( showDebuggingOutput and lineNr < 60) : print( '  B: ' + str(poreSizeRangeArray[i-1]) + ' < ' + str(round(diameter,6)) + ' ('+str(round( area, 1))+') '+ ' <= ' + str( poreSizeRangeArray[i]) )
                                if ( showDebuggingOutput and poreSizeRangeArray[i] > diameter and lineNr < 20) : print( '  B: ' + str(poreSizeRangeArray[i-1]) + ' < ' + str(round(diameter,6)) + ' ('+str(round( area, 1))+') '+ ' < ' + str( poreSizeRangeArray[i]) )
                                binFound = True
                            if ( binFound ):
                                # fill bin arrays
                                poreCountArray[i] += 1
                                poreCountSumArray[i] += 1
                                circularitySumArray[i] += circularity
                                # derivatives from the pore area
                                poreDiameterArray[i] += diameter
                                poreAreaArray[i]     += area
                                poreVolumeArray[i]   += volume
                                poreSurfaceArray[i]  += surface
                                # list of items inside a bin for stdev
                                poreDiameterListByBin[i].append( diameter )
                                poreAreaListByBin[i].append(     area )
                                poreVolumeListByBin[i].append(   volume )
                                poreSurfaceListByBin[i].append(  surface )
                                # calculate sums
                                poreDiameterSum += diameter
                                poreAreaSum     += area
                                poreVolumeSum   += volume
                                poreSurfaceSum  += surface
                                elementCount += 1
                                noBinFound = False
                if ( showBinning and noBinFound ): print('  No bin found for: ' + str(round(diameter,2)) + ' ('+str(round( area, 1))+') ')
            lineNr += 1
        mean = []
        meanLtd = []
        if ( elementCount > 0 ):
            mean.append( poreDiameterSum/elementCount )
            mean.append( poreAreaSum/elementCount )
            mean.append( poreVolumeSum/elementCount )
            mean.append( poreSurfaceSum/elementCount )

        if ( elementCountLimited > 0 ):
            meanLtd.append( poreDiameterSumLimited/elementCountLimited )
            meanLtd.append( poreAreaLimited/elementCountLimited )
            meanLtd.append( poreVolumeSumLimited/elementCountLimited )
            meanLtd.append( poreSurfaceSumLimited/elementCountLimited )

        median = []
        medianLtd = []
        medianPos = int(round( elementCount/2, 0))
        medianLtdPos = int(round( len(poreAreaListLimited)/2, 0))

        tmp_array = sorted( poreDiameterList )
        median.append( tmp_array[ medianPos ] )
        tmp_array = sorted( poreAreaList )
        median.append( tmp_array[ medianPos ] )
        tmp_array = sorted( poreVolumeList )
        median.append( tmp_array[ medianPos ] )
        tmp_array = sorted( poreSurfaceList )
        median.append( tmp_array[ medianPos ] )

        tmp_array = sorted( poreDiameterListLimited )
        medianLtd.append( tmp_array[ medianLtdPos ] )
        tmp_array = sorted( poreAreaListLimited )
        medianLtd.append( tmp_array[ medianLtdPos ] )
        tmp_array = sorted( poreVolumeListLimited )
        medianLtd.append( tmp_array[ medianLtdPos ] )
        tmp_array = sorted( poreSurfaceListLimited )
        medianLtd.append( tmp_array[ medianLtdPos ] )

        print( "  Processed elements: " + str( elementCount ) )
        print()
        print( "  results without upper limit:" )
        for i in range(len(calcPoreDiaNames)):
            print( "   Mean " + calcPoreDiaNames[i] + ": " + str( round(  mean[i], roundPrecision) ) + " " + xUnit )
        print()
        for i in range(len(calcPoreDiaNames)):
            print( "   Median " + calcPoreDiaNames[i] + ": " + str( round(  median[i], roundPrecision) ) + " " + xUnit )

        print()
        print()
        print( "  Processed elements below a diameter of " + str( limit )  + " " + xUnit + ": " + str( elementCountLimited ) )
        print( "  results with upper limit: " + str( round( limit, roundPrecision ) ) + " " + unit )
        for i in range(len(calcPoreDiaNames)):
            print( "   Mean " + calcPoreDiaNames[i] + ": " + str( round(  meanLtd[i], roundPrecision) ) + " " + calcPoreDiaUnits[i] )
        print()
        for i in range(len(calcPoreDiaNames)):
            print( "   Median " + calcPoreDiaNames[i] + ": " + str( round(  medianLtd[i], roundPrecision) ) + " " + calcPoreDiaUnits[i] )

         #.sort( key )

        result_file_mean = '/result_mean.csv'
        if ( showDebuggingOutput ) : print( '  Saving to mean/median table to CSV: ' + result_file_mean )
        csv_result_file = open(directory + result_file_mean, 'a')
            
        csv_result_file.write( 
            filename + "," + 
            str( round(  mean[0], roundPrecision) ) + "," +
            str( round(  mean[1], roundPrecision) ) + "," +
            str( round(  mean[2], roundPrecision) ) + "," +
            str( round(  mean[3], roundPrecision) ) + "," +
            str( round(  median[0], roundPrecision) ) + "," +
            str( round(  median[1], roundPrecision) ) + "," +
            str( round(  median[2], roundPrecision) ) + "," +
            str( round(  median[3], roundPrecision) ) + "," +
            str( round(  meanLtd[0], roundPrecision) ) + "," +
            str( round(  meanLtd[1], roundPrecision) ) + "," +
            str( round(  meanLtd[2], roundPrecision) ) + "," +
            str( round(  meanLtd[3], roundPrecision) ) + "," +
            str( round(  medianLtd[0], roundPrecision) ) + "," +
            str( round(  medianLtd[1], roundPrecision) ) + "," +
            str( round(  medianLtd[2], roundPrecision) ) + "," +
            str( round(  medianLtd[3], roundPrecision) ) + "\n" )
        csv_result_file.close()
        
        # process result line
        fullAreaPoresSum = 0
        if ( showBinning ): print ( "  ----------------------------------------") 
        if ( showDebuggingOutput ) : print( '  Saving binning table to CSV: ' + result_file )
        csv_result_file = open(directory + result_file, 'a')
        poreFullSum = 0
        minVal = 0
        maxVal = 0
        
        for i in range(len(poreSizeRangeArray)):
            poreFullSum += 100/poreAreaSum*poreAreaArray[i]
            circularity = 0
            if ( poreCountArray[i] > 0 ):
                circularity = circularitySumArray[i]/poreCountArray[i]
            debugMessage = '  - ' + str( i ) + ': ' + str( round( poreSizeRangeArray[i]/rangeFactor, roundPrecision ) ) + ' ' + unit + ': ' + str( poreCountArray[i] ) + 'x (' + str( round( ( 100/poreAreaSum*poreAreaArray[i] ), 1 ) ) + ' %, ' + str( round( poreSizeRangeArray[i]/rangeFactor, 6 ) ) + ' ' + unit + ', circ: ' + str( round(circularity, 3) ) + ')'
            if ( showBinning ): print( debugMessage )
            if ( poreFullSum > 0 ) :
                if ( minVal == 0 ):
                    minVal = round( poreSizeRangeArray[i]/rangeFactor, roundPrecision )
                if ( poreCountArray[i] > 0 ):
                    maxVal = round( poreSizeRangeArray[i]/rangeFactor, roundPrecision )
                    
                stdevDiameter = 0
                stdevArea = 0
                stdevVolume = 0
                stdevSurface = 0
                if ( poreCountArray[i] > 1 ):
                    stdevDiameter = statistics.stdev(poreDiameterListByBin[i])
                    stdevArea = statistics.stdev(poreAreaListByBin[i])
                    stdevVolume = statistics.stdev(poreVolumeListByBin[i])
                    stdevSurface = statistics.stdev(poreSurfaceListByBin[i])
                # create columns
                resultline = (str( round( poreSizeRangeArray[i]/rangeFactor, 6 ) ) + ',' + # 1: actual pore size bin
                              str( poreCountArray[i] ) + ',' +                                          # 2: pore count in the actual bin
                              str( round( ( 100/poreAreaSum*poreAreaArray[i] ), 2 ) ) + ',' +           # 3: % pore area /full pore area
                              str( round( ( poreFullSum ), 2 ) ) + ',' +                                # 4: % summed pore area/full pore area
                              str( round( poreDiameterArray[i], roundPrecision ) ) + ',' +              # 5: summed pore diameter for the actual bin
                              str( round( poreAreaArray[i], roundPrecision ) ) + ',' +                  # 6: summed pore area for the actual bin
                              str( round( poreVolumeArray[i], roundPrecision ) ) + ',' +                # 7: summed pore volume for the actual bin
                              str( round( poreSurfaceArray[i], roundPrecision ) ) + ',' +               # 8: summed pore surface for the actual bin
                              str( circularity ) + ',' +                                                # 9: mean circularity
                              str( round( stdevDiameter, roundPrecision ) ) + ',' +                     # 10: stdev of pore diameter for the actual bin
                              str( round( stdevArea, roundPrecision ) ) + ',' +                         # 11: stdev of pore area for the actual bin
                              str( round( stdevVolume, roundPrecision ) ) + ',' +                       # 12: stdev of pore volume for the actual bin
                              str( round( stdevSurface, roundPrecision ) ) + "\n" )                     # 13: stdev of pore surface for the actual bin
                if ( ignoreLastLine and ( i == (len(poreSizeRangeArray)-1) ) ): 
                    poreCountArray[i] = 0
                    if ( showDebuggingOutput ) : print( "  ignoring last line" )
                if ( poreCountArray[i] > 0 ): csv_result_file.write( resultline )
        if ( showDebuggingOutput ) : print( '  min value: ' + str(minVal) +' max value: '+ str(maxVal) )
        csv_result_file.close()

        # deviations between files
        result_arrays = [ poreDiameterArray, poreAreaArray, poreVolumeArray, poreSurfaceArray ]
        for j in range(len(calcPoreDiaNames)):
            csv_result_file = open(directory + '/result_' + calcPoreDiaNames[j] + '.csv', 'a')
            resultline = filename
            for i in range(len(poreSizeRangeArray)):
                resultline += ", " + str( round( result_arrays[j][i], roundPrecision ) )
            csv_result_file.write( resultline + "\n" )
            csv_result_file.close()

        # generate GnuPlot plot
        gnuplotPlotID += 1
        if ( yUnit == 0 ):
            if ( calcPoreDia == 0 ): # diameter
                columNr = 5
            elif ( calcPoreDia == 1 ): # area
                columNr = 6
            elif ( calcPoreDia == 2 ): # volume
                columNr = 7
            elif ( calcPoreDia == 3 ): # surface
                columNr = 8
        else:
            columNr = 2
        label = re.sub('\.csv$', '', filename)
        label = re.sub('\-cut\_pores$', '', label)
        plotBefehl = "'." + result_file + "' using 1:" + str(columNr) + " title '" + label.replace('_', '\_') + "' with linespoints" #boxes" #points" #linespoints"
        gnuplotLineBefehl += plotBefehl
        if ( filename != 'combined.csv' ): gnuplotBefehl += plotBefehl + ", "
    # end processCSV()

def createGnuplotPlot( directory, filename, plot, openPDF = False ):
    global gnuplotPlotID
    global poreSizeRangeArray
    global rangeFactor
    global yUnit
    global roundPrecision
    global minVal
    global maxVal
    global showDebuggingOutput
    global calcPoreDia
    global calcPoreDiaNames
    global calcPoreDiaUnits

    print( "  Creating gnuplot plot for " + filename )
    #result_file = '/result_' + filename
    #if os.path.exists( directory + result_file ) :    
    gp_file = open( directory + '/' + filename.rsplit('.', 1)[0] + '.gp', 'w')
    #gp_file.write( 'set logscale x' + "\n" )
    gp_file.write( "set style fill solid\n" )
    
    gp_file.write( 'set datafile separator ","' + "\n" )
    gp_file.write( "set xrange [" + str( minVal ) + ":" + str( maxVal ) + "]\n" ) # modify for unit change
    gp_file.write( "set key left top\n")
    gp_file.write( 'set terminal pdf size 17cm,10cm' + "\n" )
    #gp_file.write( "set terminal wxt size 1000,500 enhanced font 'Arial,12' persist\n" )
    gp_file.write( 'set output "' + directory + '/' + filename.rsplit('.', 1)[0] + '.pdf"' + "\n" )
    gp_file.write( 'cd "' + directory + '"' + "\n" )

    unit = getUnit()

    if ( yUnit == 0 ):
        for i in range(len(calcPoreDiaNames)):
             if ( calcPoreDia == i ): yLabel = "summed pore  " + calcPoreDiaNames[i] + " for a bin in " + calcPoreDiaUnits[i]
    else:
        yLabel = "particle count"

    #yLabel = "summierte Fläche bezogen auf die Gesamtporenflächen in %" if ( yUnit == 0 ) else "Partikelanzahl"
    xLabel = "pore diameter in " + unit
    gp_file.write( 'set xlabel "' + xLabel + '"' + "\n" )
    gp_file.write( 'set ylabel "' + yLabel + '"' + "\n" )

    poreSizeRangeStr = ','.join(str(  round( e/rangeFactor, 6 ) ) for e in poreSizeRangeArray)
    gp_file.write( 'set xtics (' + poreSizeRangeStr + ') rotate by 45 right' + "\n" )
        
    gp_file.write( "plot " + plot + "\n" )
    gp_file.close()
    os.system('gnuplot "' + directory + '/' + filename.rsplit('.', 1)[0] + '.gp"')
    if ( openPDF  ):
        pdfPath = directory + '/' + filename.rsplit('.', 1)[0] + '.pdf'
        if ( os.path.exists( pdfPath ) ) :
            print( "  opening '" + pdfPath + "'" )
            subprocess.Popen( pdfPath ,shell=True)
        else:
            print( "  Error creating '" + pdfPath + "'!" )
    if ( showDebuggingOutput ) : print( "  GnuPlot creation done -" )

######################
# Start processing ...
######################
processArguments()

#if ( calcPoreDia == 0 ): # diameter
#max_value = 104
#poreSizeRangeArray = [i for i in range(1,max_value,3)]

poreSizeRangeArray = []
#pixelSize = 5.8281
#pixelSize = 11.6562
binCount = round( poreDiameterLimit/pixelSize+2 )
for i in range(1,binCount):
    poreSizeRangeArray.append( round(i*pixelSize,4)  )

if ( showDebuggingOutput ) : print( "I am living in '" + home_dir + "'" )

if ( singeFile < 0 ):
    workingDirectory = filedialog.askdirectory(title='Please select the working directory')
    if ( os.path.isdir(workingDirectory) ) :
        if ( showDebuggingOutput ) : print( "Selected working directory: " + workingDirectory )
        combineCSV( workingDirectory )
    else:
        print( 'directory "' + workingDirectory + '" does not exist or is no directory!' )

    if( fileCount > 1 ):
        processedFiles=0
        ignoredFiles=0
        print( '' )
        print( 'Start main processing:' )
        initMeanResultCSV( workingDirectory )
        #main process
        for file in os.listdir(workingDirectory):
            if ( file.endswith(".csv") or file.endswith(".CSV")):
                filename = os.fsdecode(file)
                if ( 'result_' not in filename ):
                    processedFiles += 1
                    print( " Analysing \"" + filename + "\" (" + str( processedFiles ) + "/" + str( fileCount ) + "):" )
                    processCSV( workingDirectory, filename )
                    openPDF = False
                    if ( filename == 'combined.csv' ):
                        if ( yUnit == 0 ):
                            if ( calcPoreDia == 0 ): # diameter
                                filename == 'combined.csv'
                            elif ( calcPoreDia == 1 ): # area
                                columNr = 6
                            elif ( calcPoreDia == 2 ): # volume
                                columNr = 7
                            elif ( calcPoreDia == 3 ): # surface
                                columNr = 8
                        else:
                            columNr = 2
                        openPDF = True
                    if ( runGnuPlot_Script ): createGnuplotPlot( workingDirectory, filename, gnuplotLineBefehl, openPDF )
                    gnuplotLineBefehl = ''
                    print( "" )
                else:
                    ignoredFiles += 1
                    if ( showDebuggingOutput ) : print( ' Ignoring result file: ' + filename )

        if ( gnuplotBefehl != '' ):
            if ( processedFiles > 0 ):
                print( 'Summary' )
                if ( runGnuPlot_Script ): createGnuplotPlot( workingDirectory, 'full', gnuplotBefehl, True )
                print( '  Processed ' + str( processedFiles ) + ' files' )
                if ( showDebuggingOutput ) : print( '  Ignored ' + str( ignoredFiles ) + ' files' )
            else: 
                print( '  No files processed!' )
    else:
        print( "no Files found!" )
else:
    file = filedialog.askopenfilename(title="Select file", filetypes=[("csv files", "*.csv")])
    if ( os.path.exists(file) ) :
        filename = os.path.split(file)[1]
        workingDirectory = os.path.split(file)[0]
        print( " Analysing \"" + filename + "\":" )
        processCSV( workingDirectory, filename )
    else:
        print("file does not exist")
print( '' )
print("-------")
print("DONE!")