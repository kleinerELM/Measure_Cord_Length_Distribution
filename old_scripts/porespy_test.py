
import os, sys, getopt
import tkinter as tk
from tkinter import filedialog
import time
import porespy as ps
import matplotlib.pyplot as plt
import imageio

def programInfo():
    print("#########################################################")
    print("# Testscript for porespy                                #")
    print("#                                                       #")
    print("# © 2020 Florian Kleiner                                #")
    print("#   Bauhaus-Universität Weimar                          #")
    print("#   F. A. Finger-Institut für Baustoffkunde             #")
    print("#                                                       #")
    print("#########################################################")
    print()

#### process given command line arguments
def processArguments( settings ):
    argv = sys.argv[1:]
    usage = sys.argv[0] + " [-h] [-d]"
    try:
        opts, args = getopt.getopt(argv,"hd",[])
    except getopt.GetoptError:
        print( usage )
    for opt, arg in opts:
        if opt == '-h':
            print( 'usage: ' + usage )
            print( '-h,                  : show this help' )
            print( '-d                   : show debug output' )
            print( '' )
            sys.exit()
        elif opt in ("-d"):
            print( 'show debugging output' )
            settings['showDebuggingOutput'] = True
    # print information for the main settings:
    #print( 'Settings:')
    #print( '' )
    return settings

if __name__ == '__main__':

    root = tk.Tk()
    root.withdraw()
    settings = {
        "showDebuggingOutput" : False,
        "home_dir" : os.path.dirname(os.path.realpath(__file__)),
        "workingDirectory" : "",
        "targetDirectory"  : "",
        "referenceFilePath" : "",
        "count" : 0,
    }

    programInfo()
    settings = processArguments( settings )

    print( "Please select a binary image.", end="\r" )
    settings["referenceFilePath"] = filedialog.askopenfilename(title='Please select a binary image',filetypes=[("Tiff images", "*.tif;*.tiff")])
    #workingDirectory = filedialog.askdirectory(title='Please select the image / working directory')
    if ( settings['showDebuggingOutput'] ) :
        print( "I am living in '" + settings['home_dir'] + "'" )
    #    print( "Selected working directory: " + workingDirectory, end='\n\n' )

    im = imageio.imread( settings["referenceFilePath"] )
    chords_x = ps.filters.apply_chords(im, axis=0, spacing=0, trim_edges=True)
    cld_x = ps.metrics.chord_length_distribution( chords_x, bins=100, log=True )

    fig, (ax0) = plt.subplots( ncols=1, nrows=1, figsize=(20,10) )
    print( cld_x.bin_centers )
    ax0.bar(cld_x.bin_centers,cld_x.relfreq,width=cld_x.bin_widths,edgecolor='k')

    plt.show()


    '''
    #data = ps.metrics.two_point_correlation_fft(im)
    fig = plt.plot(*data, 'bo-')
    plt.ylabel('probability')
    plt.xlabel('correlation length [voxels]')
    plt.show()

    print('ps.filters.local_thickness')
    #im = ps.generators.blobs( shape=[200, 200], porosity=0.5, blobiness=2 )
    lt = ps.filters.local_thickness(im)
    plt.imshow(lt)
    plt.show()
    print('ps.filters.apply_chords')
    cr = ps.filters.apply_chords(im)
    print('ps.filters.flood')
    cr = ps.filters.flood(cr, mode='size')
    plt.imshow(cr)
    plt.show(block=False)
    '''
    print( "Script DONE!" )