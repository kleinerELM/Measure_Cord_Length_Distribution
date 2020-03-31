# Measure_Cord_Length_Distribution
Automated calculation of cord lengths of pores using a binarized image

Ported to python. The imageJ Plugin seems to generate false results. Start the script using: 

./python CordLengthDistribution.py

```
#########################################################
# A Script to process the Cord Length Distribution of a #
# masked image                                          #
#                                                       #
# © 2019 Florian Kleiner                                #
#   Bauhaus-Universität Weimar                          #
#   F. A. Finger-Institut für Baustoffkunde             #
#                                                       #
#########################################################

usage: D:\Nextcloud\Uni\WdB\REM\Fiji Plugins & Macros\Selbstgeschrieben\Measure_Cord_Length_Distribution\CordLengthDistribution.py [-h] [-o] [-s] [-p] [-d]
-h,                  : show this help
-o,                  : setting output directory name [resultsCLD]
-p,                  : set page position of the mask in a TIFF [2]
-d                   : show debug output
```


to finally process diagrams use: .\generate_diagram.py -s 1 -c 3 -l 500
and select sumCLD.csv