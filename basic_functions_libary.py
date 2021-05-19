# -*- coding: utf-8 -*-
import os, sys, getopt
import cv2
import math
import numpy as np
import time
from skimage.exposure import equalize_adapthist, equalize_hist, rescale_intensity
from skimage.filters.thresholding import dtype_limits, _mean_std#, threshold_phansalkar

def denoise_nlm_cv2( image, h=15, templateWindowSize=7, searchWindowSize=16, sigma_est=0 ):
    """image denoising based on opencv with woking parameter set
    Parameters
    ----------
    image : ndarray
        Input image.
    h :
    templateWindowSize :
    searchWindowSize :

    Returns
    -------
    denoised image : ndarray
        Output image

    Examples
    --------
    >>> image = denoise_nlm_cv2( image, sigma_est=0 )
    """

    # from https://github.com/thorstenwagner/ij-nl-means/blob/master/src/main/java/de/biomedical_imaging/ij/nlMeansPlugin/NLMeansDenoising_.java
    """

        if (type == ImagePlus.COLOR_256 || type == ImagePlus.COLOR_RGB) {

            // Color Image

            if (sigma > 0 && sigma <= 25) {
                n = 1;
                w = 10;
//                n = 3;
//                w = 17;
                hfactor = 0.55;
            } else if (sigma > 25 && sigma <= 55) {
                n = 2;
                w = 17;
                hfactor = 0.4;
            } else {
                n = 3;
                w = 17;
                hfactor = 0.35;
            }
        } else {

            // Gray Image

            if (sigma > 0 && sigma <= 15) {
                n = 1;
                w = 10;
                hfactor = 0.4;
            } else if (sigma > 15 && sigma <= 30) {
                n = 2;
                w = 10;
                hfactor = 0.4;
            } else if (sigma > 30 && sigma <= 45) {
                n = 3;
                w = 17;
                hfactor = 0.35;
            } else if (sigma > 45 && sigma <= 75) {
                n = 4;
                w = 17;
                hfactor = 0.35;
            } else {
                n = 5;
                w = 17;
                hfactor = 0.3;
            }
        }

    """
    t1 = time.time()
    print( "  denoising image using OpenCV2", end="", flush=True )

    denoised = image
    cv2.fastNlMeansDenoising( image,
                            denoised,
                            h=15,#0.6 * sigma_est,
                            templateWindowSize=7,
                            searchWindowSize=(15+1)
                            )
    print( ", took %f s" % (time.time() - t1) )

    return denoised

def equalize_histogram( source ):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(source)

def threshold_phansalkar(image, window_size=15, k=0.25, r=None, p=3.0, q=10.0):
    """Applies Phansalkar local threshold to an array. Phansalkar is a
    modification of Sauvola technique to deal with low contrast images.
    This method is using the following formula::
        T = m(x,y) * (1 + p * exp( -q * m(x,y) ) + k * ((s(x,y) / R) - 1))
    where m(x,y) and s(x,y) are the mean and standard deviation of
    pixel (x,y) neighborhood defined by a rectangular window with size w
    times w centered around the pixel. k,p and q are configurable parameters.
    R is the maximum standard deviation of a greyscale image.
    Parameters
    ----------
    image : ndarray
        Input image.
    window_size : int, or iterable of int, optional
        Window size specified as a single odd integer (3, 5, 7, …),
        or an iterable of length ``image.ndim`` containing only odd
        integers (e.g. ``(1, 5, 5)``).
    k : float, optional
        Value of the positive parameter k.
    r : float, optional
        Value of R, the dynamic range of standard deviation.
        If None, set to the half of the image dtype range.
    p : float, optional
        Value of the parameter p.
    q : float, optional
        Value of the parameter q.
    Returns
    -------
    threshold : (N, M) ndarray
        Threshold mask. All pixels with an intensity higher than
        this value are assumed to be foreground.
    Notes
    -----
    This algorithm is originally designed for detection of cell nuclei in low
    contrast images. Therefore the historgram has to be equalized beforehand
    using skimage.exposure.equalize_adapthist().
    References
    ----------
    .. [1] Phansalskar N. et al. "Adaptive local thresholding for detection of
           nuclei in diversity stained cytology images.", International
           Conference on Communications and Signal Processing (ICCSP),
           pp. 218-220, 2011
           :DOI:`10.1109/ICCSP.2011.5739305`
    Examples
    --------
    >>> from skimage import data
    >>> from skimage.exposure import equalize_adapthist
    >>> image = data.page()
    >>> image_eq = equalize_adapthist(image)
    >>> t_phansalkar = threshold_phansalkar(image_eq, window_size=15, k=0.25, p=2.0, q=10.0)
    >>> binary_image = image_eq > t_phansalkar
    """

    if r is None:
        # set r as image.ptp()
        # since the image is processed using equalize_adapthist(), r will become always 1
        r = 1.0

    m, s = _mean_std(image, window_size)
    return m * (1 + np.power(p, (-q * m) ) + k * ((s / r) - 1))


def get_phansalkar_binary( image, window_size=15, k=0.25, p=1.5, q=10.0):
    """wrapper function directly providing a binary image
    Parameters
    ----------
    image : ndarray
        Input image.
    window_size : int, or iterable of int, optional
        Window size specified as a single odd integer (3, 5, 7, …),
        or an iterable of length ``image.ndim`` containing only odd
        integers (e.g. ``(1, 5, 5)``).
    k : float, optional
        Value of the positive parameter k.
    r : float, optional
        Value of R, the dynamic range of standard deviation.
        If None, set to the half of the image dtype range.
    p : float, optional
        Value of the parameter p.
    q : float, optional
        Value of the parameter q.
    Returns
    -------
    image : ndarray
        Output image.
    """
    t1 = time.time()
    print( "  calculating phansalkar, p={:.1f}, q={:.1f}, window_size={}".format(p, q, window_size), end="", flush=True )
    image_eq = equalize_adapthist(image)
    thresh_phansalkar = threshold_phansalkar(image_eq, window_size, k, p, q)
    binary = image_eq > thresh_phansalkar
    print( ", took {:.3f} s" .format(time.time() - t1) )
    return binary

def morph_shape(val):
    if val == 0:
        return cv2.MORPH_RECT
    elif val == 1:
        return cv2.MORPH_CROSS
    elif val == 2:
        return cv2.MORPH_ELLIPSE

def morph(image, morph_type='dilate', morph_size=2):
    """a function which provides basic morphological filters (erode/dilate) from opencv
    Parameters
    ----------
    image : ndarray
        Input image.

    morph_type : string
        either 'dilate' or 'erode'

    morph_size : integer
        window size in pixel

    Returns
    -------
    dst : ndarray
        Output image

    Examples
    --------
    >>> image = morph( image, morph_type='dilate', morph_size=2)
    """
    element = cv2.getStructuringElement(morph_shape(0), (2 * morph_size + 1, 2 * morph_size + 1),
                                       (morph_size, morph_size))
    if morph_type == 'erode':
        dst = cv2.erode(image, element)
    else:
        dst = cv2.dilate(image, element)

    return dst

def fill_holes(image):
    """a function which provides the morphological filter 'fill holes'
    Parameters
    ----------
    image : ndarray
        Input image.

    morph_type : string
        either 'dilate' or 'erode'

    morph_size : integer
        window size in pixel

    Returns
    -------
    dst : ndarray
        Output image

    Examples
    --------
    >>> image = fill_holes( image)
    """
    fill_mask = image.copy()
    h, w = image.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(fill_mask, mask, (0,0), 255)
    im_floodfill_inv = cv2.bitwise_not(fill_mask)
    return image | im_floodfill_inv

def overlay_mask(img, mask, color=(255, 0, 0)):
    if len( img.shape ) < 3: #grayscale image
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = img
    color_mask = np.zeros(img_rgb.shape, img_rgb.dtype)
    color_mask[:,:] = color
    redMask = cv2.bitwise_and(color_mask, color_mask, mask=mask)

    cv2.addWeighted(redMask, 0.3, img_rgb, 1, 0, img_rgb)

    return img_rgb

## basic functions to calculate diameter, area volume and pore surface
def getPoreDiameter( area ):
    return ( math.sqrt( area /math.pi )*2 )

def getPoreArea( diameter ):
    radius = diameter/2
    return (math.pi*(radius**2))

def getPoreVolume( diameter=None ):
    radius = diameter/2
    return 4/3*(math.pi*(radius**3))

def getPoreSurface( diameter=None ):
    radius = diameter/2
    return (4*math.pi*(radius**2))

### actual program start
if __name__ == '__main__':
    print( "This is just a libary." )