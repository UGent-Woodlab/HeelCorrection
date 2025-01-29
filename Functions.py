'''
Written by Jorden De Bolle
Last update: 18/07/2022
'''

import numpy as np
import scipy.ndimage as scpim
import scipy.interpolate as scpip
import scipy.signal as scpsig
import FileHandler as FH
import matplotlib.pyplot as plt
from numba import jit

def displayArray(data):
    '''
    Plots 2D numpy array in gray scale
    :param data: numpy array to display
    :return: None
    '''
    plt.imshow(data, cmap='gray')
    plt.colorbar()
    #plt.clim(0, 1)
    plt.show()

def addGaussianNoise(image, mean, sigma):
    '''
    Add Gaussian noise to image
    :param image: 2D numpy array
    :param mean: mean value of the Gaussian distribution from which the noise is sampled (usually 0)
    :param sigma: standard deviation of the Gaussian distribution
    :return: noisy image (2D numpy array)
    '''
    #image should be numpy array
    np.random.seed() # sets a rondom seed
    noise = np.random.normal(loc=mean, scale=sigma, size=np.shape(image))
    return image+noise

def addPoissonNoise(image):
    '''
    Apply pure Poisson noise to image
    :param image: 2D numpy array
    :return: noisy image (2D numpy array)
    '''
    #image should be numpy array
    np.random.seed() # sets a random seed
    return np.random.poisson(image)

def addSemiPoissonNoise(image, p):
    '''
    Add semi-Poisson noise to image. This is a middle ground between Gaussian noise and pure Poisson noise. The
    intensity of the noise is tuned with the parameter p.
    :param image: 2D numpy array
    :param p: float value to tune intesity of the noise
    :return: noisy image (2D numpy array)
    '''
    np.random.seed() # sets a random seed
    noise = np.random.normal(loc=0, scale=np.sqrt(image)*p)
    return image+noise

def createHeelmapHector(mono, pixelsize, SDD, number_of_spectra):
    '''
    Creates a heel map for the Hector Scanner based on the geometry of the scan. See master thesis of
    Jorden De Bolle for mathematical details
    :param mono: monochromatic projection image (or empty numpy array) to determine size of the detector (pixels)
    :param pixelsize: size of a pixel (for new detector on Hector: 0.15mm
    :param SDD: Source to Detector Distance
    :param number_of_spectra: Number of available spectra (columns) in the simulated 2D heel spectrum
    :return: heelmap (2D numpy array)
    '''
    x = np.shape(mono)[1] #width of detector
    y = np.shape(mono)[0] #number of projections (in case of sinogram) or highth of detector (in case of projection image)
    heelmap = np.zeros([y, x])
    z = x * pixelsize * 0.5 #middle column
    alpha = 2 * np.arctan(z / SDD) * 1000  # mrad
    beta = number_of_spectra  # mrad (spectra sampled every milliradian)
    minindex = int(np.floor((beta / 2 - alpha / 2)))
    maxindex = number_of_spectra - minindex
    row = np.round(np.linspace(minindex, maxindex, x)) #vary from minindex to maxindex linearly and round off to closest int
    row = row.clip(min=1, max=number_of_spectra) #clip in case minindex <1 or maxindex >number_of_spectra
    for i in range(y):
        heelmap[i] = row #every row is the same since heel effect on HECTOR is in horizontal direction (vertical effect is small)
    heelmap = np.fliplr(heelmap) #Flip left and rifht side because tube installed upside down w.r.t. Monte Carlo simulations of spectra
    return heelmap


def gaussianFirstAxis(an_array, sigma, mode='reflect', inplace=False):
    '''
    Applies a 1D Gaussian smoothing filter on an array along the first axis
    :param an_array: numpy array that needs to be smoothed
    :param sigma: standard deviation of the Gaussian filter
    :param mode: padding mode (string) ('reflect', 'nearest', ... See scipy docs)
    :param inplace: replace original array (True) or apply filter to a copy and preserve original (False)
    :return: smoothed array
    '''
    rows = np.shape(an_array)[0]
    if inplace==False:
        smoothed = np.zeros(np.shape(an_array))
        for i in range(0, rows):
            smoothed[i] = scpim.gaussian_filter1d(an_array[i], sigma=sigma, mode=mode)
        return smoothed
    else:
        for i in range(0, rows):
            an_array[i] = scpim.gaussian_filter1d(an_array[i], sigma=sigma, mode=mode)
        return an_array

def medianFirstAxis(an_array, width, mode='reflect', inplace=False):
    '''
    Applies a 1D median filter on an array along the first axis
    :param an_array: numpy array that needs to be smoothed
    :param width: window width of the filter
    :param mode: padding mode (string) ('reflect', 'nearest', ... See scipy docs)
    :param inplace: replace original array (True) or apply filter to a copy and preserve original (False)
    :return: smoothed array
    '''
    rows = np.shape(an_array)[0]
    if inplace == False:
        smoothed = np.zeros(np.shape(an_array))
        for i in range(0, rows):
            smoothed[i] = scpim.median_filter(an_array[i], width, mode=mode)
        return smoothed
    else:
        for i in range(0, rows):
            an_array[i] = scpim.median_filter(an_array[i], width, mode=mode)
        return an_array

def meanFirstAxis(an_array, width, mode='reflect', inplace=False):
    '''
    Applies a 1D mean filter on an array along the first axis
    :param an_array: numpy array that needs to be smoothed
    :param width: window width of the filter
    :param mode: padding mode (string) ('reflect', 'nearest', ... See scipy docs)
    :param inplace: replace original array (True) or apply filter to a copy and preserve original (False)
    :return: smoothed array
    '''
    rows = np.shape(an_array)[0]
    if inplace == False:
        smoothed = np.zeros(np.shape(an_array))
        for i in range(0, rows):
            smoothed[i] = scpim.uniform_filter(an_array[i], width, mode=mode)
        return smoothed
    else:
        for i in range(0, rows):
            an_array[i] = scpim.uniform_filter(an_array[i], width, mode=mode)
        return an_array

def interpolate(x1, x2, y1):
    '''
    1D linear interpolation function
    :param x1: 1D array with x-values for which y-value is known
    :param x2: 1D array with x-values for which y-values are unknown
    :param y1: 1D array with y-values corresponding to x1 (y1 = f(x1))
    :return: y-values (y2) at x-values of x2
    '''
    f = scpip.interp1d(x1, y1)
    y2 = f(x2)
    return y2

def interpolateLog(x_original, y_original, x_target):
    '''
    1D logarithmic interpolation function, based on C++ code of Arion
    :param x_original: 1D array with x-values for which y-value is known
    :param y_original: 1D array with y-values corresponding to x1 (y_original = f(x_original))
    :param x_target: 1D array with x-values for which y-values are unknown
    :return: y-values (out) at x-values of x_target
    '''
    out = np.zeros(np.shape(x_target))
    for i in range(len(x_target)):
        if x_target[i] <= x_original[0]:
            out[i] = y_original[0]
        elif x_target[i] >= x_original[-1]:
            out[i] = y_original[-1]
        else:
            j = 0
            while x_target[i] > x_original[j]:
                j += 1
            j -= 1
            x1 = x_original[j]
            y1 = y_original[j]
            x2 = x_original[j+1]
            y2 = y_original[j+1]
            x = x_target[i]
            y_out = (np.log10(y1) + (np.log10(y2) - np.log10(y1)) * (np.log10(x) - np.log10(x1)) / (np.log10(x2) - np.log10(x1)))
            out[i] = np.power(10.0, y_out)
    return out

def getProjectedRaylength(sinogram):
    '''
    Turns a monochromatic sinogram or projection image into a pseudo sinogram or projection image that contains thickness information
    :param sinogram: monochromatic sinogram of projection image that contains exp(-d) with d distance travalled through sample
                        (created with CTRex for example)
    :return: pseudo sinogram or projection image with in every pixel d
    '''
    return -np.log(sinogram)

def getBinWidths(energies):
    '''
    Calculate widths of energy bins
    :param energies: numpy array of energy values at which spectrum is sampled
    :return: list that contains the width of the energy bins (1 element shorter than energies)
    '''
    widths = []
    for i in range(1, np.shape(energies)[0]):
        diff = energies[i] - energies[i-1]
        widths.append(diff)
    return widths

def calcSolidAngle(SDD, pixelsize, pixelx, pixely): #SDD and pixelsize in mm
    '''
    Calculate the solid angle of a pixel of the detector based on its location w.r.t. the X-ray source
    :param SDD: Source to Detector Distance (in mm)
    :param pixelsize: size of the pixel (in mm)
    :param pixelx: x-coordinate of the pixel (expressed in pixels) where origin is at the middle of the detecor
    :param pixely: y-coordinate of the pixel (expressed in pixels) where origin is at the middle of the detecor
    :return: solid angle of the pixel expressed in steradians
    '''
    angle = np.arctan((np.sqrt((pixelx * pixelsize)**2 + (pixely*pixelsize)**2))/SDD)
    x = SDD/np.cos(angle)
    solidangle = pixelsize**2 * np.cos(angle) / x**2
    return solidangle

def RMSE(array1, array2):
    '''
    Calculates the root mean squared error (RMSE) between two numpy arrays
    :param array1: numpy array
    :param array2: numpy array
    :return: RMSE (float)
    '''
    error = np.power(array1-array2, 2)
    error = np.sum(error)/array1.size
    error = np.sqrt(error)
    return error

@jit #just in time compilation by Numba for fast execustion
def NormaliseProjection(projection, IO):
    '''
    Normalize a simulated projection image or sinogram by dividing by a flat field
    :param projection: raw image
    :param IO: flat field image
    :return: normalized image
    '''
    # clip bad values due to noise
    projection[projection <= 0.0] = 1.0  # 1, not 0, otherwise infinity in log
    # normalisation
    dev = np.divide(projection, IO)
    # clip bad values due to noise
    dev[dev > 1.0] = 1.0
    return dev

@jit #just in time compilation by Numba for fast execustion
def NormaliseExperimental(projection, IO, dark):
    '''
    Normalize a simulated projection image or sinogram by subtracting a dark field image and dividing by a flat field image
    :param projection: raw image
    :param IO: flat field image
    :param dark: dark field image
    :return: normalized image
    '''
    # no clipping of bad values (bad values due to scattering and must be taken into account)
    top = np.subtract(projection, dark)
    bottom = np.subtract(IO, dark)
    norm = np.divide(top, bottom)
    return norm
