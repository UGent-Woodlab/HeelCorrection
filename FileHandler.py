'''
Written by Jorden De Bolle
Last update: 18/07/2022
'''
import numpy as np
from libtiff import TIFF, TIFFimage
from PIL import Image
import os


datafolder = r'C:\Users\jdebolle\Documents\PythonProjects\HeelingCode_final\Data'
savefolder = r'C:\Users\jdebolle\Documents\PythonProjects\HeelingCode_final\Projections'

def im2array(filename):
    '''
    Opens tiff images
    :param filename: path to tiff file that needs to be opened
    :return: numpy array that contains tiff image
    '''
    tif = TIFF.open(filename, mode='r')
    temp = tif.read_image()
    descr = tif.GetField('ImageDescription')
    #print(descr)
    if (descr is not None):
        s = descr.split()
        slope = np.float32(s[2])
        offset = np.float32(s[5])
        temp = temp * slope + offset
    tif.close()
    return temp #temp is np.array

def im2array2(filename, slope, offset):
    '''
    Opens tiff images
    :param filename: path to tiff file that needs to be opened
    :param slope: slope of tiff
    :param offset: offset of tiff
    :return: numpy array that contains tiff image
    '''
    temp = Image.open(filename)
    temp = np.array(temp)
    temp = temp * slope + offset
    return temp #temp is np.array

def tex2array(filename):
    '''
    Reads a text file and puts data into numpy array
    :param filename: path to text file
    :return: numpy array with data from text file
    '''
    data = np.loadtxt(filename)
    return data

def array2tex(filename, arr):
    '''
    Writes data from a numpy array to a text file
    :param filename: name of the file where data is saved
    :param arr: numpy array with data
    :return: None
    '''
    a_file = open(filename, "wb")
    np.savetxt(a_file, arr)
    a_file.close()

def array2image(filename, arr):
    '''
    Writes data from numpy array to tiff image
    :param filename: name of the tiff file
    :param arr: numpy array with data
    :return: None
    '''
    arr = np.array(arr, dtype=np.uint16)
    im = Image.fromarray(arr, mode='I;16')
    im.save(filename, astype=np.uint16)

def getFormat(tiffile):
    #Code copied from CTRex
    """!
    Get the prefix and format of a file without the directory.

    Example: prefix_Z_000186.tif will return prefix_Z_ and '%s%05d.tif'
    @param tiffile: The filename of which you want to know prefix and format. Can be a full path to this file.
    @return prefix and format.
    """
    filename, file_extension = os.path.splitext(os.path.basename(tiffile))
    length = len(filename)
    i = length
    while i > 0 and filename[i - 1].isdigit():
        i -= 1
    prefix = filename[0:i]
    fmt = '%s%0' + str(length-i) + 'd' + file_extension
    return prefix, fmt


def getSlopeOffset(limits, dtype=np.uint8):
    # Code copied from CTRex
    """!
    Calculates slope and offset, given the limits on which to clip the volume data.

    @param limits: [minimum,maximum] values used to clip the volume data.
    @param dtype: The dtype of the eventual tiff file.
    @return slope, offset
    """
    try:
        return float(limits[1] - limits[0]) / np.iinfo(dtype).max, float(limits[0])
    except ValueError:  # float dtype?
        return 1., float(limits[0])


def makeDescription(slope, offset):
    # Code copied from CTRex
    """!
    Returns a description to go with a tiff file scaled according to the given slope and offset.

    realvalue = tiffvalue * slope + offset
    @param slope: Tiff file was scaled according to this slope.
    @param offset:  Tiff file was scaled according to this offset.
    @return The description (string).
    """
    return 'slope = %6.5E offset = %6.5E' % (slope, offset)


def coalesce(*arg):
    # Code copied from CTRex
    # type: #(Any1) -> Optional[Any1]
    """!
    Return first non-None argument or None if all arguments are None.

    @param arg: Arguments to search between.
    @return First non-None argument.
    """
    for el in arg:
        if el is not None:
            return el
    return None

def scale(data, limits=None, dtype=np.uint8, inplace=False, returndescription=False):
    # Code copied from CTRex
    """!
    Given 'real' (float) volume data, scale it to fit into dtype (ints).

    This function can be used to prepare data to write out as tiff files.
    Input data will typically be floats between 0 and 1 (or whatever is in limits).
    This function clips the data to these limits and then scales it so that
    the minimum limit becomes 0 and the maximum limit becomes the maximum value
    of dtype (255 for np.uint8).
    Data is put astype(dtype) at the end.

    Returning the description may be useful for the tiff files if you write this data out.

    @param data: A 3D numpy array containing volume data.
    @param limits: [minimum,maximum] to which to clip the data.
    @param dtype: The output dtype the data must fit in.
    @param inplace: scale back in place if float to uint else make a temp array (doubles mem needs!)
    @param returndescription: Whether to return the description for the reverse scaling.
    @return The scaled 3D array (and description if returndescription)
    """
    limits = coalesce(limits, [0., 1.])
    dtype = np.dtype(dtype)
    # noinspection PyTypeChecker
    slope, offset = getSlopeOffset(limits, dtype)
    if data.dtype in (
    float, np.float16, np.float32, np.float64):  # when you store them as floats, they usually are scaled
        if inplace:  # changes original data
            data -= offset
            data /= slope
        else:
            data = (data - offset) / slope  # copy does not affect original data
        out = np.rint(data)
        np.clip(out, 0, np.iinfo(dtype).max, out=out)
    elif data.dtype != dtype:  # ints but not the same
        out = data * np.iinfo(dtype).max / np.iinfo(data.dtype).max
    else:
        out = data
    out = out.astype(dtype)
    if returndescription:
        return out, makeDescription(slope, offset)
    return out


def writeSliceToDisk(slc, dstfilename, limits, dtype=np.uint8, description=None):
    #Code copied from CTRex
    """!
    Given a slice, write this to the specified position on disk.

    The slice is first scaled according to limits to fit into the
    given output dtype. Then it is written to disk as a tiff file.
    @param slc: The slice to write to disk (2D array).
    @param dstfilename: The path to which to write (string)
    @param limits: The limits with which to scale the slice.
    @param dtype: The output dtype.
    @param description: If this is None, the slice still needs to be scaled. Else this is probably called by writeDataToDisk
    @return nothing
    """
    if slc is None:
        print("You are trying to write None to disk!")
        return

    # scale slice
    if description is None:
        out, description = scale(slc, limits, dtype, returndescription=True)
    else:
        out = slc

    # write away:
    dirname = os.path.dirname(dstfilename)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)

    if getFormat(dstfilename)[1].endswith('.tif'):
        tiff = TIFFimage(out, description=description)
        tiff.write_file(dstfilename, compression='none', verbose=False)
        del tiff
    else:
        img = Image.fromarray(out)
        img.save(dstfilename)
        del img

def array2image2(filename, arr):
    '''
    Writes data from numpy array to tiff image. This function is based on code from CTRex and should be better than
    the previous one.
    :param filename: name of the tiff file
    :param arr: numpy array with data
    :return: None
    '''
    writeSliceToDisk(arr, filename, limits=[np.min(arr), np.max(arr)], dtype=np.uint16)


def loadSpectrum(tube, voltage):
    '''
    Calculates the correct 1D (no heeling) spectrum as the mean of the 2D heel spectrum from the correct file.
    Make sure the name of the tube is correct and the voltage is available in the data files
    :param tube: used X_ray tube (string)
    :param voltage: used tube voltage (integer)
    :return: numpy array with values for photon energy where spectrum is sampled, numpy array with number of photons per
    electron per kV and per steradian for each photon energy
    '''
    file = datafolder + '\\SpectrumFilesHeelSmoothed\\' + tube + '_W_' + str(voltage) + '.spec'
    data = tex2array(file)
    energies = np.transpose(data)[0]
    index = np.argmax(energies>=1.0) #necessary for interpolations later
    energies = energies[index:]
    data = data[index:, 1:]
    data = np.mean(data, axis=1)
    return energies, data

def loadSpectraHeel(tube, voltage):
    '''
    Reads the correct 2D (with heeling) spectrum from the correct file. Make sure the name of the
    tube is correct and the voltage is available in the data files. Currently only works for 'Hector' tube.
    :param tube: used X_ray tube (string)
    :param voltage: used tube voltage (integer)
    :return: numpy array with values for photon energy where spectrum is sampled, numpy array with number of photons per
    electron per kV and per steradian for each photon energy and for each beam angle (angle from left to right in the
    conical X_ray beam) from 1 to 499 mrad in steps of 1 mrad.
    '''
    file = datafolder + '\\SpectrumFilesHeelSmoothed\\' + tube + '_W_' + str(voltage) + '.spec'
    data = tex2array(file)
    energies = np.transpose(data)[0]
    index = np.argmax(energies>=1.0) #necessary for interpolations later
    energies = energies[index:]
    data = data[index:, 1:]
    return energies, data

def loadAttenuations(material):
    '''
    Reads the mass attenuation coefficient of material from the correct file
    :param material: name of the material (string)
    :return: numpy array with photon energies for which the mass attenuation coefficient is sampled, numpy array with
    the mass attenuation coefficient (in 1/cm) for every photon energy
    '''
    file = datafolder + '\\AttenuationFiles\\' + material + '.att'
    data = tex2array(file)
    energies = data[:, 0]
    index = np.argmax(energies >= 1.0)  # necessary for interpolations later
    energies = energies[index:]
    data = data[index:, 3] + data[index:, 2] #coherent scattering not included
    #see e.g. https://physics.nist.gov/cgi-bin/Xcom/xcom3_1
    return energies, data

def loadDetectorResponse(detector):
    '''
    Reads the response of the detector from the correct file
    :param detector: name of the used detector (string)
    :return: numpy array with photon energies for which the response is sampled, numpy array with
    the detector response (in keV) for every photon energy
    '''
    file = datafolder + '\\ResponseFiles\\' + detector + '.resp'
    data = tex2array(file)
    energies = data[:,0]
    index = np.argmax(energies>=1.0) #necessary for interpolations later
    energies = energies[index:]
    data = data[index:, 1]
    return energies, data
