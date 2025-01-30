'''
Written by Jorden De Bolle
Last update: 18/07/2022
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
import Polychromatic as poly
import Functions as fun
import time
import FileHandler as FH
from scipy import signal
from numba import njit, prange, get_num_threads, set_num_threads, cuda, float32, int16
import numba
import os
import re
import math
import shutil

numba.config.THREADING_LAYER = 'safe'

#general
def ProjectWedge(a, b, SOD, SDD, detector_height, detector_width, pixelsize, height_pitch=0):
  '''
  Calculates the projected thickness of the wedge for a conical beam (pseudo-projection image). The equations are derived
  in appendix A of the master thesis of Jorden De Bolle. The wedge should be placed in the system with the flat side facing
  the source.
  :param a: width of the wedge (maximal thickness, thickness at the bottom)
  :param b: height of the wedge
  :param SOD: source to object distance in mm(distance between source and flat side of the wedge (wedge plane closest to source))
  :param SDD: source to detector distance in mm
  :param detector_height: height of detector in pixels
  :param detector_width: width of detector in pixels
  :param pixelsize: size of a pixel
  :param height_pitch: downwards shift of the wedge in mm. In general this is 0 mm, then wedge is placed on top of optical axis
  :return: pseudo-projection of wedge that contains in every pixel the distance travelled through the wedge by ray that hits pixel
  '''
  # a = width in mm
  # b = heighth in mm
  # detector_height and width in pixels
  # height_pitch=0 --> wedge on top of optical axis
  # height_pitch > 0: shift wedge down
  # height_pitch expressed in mm
  projection = np.zeros([detector_height, detector_width])
  for i in range(detector_height):
    for j in range(detector_width):
      pixx = ((detector_height/2 - i))*pixelsize
      pixy = -((detector_width/2 - j))*pixelsize

      #first intersection point
      t1 = 1/b * (-1+(SOD)/(SOD+a))*pixx
      t1 = t1 - SDD/(SOD+a)
      t2 = -1-height_pitch*(1/b)*(-1+(SOD)/(SOD+a))
      t = t2/t1
      x0 = pixx*t
      y0 = pixy*t
      z0 = SDD*t

      #second intersection point
      t = SOD/SDD
      x1 = pixx*t
      y1 = pixy*t
      z1 = SDD*t

      if z1 < z0 and x0>=-height_pitch:
        d = np.sqrt((x1-x0)**2 + (y1-y0)**2 + (z1-z0)**2) * 0.1 #*0.1 because we need it in cm
        projection[i, j] = d
      else:
        projection[i, j] = 0 #to account for the fact that the planes that define the wedge are mathematically infinite but the wedge is not
  return projection

#simulated LUT
def CreateLUTSimulation(a, b, SOD, SDD, detector_height, detector_width, LUT_height, pixelsize, tube, detector, voltage, material, density, filter=False, filtermaterial='Al', filterdensity=2.7, filterthickness=1.0, height_pitch=0, splits=1):
  '''
  Creates a simulated LUT: calculation of the projected thickness of the wedge and simulation of the polychromatic projection of the wedge and its processing.
  The simulation of the projection of the wedge uses a lot of RAM so the calculation is split in several parts.
  :param a: maximal thickness of the wedge (thickness at the bottom) (in mm)
  :param b: height of the wedge (in mm)
  :param SOD: source to objecte distance (distance from source to flat vertical plane of wedge, wedge is placed in scanner such that vertical plane faces the source)
  :param SDD: source to detector distance (should be same in scan as in projection of wedge in order to have same heel effect!!!)
  :param detector_height: height of the detector in pixels as used for the projection images in the data set that is being corrected
  :param detector_width: width of the detector in pixels as used for the projection images in the data set that is being corrected
  :param LUT_height: eventual height of the LUT that will be used (for example 1010 pixels if projection of wedge only uses top 1000 pixels of detector)
  :param pixelsize: size of a pixel (in mm)
  :param tube: name of the used scanner (currently only 'Hector' is available
  :param detector: name of the used detector
  :param voltage: used tube voltage (will be used in the simulation of the projection of the wedge)
  :param material: material of the wedge (for example 'POM')
  :param density: mass density of the wedge (in g/cm^3)
  :param filter: bool, is spectrum filtered or not
  :param filtermaterial: material of the used filter
  :param filterdensity: mass density of the filter material
  :param filterthickness: thickness of the filter (in mm)
  :param height_pitch: downwards shift of the wedge in mm (if 0 mm, then wedge is on top of optical axis)
  :param splits: The simulation of the projection of the wedge uses a lot of RAM, so the calculation is split (best use something like 10 splits)
  :return: two 2D numpy arrays, one with normalized and processed projection of the wedge, one with projected thickness of the wedge, these together form the LUT
  '''
  wedge_raylengths = ProjectWedge(a, b, SOD, SDD, detector_height, detector_width, pixelsize, height_pitch)
  solidangles = poly.calcSolidAnglesProjection(wedge_raylengths, pixelsize, SDD)
  Energies, spec, heelmap, response = poly.InitialiseScannerHeel(wedge_raylengths, tube, detector, voltage, pixelsize, SDD)
  atts = poly.InitialiseObject(material, Energies)

  if filter==True:
    spec = poly.FilterSpectrumHeel(filtermaterial, filterdensity, filterthickness, spec, Energies)

  #split in different parts to save RAM during simulation of wedge projection image
  increment = int(detector_width/splits)
  rest = detector_width%splits
  polyprojection = np.zeros([detector_height, detector_width])
  for i in range(splits-1):
    wrl_part = wedge_raylengths[:, i*increment:(i+1)*increment]
    solidangles_part = solidangles[:, i*increment:(i+1)*increment]
    heelmap_part = heelmap[:, i*increment:(i+1)*increment]
    multi = poly.singleObjectAttenuations(atts, density, wrl_part, exp=False)
    polyprojection[:, i*increment:(i+1)*increment] = poly.makePolyHeelSino(multi, Energies, spec, heelmap_part, response, solidangles_part, 1, 0.1, voltage, 0.001)
  #final split is different due to rest
  wrl_part = wedge_raylengths[:, (splits-1) * increment:(splits) * increment + rest]
  solidangles_part = solidangles[:, (splits-1) * increment:(splits) * increment + rest]
  heelmap_part = heelmap[:, (splits-1) * increment:(splits) * increment + rest]
  multi = poly.singleObjectAttenuations(atts, density, wrl_part, exp=False)
  polyprojection[:, (splits-1) * increment:(splits) * increment + rest] = poly.makePolyHeelSino(multi, Energies, spec, heelmap_part, response, solidangles_part, 1, 0.1, voltage, 0.001)

  IO = poly.createIOHeelSino(wedge_raylengths, Energies, spec, heelmap, response, solidangles, 1, 0.1, voltage, 0.001)
  normalized_wedge_poly = fun.NormaliseExperimental(polyprojection, IO, 0.0) #dark field is 0.0

  normalized_wedge_poly = normalized_wedge_poly[:LUT_height, :]
  wedge_raylengths = wedge_raylengths[:LUT_height, :]

  for i in range(detector_width): #processing of every column in simulated projection image of wedge
    column = normalized_wedge_poly[:, i]
    diffs = np.zeros(len(column))
    for j in range(len(column) - 1):
      diffs[j] = column[j] - column[j + 1] #find the difference between value and next value
    maxindex = np.where(diffs == np.min(diffs))[0][0] #find most negative value, this defines edge of the wedge
    column[maxindex:] = 0.0 #from maxindex to end, value 0 is assigned for correct interpolation on edge cases later
    normalized_wedge_poly[:, i] = column

  return wedge_raylengths, normalized_wedge_poly

#experimental LUT
def CalcScatteringLUT(normalized_wedge_poly, scatter_space): # scatter_space = number of pixels above the wedge that can be used for scattering corrections
  # Calculate scatter value in experimental acquisition of wedge (normalized_wedge_poly is normalized experimental wedge projection image)
  region = normalized_wedge_poly[:scatter_space, :]
  scattering = np.mean(region) - 1.0
  return scattering

#general
def CalcScatteringNormalizedProjection(projection, scatter_space):
  # Calculates scattering value in a normalized projection image of the data set. This value is later subtracted.
  # scatter_space defines region in which scattering is calculated: [x_1, x_2, y_1, y_2]
  x_1 = scatter_space[0]
  x_2 = scatter_space[1]
  y_1 = scatter_space[2]
  y_2 = scatter_space[3]
  proj = projection[y_1:y_2, x_1:x_2]
  scattering = np.mean(proj) - 1.0
  return scattering

#experimental LUT
def DenoiseColumn(column):
  '''
  When using a simulated LUT, the normalized projection of the wedge contains noise. This needs to be smoothed
  and this is done using a Savitsky-Golay filter. Below the wedge, all values in the projection image are set
  to 0 for correct interpolation at edge cases later. This step is performed column by column. This function performs
  the operations on one column.
  '''
  diffs = np.zeros(len(column))
  for i in range(len(column) - 1):
    diffs[i] = column[i] - column[i + 1] #find the difference between value and next value
  # at bottom of wedge, value goes from small (thickest part of wedge) to large again (no more wedge) so diff becomes negative
  maxindex = np.where(diffs == np.min(diffs))[0][0] #find most negative value, this defines edge of the wedge
  yhat = signal.savgol_filter(column[:maxindex], 51, 3) #smooth column with Savitsky-Golay filter
  fitted = np.zeros(len(column))
  fitted[:maxindex] = yhat #result is smoothed column until maxindex
  fitted[maxindex:] = 0.0 #from maxindex to end, value 0 is assigned for correct interpolation on edge cases later
  return fitted

#experimental LUT
def CreateLUTExperimental(wedge_poly_projection, IO, dark, a, b, SOD, SDD, detector_height, detector_width, LUT_height, pixelsize, scatter_space, height_pitch=0):
  '''
  Prepares a lookup table starting from a raw projection image from a wedge, flat field , dark field and configuration of the wedge in the scanner.
  :param wedge_poly_projection: raw projection image of the wedge
  :param IO: flat field image to normalize the wedge
  :param dark: dark field image to normalize the wedge
  :param a: thickness of the wedge (at the bottom)
  :param b: height of the wedge
  :param SOD: source to object distance (defined as dinstance from source to flat vertial plane of the wedge. This plane must face the source.)
  :param SDD: source to detector distance
  :param detector_height: height of the detector used for the projections in the data set that needs to be corrected in pixels
  :param detector_width: width of the detector used for the projections in the data set that needs to be corrected in pixels
  :param LUT_height: How high (in pixels) will the LUT eventually be (for example, if detector is 2000 pixels high but wedge only uses top half, 1010 pixels could be a good value here)
  :param pixelsize: size of a pixel in mm
  :param scatter_space: number of pixels in the vertical direction above the wedge that are available for scattering correction
  :param height_pitch: downwards shift of the wedge in mm (wedge on top of optical axis if pitch is 0 mm)
  :return: two 2D numpy arrays, one with normalized and processed projection of the wedge, one with projected thickness of the wedge, these together form the LUT
  '''
  normalized_wedge_poly = fun.NormaliseExperimental(wedge_poly_projection, IO, dark) #normalization
#  if np.shape(wedge_poly_projection)[0] != detector_width: #LUT and projections that need to be corrected need to represent the same region of the detector # CODE DIT NOT WORK BECAUSE OF THIS
#    left = int((np.shape(wedge_poly_projection)[0] - detector_width)/2)
#    right = np.shape(wedge_poly_projection)[0] - left
#    normalized_wedge_poly = normalized_wedge_poly[:, left:right] 
  scattering = CalcScatteringLUT(normalized_wedge_poly, scatter_space) #calculate scattering calue
  normalized_wedge_poly = normalized_wedge_poly - scattering # correct for scattering
  for i in range(detector_width): #smooth and preprocess every column (all values 0 below the wedge)
    normalized_wedge_poly[:, i] = DenoiseColumn(normalized_wedge_poly[:, i])
  normalized_wedge_poly = normalized_wedge_poly[:LUT_height, :] #cut unnecessary part of the LUT
  wedge_raylengths = ProjectWedge(a, b, SOD, SDD, detector_height, detector_width, pixelsize, height_pitch) #caclculate projected thickness
  wedge_raylengths = wedge_raylengths[:LUT_height, :] #cut unnecessary part of LUT
  return wedge_raylengths, normalized_wedge_poly

#general
@njit
def LookUp(wedge_raylengths, normalized_wedge_poly, normalized_intensity, column_idx):
  '''
  Finds the closest transmission value from in the normalized projection image that is being corrected in the LUT.
  Performs linear interpolation between to closest intensities in LUT to find best fitting thickness.
  !!!This function only runs on CPU!!!
  :param wedge_raylengths: part of LUT that contains projected thickness of wedge
  :param normalized_wedge_poly: part of LUT that contains (processed) projection image of wedge
  :param normalized_intensity: transmission value from pixel in projection image that is being corrected
  :param column_idx: index of the column in the projection image of the pixel from which normalized_intensity originates
  :return: distance travelled through wedge that corresponds with the to be corrected transmission value
  '''
  column_poly = normalized_wedge_poly[:, column_idx]
  column_rls = wedge_raylengths[:, column_idx]
  next_idx = np.argmax(column_poly <= normalized_intensity) #the largest of the values smaller than normalized_intensity
  #if next_idx == None or next_idx == 0 or next_idx >= np.size(column_poly): #deal with boundaries and intensities = 1.0
  #  d = 0
  #else:
  #!!! if-statement does not work when using njit from numba -->found a workaround by interpolation in specific way:
  next = column_poly[next_idx]
  prev = column_poly[next_idx-1]
  frac = np.abs((normalized_intensity-prev)/(next-prev)) #fraction of distance between prev en next where transmission value from projection image is found
  d = column_rls[next_idx] - (1-frac)*np.abs(column_rls[next_idx-1] - column_rls[next_idx]) #linear interpolation to find best corresponding distance travelled through wedge
  return d

#general
@njit()
def CorrectNormalizedProjection(wedge_raylengths, normalized_wedge_poly, normalized_projection):
  '''
  Performs the correction by the LUT for every pixel in a normalized projection image.
  !!!This function only runs on CPU!!!
  :param wedge_raylengths: part of LUT that contains projected thickness of wedge
  :param normalized_wedge_poly: part of LUT that contains (processed) projection image of wedge
  :param normalized_projection: normalized projection that is being corrected
  :return: corrected projection image
  '''
  set_num_threads(6) #in case of parallelization on CPU
  rows, columns = np.shape(normalized_projection)
  corrected_projection = np.zeros_like(normalized_projection) #create empty corrected projection
  # iterate over all pixels in projection image
  for i in range(rows):
    for j in range(columns):
      value = normalized_projection[i, j] #transmission value that needs to be corrected
      corrected_projection[i, j] = LookUp(wedge_raylengths, normalized_wedge_poly, value, j) #find corrected value in LUT
  corrected_projection = np.multiply(corrected_projection, -1) #make negative (-d)
  corrected_projection = np.exp(corrected_projection) #take exponential (exp(-d))
  return corrected_projection

#general
@cuda.jit(device=True)
def first_smaller_value(value, device_array):
  '''
  CUDA device function to find largest value in a 1D array that is smaller than a given value.
  This function is used in the GPU version of the LUT.
  :param value: given value
  :param device_array: array in which first smaller value needs to be found
  :return: index of first smaller value in the array
  '''
  val = device_array[0]
  idx = 0
  while val > value:
    idx += 1
    val = device_array[idx]
  return idx

#general
@cuda.jit
def CorrectNormalizedProjectionGPU(wrl_gpu, nwp_gpu, norm_gpu):
  '''
  This function performs the LookUp operation, interpolation and correction of every pixel in a projection image on the GPU.
  :param wrl_gpu: array in the GPU memory that contains the projected thickness of the wedge
  :param nwp_gpu: array in the GPU memory that contains the normalized and processed projection of the wedge
  :param norm_gpu: array in the GPU memory that contains the normalized projection image that needs to be corrected
  '''
  x, y = cuda.grid(2)
  if x < norm_gpu.shape[0] and y < norm_gpu.shape[1]: #this is how all pixels are traveresed in parallel by threads on the GPU
    normalized_intensity = norm_gpu[x, y]
    next_idx = first_smaller_value(normalized_intensity, nwp_gpu[:, y]) #CUDA device function
    next = nwp_gpu[next_idx, y]
    prev = nwp_gpu[next_idx-1, y]
    frac = abs((normalized_intensity - prev) / (next - prev)) #fraction of distance between prev en next where transmission value from projection image is found
    d = wrl_gpu[next_idx, y] - (1.0 - frac) * abs(wrl_gpu[next_idx - 1, y] - wrl_gpu[next_idx, y]) #linear interpolation to find best corresponding distance travelled through wedge
    norm_gpu[x, y] = math.exp(-1.0*d) #replace old value in projection by corrected value
    cuda.syncthreads()

#experimental LUT
def CorrectNormalizedDatasetExperimental(datafolder, scanprefix, savefolder, wedge_poly_file, wedge_IO_file, wedge_dark_file, a, b, SOD, SDD, detector_height, detector_width, LUT_height, pixelsize, scatter_space_LUT, scatter_space_projections, height_pitch=0.0):
  '''
  Performs the correction with an experimental LUT on CPU on a full data set. Also prints time that was needed for the total correction at the end in seconds.

  :param datafolder: path to folder with normalized projection images that need to be corrected
  :param scanprefix: prefix in the names of the normalized projection images
  :param savefolder: path to folder where corrected images need to be saved
  :param wedge_poly_file: file with raw projection image of the wedge
  :param wedge_IO_file:  file with flat field image for normalizing the wedge projection
  :param wedge_dark_file: file with dark field image for normalizing the wedge projection
  :param a: maximal thickness of the wedge (thickness at the bottom) (in mm)
  :param b: height of the wedge (in mm)
  :param SOD: source to objecte distance (distance from source to flat vertical plane of wedge, wedge is placed in scanner such that vertical plane faces the source)
  :param SDD: source to detector distance (should be same in scan as in projection of wedge in order to have same heel effect!!!)
  :param detector_height: height of the detector in pixels as used for the projection images in the data set that is being corrected
  :param detector_width: width of the detector in pixels as used for the projection images in the data set that is being corrected
  :param LUT_height: eventual height of the LUT that will be used (for example 1010 pixels if projection of wedge only uses top 1000 pixels of detector)
  :param pixelsize: size of a pixel in mm
  :param scatter_space_LUT: height in pixels available above the wedge to calculate scattering in projection of the wedge
  :param scatter_space_projections: [x_1, x_2, y_1, y_2], defines a region in the projection images that is used for calculating the scattering value
  :param height_pitch: downwards shift of the wedge in mm (if 0 mm, then wedge is on top of optical axis)
  '''
  wedge_poly = FH.im2array(wedge_poly_file) #open raw experimental projection of wedge
  wedge_io = FH.im2array(wedge_IO_file) #open flat field
  wedge_dark = FH.im2array(wedge_dark_file) #open dark field
  #prepare experimental LUT:
  wrl, nwp = CreateLUTExperimental(wedge_poly, wedge_io, wedge_dark, a, b, SOD, SDD, detector_height, detector_width, LUT_height, pixelsize, scatter_space_LUT, height_pitch)

  #make list of files that need to be corrected
  datfiles = [f for f in os.listdir(datafolder) if re.match(scanprefix + r'*', f)]

  numfiles = len(datfiles)
  for i, datfile in enumerate(datfiles): #iterate over all files
    print('Correcting projection {} of {}'.format(i+1, numfiles), end='\r')
    proj = FH.im2array(datafolder + '\\' + datfile) #open normalized projection image that needs to be corrected
    scattering = CalcScatteringNormalizedProjection(proj, scatter_space_projections) #calculate scattering
    proj = proj - scattering #scattering correction
    corr = CorrectNormalizedProjection(wrl, nwp, proj) #correct the projection image on CPU
    FH.array2image2(savefolder+ '\\' + datfile, corr) #save the corrected image

#experimental LUT
def CorrectNormalizedDatasetExperimentalGPU(datafolder, scanprefix, savefolder, wedge_poly_file, wedge_IO_file, wedge_dark_file, a, b, SOD, SDD, detector_height, detector_width, LUT_height, pixelsize, scatter_space_LUT, scatter_space_projections, height_pitch=0.0):
  '''
  Performs the correction with an experimental LUT on GPU on a full data set. Also prints time that was needed for the total correction at the end in seconds.

  :param datafolder: path to folder with normalized projection images that need to be corrected
  :param scanprefix: prefix in the names of the normalized projection images
  :param savefolder: path to folder where corrected images need to be saved
  :param wedge_poly_file: file with raw projection image of the wedge
  :param wedge_IO_file:  file with flat field image for normalizing the wedge projection
  :param wedge_dark_file: file with dark field image for normalizing the wedge projection
  :param a: maximal thickness of the wedge (thickness at the bottom) (in mm)
  :param b: height of the wedge (in mm)
  :param SOD: source to objecte distance (distance from source to flat vertical plane of wedge, wedge is placed in scanner such that vertical plane faces the source)
  :param SDD: source to detector distance (should be same in scan as in projection of wedge in order to have same heel effect!!!)
  :param detector_height: height of the detector in pixels as used for the projection images in the data set that is being corrected
  :param detector_width: width of the detector in pixels as used for the projection images in the data set that is being corrected
  :param LUT_height: eventual height of the LUT that will be used (for example 1010 pixels if projection of wedge only uses top 1000 pixels of detector)
  :param pixelsize: size of a pixel in mm
  :param scatter_space_LUT: height in pixels available above the wedge to calculate scattering in projection of the wedge
  :param scatter_space_projections: [x_1, x_2, y_1, y_2], defines a region in the projection images that is used for calculating the scattering value
  :param height_pitch: downwards shift of the wedge in mm (if 0 mm, then wedge is on top of optical axis)
  '''
  wedge_poly = FH.im2array(wedge_poly_file) #open raw experimental projection of wedge
  wedge_io = FH.im2array(wedge_IO_file) #open flat field
  wedge_dark = FH.im2array(wedge_dark_file) #open dark field
  #prepare experimental LUT:
  wrl, nwp = CreateLUTExperimental(wedge_poly, wedge_io, wedge_dark, a, b, SOD, SDD, detector_height, detector_width, LUT_height, pixelsize, scatter_space_LUT, height_pitch)

  #copy LUT to the GPU device momory:
  wrl_gpu = cuda.to_device(wrl)
  nwp_gpu = cuda.to_device(nwp)

  # make list of files that need to be corrected
  datfiles = [f for f in os.listdir(datafolder) if re.match(scanprefix + r'*', f)]

  #shape of the projection images, used by gpu to launch parallel threads
  shape0 = np.shape(FH.im2array(datafolder + '\\' + datfiles[0]))[0]
  shape1 = np.shape(FH.im2array(datafolder + '\\' + datfiles[0]))[1]

  #Initiate the CUDA kernel, this defines the number of threads that will be launched
  threadsperblock = (16, 16)
  blockspergrid_x = math.ceil(shape0 / threadsperblock[0])
  blockspergrid_y = math.ceil(shape1 / threadsperblock[1])
  blockspergrid = (blockspergrid_x, blockspergrid_y)

  numfiles = len(datfiles)
  start = time.time()
  for i, datfile in enumerate(datfiles): #iterate over all projections that need to be corrected
    print('Correcting projection {} of {}'.format(i + 1, numfiles), end='\r')
    proj = FH.im2array(datafolder + '\\' + datfile) #open normalized projection that will be corrected
    scattering = CalcScatteringNormalizedProjection(proj, scatter_space_projections) #calculate scattering
    proj = proj - scattering #scattering correction
    proj_gpu = cuda.to_device(proj) #copy projection to GPU device memory
    CorrectNormalizedProjectionGPU[blockspergrid, threadsperblock](wrl_gpu, nwp_gpu, proj_gpu) #correct projection on the GPU
    corr = proj_gpu.copy_to_host() #copy corrected projection back to RAM
    FH.array2image2(savefolder + '\\' + datfile, corr) #save corrected projection
  print(time.time() - start)



# experimental LUT with dataset normalisation, added by louis 2025-01-24
def CorrectDatasetExperimentalGPU(datafolder, scanprefix, scan_io_file, scan_dark_file, savefolder, wedge_poly_file, wedge_IO_file, wedge_dark_file, a, b, SOD, SDD, detector_height, detector_width, LUT_height, pixelsize, scatter_space_LUT, scatter_space_projections, height_pitch=0.0):
  '''
  Performs the correction with an experimental LUT on GPU on a full data set. Also prints time that was needed for the total correction at the end in seconds.

  :param datafolder: path to folder with normalized projection images that need to be corrected
  :param scanprefix: prefix in the names of the normalized projection images
  :param savefolder: path to folder where corrected images need to be saved
  :param wedge_poly_file: file with raw projection image of the wedge
  :param wedge_IO_file:  file with flat field image for normalizing the wedge projection
  :param wedge_dark_file: file with dark field image for normalizing the wedge projection
  :param a: maximal thickness of the wedge (thickness at the bottom) (in mm)
  :param b: height of the wedge (in mm)
  :param SOD: source to objecte distance (distance from source to flat vertical plane of wedge, wedge is placed in scanner such that vertical plane faces the source)
  :param SDD: source to detector distance (should be same in scan as in projection of wedge in order to have same heel effect!!!)
  :param detector_height: height of the detector in pixels as used for the projection images in the data set that is being corrected
  :param detector_width: width of the detector in pixels as used for the projection images in the data set that is being corrected
  :param LUT_height: eventual height of the LUT that will be used (for example 1010 pixels if projection of wedge only uses top 1000 pixels of detector)
  :param pixelsize: size of a pixel in mm
  :param scatter_space_LUT: height in pixels available above the wedge to calculate scattering in projection of the wedge
  :param scatter_space_projections: [x_1, x_2, y_1, y_2], defines a region in the projection images that is used for calculating the scattering value
  :param height_pitch: downwards shift of the wedge in mm (if 0 mm, then wedge is on top of optical axis)
  '''
  wedge_poly = FH.im2array(wedge_poly_file) #open raw experimental projection of wedge
  wedge_io = FH.im2array(wedge_IO_file) #open flat field
  wedge_dark = FH.im2array(wedge_dark_file) #open dark field
  #prepare experimental LUT:
  wrl, nwp = CreateLUTExperimental(wedge_poly, wedge_io, wedge_dark, a, b, SOD, SDD, detector_height, detector_width, LUT_height, pixelsize, scatter_space_LUT, height_pitch)

  #copy LUT to the GPU device momory:
  wrl_gpu = cuda.to_device(wrl)
  nwp_gpu = cuda.to_device(nwp)

  # make list of files that need to be corrected
  datfiles = [f for f in os.listdir(datafolder) if re.match(scanprefix + r'*', f)]
    
  # load scan IO and DI
  scan_io = FH.im2array(scan_io_file) #open flat field
  scan_dark = FH.im2array(scan_dark_file) #open dark field

  #shape of the projection images, used by gpu to launch parallel threads
  shape0 = np.shape(FH.im2array(datafolder + '\\' + datfiles[0]))[0]
  shape1 = np.shape(FH.im2array(datafolder + '\\' + datfiles[0]))[1]

  #Initiate the CUDA kernel, this defines the number of threads that will be launched
  threadsperblock = (16, 16)
  blockspergrid_x = math.ceil(shape0 / threadsperblock[0])
  blockspergrid_y = math.ceil(shape1 / threadsperblock[1])
  blockspergrid = (blockspergrid_x, blockspergrid_y)

  numfiles = len(datfiles)
  start = time.time()
  for i, datfile in enumerate(datfiles): #iterate over all projections that need to be corrected
    print('Correcting projection {} of {}'.format(i + 1, numfiles), end='\r')
    proj = FH.im2array(datafolder + '\\' + datfile) #open normalized projection that will be corrected
    proj = fun.NormaliseExperimental(proj, scan_io, scan_dark) #normalization
    scattering = CalcScatteringNormalizedProjection(proj, scatter_space_projections) #calculate scattering
    proj = proj - scattering #scattering correction
    proj_gpu = cuda.to_device(proj) #copy projection to GPU device memory
    CorrectNormalizedProjectionGPU[blockspergrid, threadsperblock](wrl_gpu, nwp_gpu, proj_gpu) #correct projection on the GPU
    corr = proj_gpu.copy_to_host() #copy corrected projection back to RAM
    FH.array2image2(savefolder + '\\' + datfile, corr) #save corrected projection
  # Add a newline after the loop to avoid overwriting the final output
  print()  

  # Print the elapsed time on its own line
  elapsed_time = time.time() - start
  elapsed_minutes = elapsed_time / 60
  print('Time elapsed: {:.2f} minutes'.format(elapsed_minutes))





# does previus function but copies folder to local drive: if your data is on a slow server.
def CorrectDatasetExperimentalGPUcopyToLocalDrive(datafolder, scanprefix, scan_io_file, scan_dark_file, savefolder,
                                                  wedge_poly_file, wedge_IO_file, wedge_dark_file, a, b, SOD, SDD,
                                                  detector_height, detector_width, LUT_height, pixelsize,
                                                  scatter_space_LUT, scatter_space_projections, local_temp_dir,
                                                  height_pitch=0.0):
    """
    Copies the dataset to a local directory, corrects it, and then copies it back.
    
    :param local_temp_dir: Path to the local directory where the dataset will be temporarily stored
    """

    start_total = time.time()

    # Create a temporary folder for processing
    local_datafolder = os.path.join(local_temp_dir, os.path.basename(datafolder))
    local_savefolder = os.path.join(local_temp_dir, os.path.basename(savefolder))
    
    # Ensure local temp directories exist
    os.makedirs(local_temp_dir, exist_ok=True)
    os.makedirs(local_savefolder, exist_ok=True)
    
    # Copy data to local drive
    print(f"Copying {datafolder} to {local_datafolder}...")
    shutil.copytree(datafolder, local_datafolder, dirs_exist_ok=True)
    
    # Update paths to local versions
    local_scan_io_file = os.path.join(local_datafolder, os.path.basename(scan_io_file))
    local_scan_dark_file = os.path.join(local_datafolder, os.path.basename(scan_dark_file))
    
    # Perform correction on the local copy
    CorrectDatasetExperimentalGPU(
        datafolder=local_datafolder,
        scanprefix=scanprefix,
        scan_io_file=local_scan_io_file,
        scan_dark_file=local_scan_dark_file,
        savefolder=local_savefolder,
        wedge_poly_file=wedge_poly_file,
        wedge_IO_file=wedge_IO_file,
        wedge_dark_file=wedge_dark_file,
        a=a,
        b=b,
        SOD=SOD,
        SDD=SDD,
        detector_height=detector_height,
        detector_width=detector_width,
        LUT_height=LUT_height,
        pixelsize=pixelsize,
        scatter_space_LUT=scatter_space_LUT,
        scatter_space_projections=scatter_space_projections,
        height_pitch=height_pitch
    )
    
    # Copy corrected data back to the original save folder
    print(f"Copying corrected data back to {savefolder}...")
    shutil.copytree(local_savefolder, savefolder, dirs_exist_ok=True)
    
    # Clean up local temporary files
    print(f"Cleaning up {local_datafolder} and {local_savefolder}...")
    shutil.rmtree(local_datafolder)
    shutil.rmtree(local_savefolder)
    
    print("Correction completed successfully!")
    elapsed_total = time.time() - start_total
    print(f"Total operation time: {elapsed_total / 60:.2f} minutes")








#simulated LUT
def CorrectNormalizedDatasetSimulated(datafolder, scanprefix, savefolder, a, b, SOD, SDD, detector_height, detector_width, LUT_height, pixelsize, tube, detector, voltage, material, density, scatter_space_projections,  filter=False, filtermaterial='Al', filterdensity=2.7, filterthickness=1.0, splits=1, height_pitch=0.0):
  '''
  Performs the correction with an simulated LUT on CPU on a full data set. Also prints time that was needed for the total correction at the end in seconds.

  :param datafolder: path to folder with normalized projection images that need to be corrected
  :param scanprefix: prefix in the names of the normalized projection images
  :param savefolder: path to folder where corrected images need to be saved
  :param a: maximal thickness of the wedge (thickness at the bottom) (in mm)
  :param b: height of the wedge (in mm)
  :param SOD: source to objecte distance (distance from source to flat vertical plane of wedge, wedge is placed in scanner such that vertical plane faces the source)
  :param SDD: source to detector distance (should be same in scan as in projection of wedge in order to have same heel effect!!!)
  :param detector_height: height of the detector in pixels as used for the projection images in the data set that is being corrected
  :param detector_width: width of the detector in pixels as used for the projection images in the data set that is being corrected
  :param LUT_height: eventual height of the LUT that will be used (for example 1010 pixels if projection of wedge only uses top 1000 pixels of detector)
  :param pixelsize: size of a pixel (in mm)
  :param tube: name of the used scanner (currently only 'Hector' is available
  :param detector: name of the used detector
  :param voltage: used tube voltage (will be used in the simulation of the projection of the wedge)
  :param material: material of the wedge (for example 'POM')
  :param density: mass density of the wedge (in g/cm^3)
  :param scatter_space_projections: [x_1, x_2, y_1, y_2], defines a region in the projection images that is used for calculating the scattering value
  :param filter: bool, is spectrum filtered or not
  :param filtermaterial: material of the used filter
  :param filterdensity: mass density of the filter material
  :param filterthickness: thickness of the filter (in mm)
  :param splits: The simulation of the projection of the wedge uses a lot of RAM, so the calculation is split (best use something like 10 splits)
  :param height_pitch: downwards shift of the wedge in mm (if 0 mm, then wedge is on top of optical axis)
  '''

  #prepare simulated LUT:
  #wrl = projected thickness of wedge
  #nwp = normalized and processed projection image of the wedge
  wrl, nwp = CreateLUTSimulation(a, b, SOD, SDD, detector_height, detector_width, LUT_height, pixelsize, tube, detector, voltage, material, density, filter, filtermaterial, filterdensity, filterthickness, height_pitch, splits=splits)

  # make list of files that need to be corrected
  datfiles = [f for f in os.listdir(datafolder) if re.match(scanprefix + r'*', f)]

  numfiles = len(datfiles)
  start = time.time()
  for i, datfile in enumerate(datfiles): #iterate over all files that need to be corrected
    print('Correcting projection {} of {}'.format(i + 1, numfiles), end='\r')
    proj = FH.im2array(datafolder + '\\' + datfile) #open normalized projection image
    scattering = CalcScatteringNormalizedProjection(proj, scatter_space_projections) #calculate scattering
    proj = proj - scattering #scattering correction
    corr = CorrectNormalizedProjection(wrl, nwp, proj) #correct projection on CPU
    FH.array2image2(savefolder + '\\' + datfile, corr) #save corrected projection image
  print(time.time() - start)

#simulated LUT
def CorrectNormalizedDatasetSimulatedGPU(datafolder, scanprefix, savefolder, a, b, SOD, SDD, detector_height, detector_width, LUT_height, pixelsize, tube, detector, voltage, material, density, scatter_space_projections,  filter=False, filtermaterial='Al', filterdensity=2.7, filterthickness=1.0, splits=1, height_pitch=0.0):
  '''
  Performs the correction with a simulated LUT on GPU on a full data set. Also prints time that was needed for the total correction at the end in seconds.

  :param datafolder: path to folder with normalized projection images that need to be corrected
  :param scanprefix: prefix in the names of the normalized projection images
  :param savefolder: path to folder where corrected images need to be saved
  :param a: maximal thickness of the wedge (thickness at the bottom) (in mm)
  :param b: height of the wedge (in mm)
  :param SOD: source to objecte distance (distance from source to flat vertical plane of wedge, wedge is placed in scanner such that vertical plane faces the source)
  :param SDD: source to detector distance (should be same in scan as in projection of wedge in order to have same heel effect!!!)
  :param detector_height: height of the detector in pixels as used for the projection images in the data set that is being corrected
  :param detector_width: width of the detector in pixels as used for the projection images in the data set that is being corrected
  :param LUT_height: eventual height of the LUT that will be used (for example 1010 pixels if projection of wedge only uses top 1000 pixels of detector)
  :param pixelsize: size of a pixel (in mm)
  :param tube: name of the used scanner (currently only 'Hector' is available
  :param detector: name of the used detector
  :param voltage: used tube voltage (will be used in the simulation of the projection of the wedge)
  :param material: material of the wedge (for example 'POM')
  :param density: mass density of the wedge (in g/cm^3)
  :param scatter_space_projections: [x_1, x_2, y_1, y_2], defines a region in the projection images that is used for calculating the scattering value
  :param filter: bool, is spectrum filtered or not
  :param filtermaterial: material of the used filter
  :param filterdensity: mass density of the filter material
  :param filterthickness: thickness of the filter (in mm)
  :param splits: The simulation of the projection of the wedge uses a lot of RAM, so the calculation is split (best use something like 10 splits)
  :param height_pitch: downwards shift of the wedge in mm (if 0 mm, then wedge is on top of optical axis)
  '''

  # prepare simulated LUT:
  # wrl = projected thickness of wedge
  # nwp = normalized and processed projection image of the wedge
  wrl, nwp = CreateLUTSimulation(a, b, SOD, SDD, detector_height, detector_width, LUT_height, pixelsize, tube, detector, voltage, material, density, filter, filtermaterial, filterdensity, filterthickness, height_pitch, splits=splits)

  # copy LUT to the GPU device momory:
  wrl_gpu = cuda.to_device(wrl)
  nwp_gpu = cuda.to_device(nwp)

  # make list of files that need to be corrected
  datfiles = [f for f in os.listdir(datafolder) if re.match(scanprefix + r'*', f)]

  # shape of the projection images, used by gpu to launch parallel threads
  shape0 = np.shape(FH.im2array(datafolder + '\\' + datfiles[0]))[0]
  shape1 = np.shape(FH.im2array(datafolder + '\\' + datfiles[0]))[1]

  # Initiate the CUDA kernel, this defines the number of threads that will be launched
  threadsperblock = (16, 16)
  blockspergrid_x = math.ceil(shape0 / threadsperblock[0])
  blockspergrid_y = math.ceil(shape1 / threadsperblock[1])
  blockspergrid = (blockspergrid_x, blockspergrid_y)

  numfiles = len(datfiles)
  start = time.time()
  for i, datfile in enumerate(datfiles): #iterate over all files that need to be corrected
    print('Correcting projection {} of {}'.format(i + 1, numfiles), end='\r')
    proj = FH.im2array(datafolder + '\\' + datfile) #open normalized projection image
    scattering = CalcScatteringNormalizedProjection(proj, scatter_space_projections) #calculate scattering
    proj = proj - scattering #scattering correction
    proj_gpu = cuda.to_device(proj) #copy projection to GPU device memory
    CorrectNormalizedProjectionGPU[blockspergrid, threadsperblock](wrl_gpu, nwp_gpu, proj_gpu) #correct projection on the GPU
    corr = proj_gpu.copy_to_host() #copy corrected projection back to RAM
    FH.array2image2(savefolder + '\\' + datfile, corr) #save corrected projection image
  print(time.time() - start)

# Example of correction of a full data set______________________________________________________________________________
# datafolder = r'F:\Users\labo\HeelEffect\Data\Helical 6_normalised'

# savefolder = r'F:\Users\labo\HeelEffect\Data\Helical 6_corrected_experimentalLUT_100kV0.5mmAl_28092022'

# wedge_poly_file = r'F:/Users/labo/HeelEffect/Code/HeelEffect/joined_wedge.tif'
# wedge_io_file = r'F:/Users/labo/HeelEffect/Code/HeelEffect/io_100kV_10W(110microA)_140avg.tif'
# wedge_dark_file = r'F:/Users/labo/HeelEffect/Code/HeelEffect/di_70avg.tif'


#CorrectNormalizedDatasetExperimentalGPU(datafolder, 'Bristlecone Helical', savefolder, wedge_poly_file, wedge_io_file,
#                                         wedge_dark_file, 40.6, 40.3, 229.8, 1050.0, 2000, 1700, 1010, 0.2,
#                                         scatter_space_LUT=50, scatter_space_projections=[10, 30, 200, 800], height_pitch=0)

#CorrectNormalizedDatasetSimulatedGPU(datafolder, 'Bristlecone Helical', savefolder, 40.6, 40.3, 229.8, 1050, 2000, 1700,
#                                 1010, 0.2, 'Hector', 'PerkinElmer', 90, 'POM', 1.415,
#                                scatter_space_projections=[10, 30, 200, 800], filter=True, filterthickness=0.05, splits=10, height_pitch=0.0)

# print(numba.threading_layer())