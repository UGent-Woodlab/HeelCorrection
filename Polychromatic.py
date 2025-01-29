'''
Written by Jorden De Bolle
Last update: 18/07/2022
'''
'''
These functions are the core of the simulation tool. They transform pseudo-projections or pseudo-sinograms that contain
thickness information into polychromatic simulated projection images or sinograms. See the master thesis of Jorden De Bolle
for conceptual and mathematical details. See also the Phd thesis of Jelle Dhaene.
'''

import FileHandler as FH
import numpy as np
import Functions as fun

def InitialiseObject(material, tubeEnergies):
    '''
    Calculates the mass attenuation coefficient of a material at the energies at which the X-ray
    spectrum is sampled by logarithmic interpolation
    :param material: material of which object is made (string)
    :param tubeEnergies: numpy array of photon energies at which spectrum is sampled
    :return: mass attenuation coefficient of material at tubeEnergies (numpy array)
    '''
    attEnergies, atts = FH.loadAttenuations(material)
    newatts = fun.interpolateLog(attEnergies, atts, tubeEnergies)
    return newatts

def InitialiseScanner(tube, detector, voltage):
    '''
    Initializes information of the used scanner in the simulation for a scan without heel effect
    :param tube: used tube (string) (currently only 'Hector' is available)
    :param detector: used detector (string, for example 'PerkinElmer')
    :param voltage: used tube voltage (int) (make sure that the spectrum file of the chosen voltage exists)
    :return: Energies at which spectrum is sampled, 1D spectrum (no heeling), response of the detector interpolated at
    energies at which spectrum is sampled
    '''
    Energies, spec = FH.loadSpectrum(tube, voltage)
    detectorEnergies, response = FH.loadDetectorResponse(detector)
    newresponse = fun.interpolate(detectorEnergies, Energies, response)
    return Energies, spec, newresponse

def InitialiseScannerHeel(mono, tube, detector, voltage, pixelsize, SDD):
    '''
    Initializes information of the used scanner in the simulation for a scan with heel effect. Also calculates
    and returns the heel map (only the Hector scanner is currently available) based on the size and shape of the
    detector and the SDD (see master thesis of Jorden De Bolle for mathematical details)
    :param mono: monochromatic projection or sinogram or empty numpy array to define shape of the detector (numpy array)
    :param tube: name of the used scanner (string, currently only 'Hector' is available)
    :param detector: name of the used detector (string, for example 'PerkinElmer')
    :param voltage: used tube voltage (int) (make sure that the spectrum file of the chosen voltage exists)
    :param pixelsize: size of the pixels in the detector (float, 0.2 mm for the Perkin Elmer detector)
    :param SDD: source to detector distance in mm
    :return: Energies at which spectrum is sampled, 1D spectrum (no heeling), the heel map, response of the detector interpolated at
    energies at which spectrum is sampled
    '''
    Energies, spec = FH.loadSpectraHeel(tube, voltage)
    detectorEnergies, response = FH.loadDetectorResponse(detector)
    newresponse = fun.interpolate(detectorEnergies, Energies, response)
    heelmap = fun.createHeelmapHector(mono, pixelsize, SDD, np.shape(spec)[1])
    return Energies, spec, heelmap, newresponse

def FilterSpectrum(material, density, thickness, spectrum, energies): #thickness in mm
    '''
    Filters the 1D X-ray spectrum used in a simulation without heeling with a filter of certain material with certain
    density and thickness
    :param material: filter material (string, vb 'Al' for aluminum)
    :param density: mass density of filter in g/cm^3 (float)
    :param thickness: thickness of the filter in mm
    :param spectrum: the 1D spectrum that needs to be filtered (1D numpy array)
    :param energies: values for the photon energies at which the spectrum is filtered
    :return: filtered spectrum
    '''
    #important remark: this filtering assumes that all rays traverse the same thickness of filter material
    #in practice this is not true, but mostly the error is negligible
    thickness = thickness * 0.1 #thickness in cm
    att = InitialiseObject(material, energies) #load mass attenuation coefficients and interpolate
    att = np.multiply(att, density) #multiply with density for atteuation coefficient
    att = np.multiply(att, thickness) #multiply with thickness for law of Lambert Beer
    att = np.exp(-1.0*att) #take exponential for law of Lambert Beer
    filtered = np.multiply(spectrum, att) #multiply with spectrum to get new spectrum
    return filtered

def FilterSpectrumHeel(material, density, thickness, spectrum, energies): #thickness in mm
    '''
    Filters the 2D X-ray spectrum used in a simulation with heeling with a filter of certain material with certain
    density and thickness
    :param material: filter material (string, vb 'Al' for aluminum)
    :param density: mass density of filter in g/cm^3 (float)
    :param thickness: thickness of the filter in mm
    :param spectrum: the 2D spectrum that needs to be filtered (2D numpy array)
    :param energies: values for the photon energies at which the spectrum is sampled
    :return: filtered 2D spectrum
    '''
    #important remark: this filtering assumes that all rays traverse the same thickness of filter material
    #in practice this is not true
    thickness = thickness * 0.1  # thickness in cm
    att = InitialiseObject(material, energies)
    att = np.multiply(att, density)
    att = np.multiply(att, thickness)
    att = np.exp(-1.0 * att)
    filtered = np.einsum('ij, i -> ij', spectrum, att) #multiply every column in the 2D spectrum with exponential factor
    return filtered

def singleObjectAttenuations(atts, density, mono, exp = True): #Lookuptable immediately gives d, not exp(-d)
    '''
    Prepare a 3D numpy array that contains the attenuation factor (exp(-\mu*d)) for every pixel of a pseudo-projection
    or pseudo-sinogram of a single object and for every energy bin at which the used X-ray spectrum is sampled.
    :param atts: numpy array of mass attenuation coefficients of material of which object is made
    :param density: mass density of the object in g/cm^3 (flaot)
    :param mono: monochromatic image or sinogram that contains exp(-d) (if exp=True) or pseudo-projection or
                pseudo-sinogram that contains d (if exp=False)
    :param exp: see above
    :return: 3D numpy array that contains the attenuation factor for every pixel of a pseudo-projection
    or pseudo-sinogram of a single object and for every energy bin.
    '''
    #get array with exp(-mu_i*d) so that different objects can be combined later
    #returns some form of sinogram, different sinograms from different objects can be multiplied
    if exp == True:
        d = fun.getProjectedRaylength(mono) #d in cm
    else:
        d = mono #d in cm
    singleobjectatt = np.einsum('ij, k->ijk', d, atts*density)
    singleobjectatt = np.exp(-1.0*singleobjectatt) #, dtype='float16')
    return singleobjectatt

def combineObjects(monos, materials, densities, tubeEnergies):
    '''
    Colculates the 3D array that contains the total attenuation factor due to several objects by combining
    singleobjectattenuations from the previous function for different objects
    :param monos: python list of monochromatic projections or sinograms that contains exp(-d) in every pixel (list of numpy arrays)
    :param materials: python list of materials of the separate objects defined by monos (list of strings)
    :param densities: python list of mass densities in g/cm^3 (list of floats)
    :param tubeEnergies: values for the photon energies at which the spectrum is sampled (numpy array)
    :return: 3D numpy array that contains the total attenuation factor due to several objects for every energy bin
    '''
    atts = InitialiseObject(materials[0], tubeEnergies)
    multi = singleObjectAttenuations(atts, densities[0], monos[0])
    l = len(monos)
    for i in range(1, l):
        atts = InitialiseObject(materials[i], tubeEnergies)
        single = singleObjectAttenuations(atts, densities[i], monos[i])
        multi = multi*single
    return multi

def calcSolidAnglesSino(sino, pixelsize, SDD): #calculate outside of other functions to avoid recalculation
    '''
    Calculates the solid angles of the pixels in a sinogram
    :param sino: the (monochromatic) sinogram (numpy array) for which solid angles must be calculated (defines shape)
    :param pixelsize: size of the pixels in the sinogram
    :param SDD: source to detector distance in mm
    :return: numpy array of same shape as sinogram that contains the solid angle of each pixel
    '''
    solidsangles = np.zeros(np.shape(sino))
    detectorwidth = np.shape(solidsangles)[1]
    rows, columns = np.shape(sino)
    for i in range(rows):
        for j in range(columns):
            solidsangles[i, j] = fun.calcSolidAngle(SDD, pixelsize, 0, j - detectorwidth/2)
    return solidsangles

def calcSolidAnglesProjection(proj, pixelsize, SDD): #calculate outside of other functions to avoid recalculation
    '''
    Calculates the solid angles of the pixels in a projection image
    :param proj: the (monochromatic) projection (numpy array) for which solid angles must be calculated (defines shape)
    :param pixelsize: size of the pixels in the projection
    :param SDD: source to detector distance in mm
    :return: numpy array of same shape as projection that contains the solid angle of each pixel
    '''
    solidsangles = np.zeros(np.shape(proj))
    detectorwidth = np.shape(solidsangles)[1]
    detectorheight = np.shape(solidsangles)[0]
    rows, columns = np.shape(proj)
    for i in range(rows):
        for j in range(columns):
            solidsangles[i, j] = fun.calcSolidAngle(SDD, pixelsize, i - detectorheight/2, j - detectorwidth/2)
    return solidsangles


# works for projections too
def createIOSino(sino, energies, spectrum, response, solidangles, P, exptime, U, gain):
    '''
    Create flat field sinogram or projection image, without heel effect. See master thesis of Jorden De Bolle
    for mathematical details.
    :param sino: monochromatic sinogram or projection or pseudo-sinogram or projection to define shape
    :param energies: photon energies at which spectrum is sampled
    :param spectrum: 1D X-ray spectrum
    :param response: detector response (interpolated to photon energies energies of spectrum)
    :param solidangles: solid angles of the projection or sinogram
    :param P: tube power (watt)
    :param exptime: exposure time (seconds)
    :param U: used tube voltage (kV, same voltage as used to select spectrum)
    :param gain: gain factor (multiplies outcome, usually something like 0.001)
    :return: flat field sinogram or flat field projection image (2D numpy array) without heel effect
    '''
    IO = np.ones(np.shape(sino))
    charge = 1.60217662 * 10**(-19) #elementary charge
    binwidths = fun.getBinWidths(energies)
    prefactor = ((P * exptime) / (U * charge)) * gain
    SRE = np.einsum('i, i -> i', spectrum, response)
    SRE = np.einsum('i, i -> i', SRE[1:], binwidths)  #start from index 1 because binwidths is one element shorter
    SRE = np.einsum('i->', SRE)
    IO = IO * SRE * prefactor
    IO = np.einsum('ij, ij->ij', IO, solidangles)
    return IO

# works for projections too
def createIOHeelSino(sino, energies, heelspectrum, heelmap, response, solidangles, P, exptime, U, gain):
    '''
    Create flat field sinogram or projection image, with heel effect. See master thesis of Jorden De Bolle
    for mathematical details.
    :param sino: monochromatic sinogram or projection or pseudo-sinogram or projection to define shape
    :param energies: photon energies at which spectrum is sampled
    :param heelspectrum: 2D X-ray spectrum
    :param heelmap: heel map that defines which column of the 2D spectrum hits which pixel
    :param response: detector response (interpolated to photon energies energies of spectrum)
    :param solidangles: solid angles of the projection or sinogram
    :param P: tube power (watt)
    :param exptime: exposure time (seconds)
    :param U: tube voltage (kV, same voltage as used to select spectrum)
    :param gain: gain factor (multiplies outcome, usually something linke 0.001)
    :return: flat field sinogram or flat field projection image (2D numpy array) with heel effect
    '''
    IO = np.zeros(np.shape(sino))
    binwidths = fun.getBinWidths(energies)
    rows, columns = np.shape(sino)
    charge = 1.60217662 * 10 ** (-19)
    prefactor = ((P*exptime)/(U*charge)) * gain
    SRE = np.einsum('ij, i -> ij', heelspectrum, response)
    SRE = np.einsum('ij, i -> ij', SRE[1:, :], binwidths)  #start from index 1 because binwidths is one element shorter
    SRE = np.einsum('ij -> j', SRE)
    SRE = SRE*prefactor
    for i in range(rows):
        for j in range(columns):
            IO[i, j] = SRE[int(heelmap[i, j])-1] #use heel map to select for every pixel what spectrum column is used
            #heelmap counts from 1-->500, columns numbered from 0-->499 so 1 is subtracted
    IO = np.einsum('ij, ij->ij', IO, solidangles) #take solid angles into account, otherwise every row in IO would be the same
    return IO

# works for projections too
def makePolySino(multi, energies, spectrum, response, solidangles, P, exptime, U, gain):
    '''
    Calculates simulated sinogram or projection image of an object or series of combined objects, without heel effect.
    See master thesis of Jorden De Bolle for mathematical details.
    :param multi: 3D numpy array returned by function combineObjects
    :param energies: photon energies at which spectrum is sampled
    :param spectrum: 1D X-ray spectrum
    :param response: detector response (interpolated to photon energies energies of spectrum)
    :param solidangles: solid angles of the projection or sinogram
    :param P: tube power (watt)
    :param exptime: exposure time (seconds)
    :param U: tube voltage (kV, same voltage as used to select spectrum)
    :param gain: gain factor (multiplies outcome, usually something linke 0.001)
    :return: simulated sinogram or projection image without heel effect (2D numpy array)
    '''
    polysino = np.ones(np.shape(multi[:, :, 0]))
    binwidths = fun.getBinWidths(energies)
    charge = 1.60217662 * 10 ** (-19)
    prefactor = ((P * exptime) / (U * charge)) * gain
    SRE = np.einsum('k, k -> k', spectrum, response)
    SRE = np.einsum('k, k -> k', SRE[1:], binwidths)  #start from index 1 because binwidths is one element shorter
    SRE = np.einsum('ijk, k -> ij', multi[:, :, 1:], SRE)
    polysino = polysino * SRE * prefactor
    polysino = np.einsum('ij, ij->ij', polysino, solidangles)
    return polysino


# works for projections too
def makePolyHeelSino(multi, energies, heelspectrum, heelmap, response, solidangles, P, exptime, U, gain):
    '''
    Calculates simulated sinogram or projection image of an object or series of combined objects, with heel effect.
    See master thesis of Jorden De Bolle for mathematical details.
    :param multi: 3D numpy array returned by function combineObjects
    :param energies: photon energies at which spectrum is sampled
    :param heelspectrum: 2D X-ray spectrum
    :param heelmap: heel map that defines which column of the 2D spectrum hits which pixel
    :param response: detector response (interpolated to photon energies energies of spectrum)
    :param solidangles: solid angles of the projection or sinogram
    :param P: tube power (watt)
    :param exptime: exposure time (seconds)
    :param U: tube voltage (kV, same voltage as used to select spectrum)
    :param gain: gain factor (multiplies outcome, usually something linke 0.001)
    :return: simulated sinogram or projection image with heel effect (2D numpy array)
    '''
    polysino = np.zeros(np.shape(multi[:, :, 0]))
    binwidths = fun.getBinWidths(energies)
    rows, columns = np.shape(polysino)
    charge = 1.60217662 * 10 ** (-19)
    prefactor = ((P*exptime)/(U*charge)) * gain
    SRE = np.einsum('ij, i -> ij', heelspectrum, response)
    SRE = np.einsum('ij, i -> ij', SRE[1:, :], binwidths)  #start from index 1 because binwidths is one element shorter
    for i in range(rows):
        for j in range(columns):
            polysino[i, j] = np.einsum('i, i ->', multi[i, j, 1:], SRE[:, int(heelmap[i, j])-1]) #multiply with correct spectrum as defined by heel map and sum over energy bins
            # heelmap counts from 1-->500, columns numbered from 0-->499 so 1 is subtracted
    polysino = np.einsum('ij, ij->ij', polysino, solidangles)
    polysino = polysino * prefactor
    return polysino






