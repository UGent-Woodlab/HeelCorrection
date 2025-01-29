'''
Written by Jorden De Bolle
Last update: 18/07/2022
'''
import Polychromatic as poly
import FileHandler as FH
import Functions as fun
import matplotlib.pyplot as plt
import matplotlib
font = {'size' : 14}
matplotlib.rc('font', **font)

# settings for tube, detector and geometry of the scan
tube = 'Hector'
U = 100 #kV (Tube voltage)
P = 10.0 #Watt (Tube power
detector = 'Varian'
gain = 0.001
exptime = 0.100 #s (exposure time)
SOD = 100 #mm
SDD = 200 #mm
detectorheight = 512 #pixels
detectorwidth = 512 #pixels
pixelsize = 0.1 #mm

#Load in monochromatic sinogram for ellipse with attenuation coefficient=1, created with CTRex
monosino = FH.im2array(r'C:\Users\jdebolle\Documents\PythonProjects\HeelingCode_final\proj_000000.tif')
#Load second sinogram of different ellipse
monosino2 = FH.im2array(r'C:\Users\jdebolle\Documents\PythonProjects\HeelingCode_final\proj2_000000.tif')

#get pseudo sinogram of first sinogram that contains thickness information
monosinod = fun.getProjectedRaylength(monosino)
#get pseudo sinogram of second sinogram that contains thickness information
monosinod2 = fun.getProjectedRaylength(monosino2)

#show pseudo sinograms
plt.imshow(monosinod, cmap='gray')
plt.colorbar(label='Path length (cm)')
plt.xlabel('Pixels')
plt.ylabel('Pixels')
plt.show()
plt.imshow(monosinod2, cmap='gray')
plt.colorbar(label='Path length (cm)')
plt.xlabel('Pixels')
plt.ylabel('Pixels')
plt.show()

#Calculate solid angles of pixels in sinogram (different for sinograms and projection images!!!) based on geometry
solidangles = poly.calcSolidAnglesSino(monosino2, pixelsize, SDD)

#Load in photon energies used in spectrum, used spectrum, detector response and calculate heelmap based on geometry
# for simulation with heeling
Energies, spec, heelmap, response = poly.InitialiseScannerHeel(monosino2, tube, detector, U, pixelsize, SDD)

#Load in photon energies used in spectrum, used spectrum and detector response for simulation without heeling
Energies2, spec2, response2 = poly.InitialiseScanner(tube, detector, U)

#show heelmap
plt.imshow(heelmap)
plt.colorbar(label='Spectrum index (= beam angle in mrad)')
plt.xlabel('Pixels')
plt.ylabel('Pixels')
plt.show()

#First ellipse is made of aluminum, second of carbon
materials = ['Al', 'C']
densities = [2.70, 3.52] #densities of aluminum and carbon (in g/cm^3)
monos = [monosino, monosino2] #Make list of used monochromatic simulations from CTRex

#show example of monochromatic sinogram
plt.imshow(monosino)
plt.show()

#combine two ellipsed, each made of a different material
multi = poly.combineObjects(monos, materials, densities, Energies2)

#Simulate polychromatic sinograms where both ellipses are contained within the same volume, each made of a different material
#With heeling
proj = poly.makePolyHeelSino(multi, Energies, spec, heelmap, response, solidangles, P, exptime, U, gain)
#Withouth heeling
proj2 = poly.makePolySino(multi, Energies2, spec2, response2, solidangles, P, exptime, U, gain)

#Simulate the corresponding flat field sinograms
#With heeling
IO = poly.createIOHeelSino(monosino2, Energies, spec, heelmap, response, solidangles, P, exptime, U, gain)
#Without heeling
IO2 = poly.createIOSino(monosino2, Energies2, spec2, response2, solidangles, P, exptime, U, gain)

#Show the results
plt.imshow(proj, cmap='gray')
plt.xlabel('Pixels')
plt.ylabel('Pixels')
plt.axhline(256, color='r')
plt.colorbar(label='Counts')
plt.show()

plt.imshow(proj2, cmap='gray')
plt.xlabel('Pixels')
plt.ylabel('Pixels')
plt.axhline(256, color='r')
plt.colorbar(label='Counts')
plt.show()

plt.imshow(IO, cmap='gray')
plt.xlabel('Pixels')
plt.ylabel('Pixels')
plt.colorbar(label='Counts')
plt.show()

plt.imshow(IO2, cmap='gray')
plt.xlabel('Pixels')
plt.ylabel('Pixels')
plt.colorbar(label='Counts')
plt.show()
