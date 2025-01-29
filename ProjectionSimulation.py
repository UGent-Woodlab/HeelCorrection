'''
Written by Jorden De Bolle
Last update: 18/07/2022
'''
'''
!THIS CODE ONLY RUNS IF THE SOFTWARE PACKAGE CTREX IS AVAILABLE!

This script serves as an example on how to create monochromatic projection images or sinograms with CTRex.
These images must contain in every pixel the information exp(-d) with d the distance travelled through the simulated
sample by the X-ray that would hit that pixel. By taking the negative logarithm of these images, d can be obtained
for every pixel and this then represents the pseudo-images or pseudo-sinograms that are the starting point of the
new polychromatic simulation tool. See the master thesis of Jorden De Bolle for more conceptual details.
'''

from setups import ProjSetupper #from CTRex
from toolkit import filetools as ft, datatools as dt #from CTRex
import os
import numpy as np
from simulation.phantom3d import phantom3d #from CTRex
from geometry.volume import Volume #from CTRex
import time

def project_monochromatic(projection_path, sods = [200]):
    setup = ProjSetupper()
    setup.interactive = False
    setup.as_sinogram = False
    setup.input_path = ft.Prefixpad(os.path.join(projection_path, 'phantom'), 'phantom_')
    setup.detector.height = 512
    setup.detector_width = 512
    setup.detector.roi = [0,512,0,512] #to make a sinogram
    setup.outputtype = np.dtype(np.uint16)
    setup.proj_per_timestep = 512 #number of projections
    sdd = 1000
    for sod in sods:
        setup.output_path = ft.Prefixpad(os.path.join(projection_path, 'projection_sod%04d'%sod), 'proj_')
        setup.readParams(extraparams={'sod':sod, 'sdd':sdd, 'vertical centre':256, 'horizontal centre': 256,
                                      'rotation axis position':256})

        setup.execute()
    setup.terminate()

def make_phantom(path):
    nx, ny, nz = (256, 256, 256)
    # (A, x0, y0, z0, phi, theta, psi, a, b, c) for ellipse
    # (A, x0, y0, z0, phi, theta, h, R) for cylinder
    # (A, x0, y0, z0, phi, theta, psi, a, r, d, t) for helix
    vol = Volume(shape=(nz,ny,nx), chunks=1)
    vol.create()
    a = np.array([
                  # [0.2 , 64.00 , 128.0 , 128.0 , 30. , 0   , 0   , 25.60 , 128.0 , 51.20],
                  # [0.2 , 115.2 , 230.4 , 197.2 , 0   , 0   , 0   , 28.54 , 57.09 , 57.09],
                  # [0.2 , 115.2 , 230.4 , 128.0 , 0   , 0   , 0   , 28.16 , 56.32 , 56.32],
                  # [0.2 , 115.2 , 230.4 , 76.80 , 0   , 0   , 0   , 27.90 , 55.81 , 55.81],
                  [1.0 , 128.0 , 128.0 , 128.0 , 90   , 90   , 90   , 53 , 30 , 100]], #Only create an ellipsoid
                  dtype = np.float32)
    # b = np.array([[0.15, 76.80 , 76.80 , 0.000 , 0.0 , 0.0 , 100., 50.],
    #               [0.15, 76.80 , 76.80 , 100.0 , 60. , 15. , 100., 25.],
    #               [1.  , 76.80 , 76.80 , 0     , 0   , 0   , 128., 5.0]] ,dtype = np.float32)
    # c = np.array([[0.4 , 76.80 , 76.80 , 100.0 , 0   , 0   , 0   , 30.   , 40., 5., 3.  ]], dtype = np.float32)
    # phantom3d(vol, ellipses = a, cylinders = b, helices=c)
    phantom3d(vol, ellipses=a)
    vol.to_disk(path = path, prefix='phantom_')


start_time = time.time()

make_phantom('C:\\Users\\jdebolle\\Documents\\PythonProjects\\thesis_code\\Projections\\phantom3')
project_monochromatic('C:\\Users\\jdebolle\\Documents\\PythonProjects\\thesis_code\\Projections\\projection_phantom3_sod200')


print("--- %s seconds ---" % (time.time() - start_time))