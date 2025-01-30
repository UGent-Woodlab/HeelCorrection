Code developed by Jorden De Bolle in 2021-2022
See the master thesis of Jorden De Bolle for conceptual details.
jorden.debolle@ugent.be
jordendebolle@gmail.com

IMPORTANT: folder with necessary data (spectra, attenuation files, detector response...) not available on Github!

Packages that need to be installed to run this code:
	- numpy
	- numba
	- matplotlib
	- scipy
	- pylibtiff
	- os
	- re
	- PIL
	- seaborn
	- time
	- math
	- cudatoolkit
	- pytest-shutil

Files needed for the simulator:
	- Folder Data
	- FileHandler.py
	- Functions.py
	- Polychromatic.py
	- ProjectionSimulation.py (the software package CTRex (developed at UGCT) is needed as well)
	- ExampleSimulation.py ()

Files needed for the Lookup Table:
	- LUT.py
	- Filehandler.py
	- Functions.py
	- Polychromatic.py

Other interesting files:
	- Analysis.py (analyse results of LUT)
	- SmoothHeelspectrum.py (smoothing of the noisy spectra that originate from Monte Carlo simulations)
