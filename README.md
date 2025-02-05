<p align="center">
  <img src="./Fig.JPG" width="100%" alt="logo">
</p>

<p align="center">
    <h1 align="center">Heel Correction</h1>
</p>



[![DOI]()]()

[De Bolle, Jorden![ORCID logo](https://info.orcid.org/wp-content/uploads/2019/11/orcid_16x16.png)](https://orcid.org/0000-0002-5179-1725)[^aut][^UG-RP][^UG-SSS];
[Verschuren, Louis![ORCID logo](https://info.orcid.org/wp-content/uploads/2019/11/orcid_16x16.png)](https://orcid.org/0000-0002-3102-4588)[^aut][^cre][^UG-WL];
[Van den Bulcke, Jan![ORCID logo](https://info.orcid.org/wp-content/uploads/2019/11/orcid_16x16.png)](https://orcid.org/0000-0003-2939-5408)[^aut][^UG-WL]
[Boone, Matthieu N.![ORCID logo](https://info.orcid.org/wp-content/uploads/2019/11/orcid_16x16.png)](https://orcid.org/0000-0002-5478-4141)[^aut][^UG-RP];

[^aut]: author
[^cre]: contact person
[^UG-WL]: UGent-Woodlab, Department of Environment, Ghent University, Coupure Links 653, Gent, 9000, Belgium
[^UG-RP]: RP-UGCT, Department of Physics and Astronomy – Radiation Physics, Ghent University, Proeftuinstraat 86, Gent, 9000, Belgium
[^UG-SSS]: Current affiliation: Department of Solid State Sciences - CoCooN research group, Ghent University, Krijgslaan 286, Gent, 9000, Belgium


<p align="left">
   This is the repository for a Python routine that corrects for the heel effect in (helical) X-ray micro-CT scans. It's primary use is increasing the accuracy with which the local mass densities in wood increment cores can be determined. The correction happens on the level of the normalised projection images of the scan. It requires a radiographic projection of a prism-shaped calibration sample that was obtained with the same tube settings and source-to-detector distance as the CT-scan. After correction and reconstruction, one obtains a 3D reconstructed volume that is to a large extent free of consequences from the heel effect and cupping ( which arises as a consequence of beam hardening). The volume consists of voxels that contain a gray-value that is proportional to the local mass density in the sample. By using the Numba library, the routine was optimised to run on an NVIDIA GPU for fast execution.
</p>


<p align="center">
	<!-- local repository, no metadata badges. --></p>
<p align="center">
		<em>Built with the tools and technologies:</em>
</p>
<p align="center">
	<img src="https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue" alt="Python">
	<img src="https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy">
	<img src="https://img.shields.io/badge/SciPy-654FF0?style=for-the-badge&logo=SciPy&logoColor=white" alt="SciPy">
	<img src="https://img.shields.io/badge/Numba-00A3E0?style=for-the-badge&logo=Numba&logoColor=white" alt="Numba">

	
</p>
<br>

#####  Table of Contents

- [ Introduction to the method](#introduction)
- [ Getting Started](#getting-started)
- [ Performing a heel correction](#performing-a-heel-correction)
- [ Cite our work](#cite-our-work)
- [ License](#license)

---

##  Introduction to the method

TODO

---

## Getting started

Before running the notebooks, ensure that you have the following dependencies installed:
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

TODO

---

## Performing a heel correction

TODO

---

## Cite our work

You can find the paper where the entire pipeline is described [here](TODO), or cite our work with the following bibtex snippet:

```tex
TODO
```


When using any of the software, also cite the proper Zenodo DOI [here](TODO)

---

##  License

This software is protected under the [GNU AGPLv3](https://choosealicense.com/licenses/agpl-3.0/) license. 

---
