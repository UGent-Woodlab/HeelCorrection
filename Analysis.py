'''
Written by Jorden De Bolle
Last update: 18/07/2022
'''
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import matplotlib
import FileHandler as FH
from scipy import signal

def CSR(image, binary):     #contrast to signal ratio
    dark_pixels = image[(1-binary).astype(bool)]
    light_pixels = image[binary.astype(bool)]
    mean = np.mean(image)
    diff = np.abs(np.mean(light_pixels) - np.mean(dark_pixels))
    # sum = np.mean(light_pixels) + np.mean(dark_pixels)
    return diff/mean

def CNR(image, binary): #contrast to noise ratio
    dark_pixels = image[(1-binary).astype(bool)]
    light_pixels = image[binary.astype(bool)]
    std = np.sqrt((np.std(light_pixels))**2 + (np.std(dark_pixels))**2)
    diff = np.abs(np.mean(light_pixels) - np.mean(dark_pixels))
    return diff/std

def SNR(image, binary):
    light_pixels = image[binary.astype(bool)]
    std = np.std(light_pixels)
    mean = np.mean(light_pixels)
    return mean/std

def GetTrend(image, column, windowlength = 1001, polyorder = 3):
    column = image[:, column]
    trend = signal.savgol_filter(column, window_length=windowlength, polyorder=polyorder)
    return trend

def GetTrend2(data, windowlength=51, polyorder = 3):
    trend = signal.savgol_filter(data, window_length=windowlength, polyorder=polyorder)
    return trend

# notcorrected = FH.im2array2(r'F:\Helical6\Reslices\Reslice of Helical.tif', slope = 1.67849E-05, offset = -1.00000E-01)[325:]
# corrected000_90kV = FH.im2array2(r'F:\Helical6\Reslices\90kV\0.00mm.tif', slope = 4.11994E-05, offset = -7.00000E-01)[326:]
# corrected005_90kV = FH.im2array2(r'F:\Helical6\Reslices\90kV\0.05mm.tif', slope = 4.11994E-05, offset = -7.00000E-01)[326:]
# corrected010_90kV = FH.im2array2(r'F:\Helical6\Reslices\90kV\0.10mm.tif', slope = 4.11994E-05, offset = -7.00000E-01)[326:]
# corrected015_90kV = FH.im2array2(r'F:\Helical6\Reslices\90kV\0.15mm.tif', slope = 4.11994E-05, offset = -7.00000E-01)[326:]
# corrected020_90kV = FH.im2array2(r'F:\Helical6\Reslices\90kV\0.20mm.tif', slope = 4.11994E-05, offset = -7.00000E-01)[326:]
# corrected025_90kV = FH.im2array2(r'F:\Helical6\Reslices\90kV\0.25mm.tif', slope = 4.11994E-05, offset = -7.00000E-01)[326:]
# corrected030_90kV = FH.im2array2(r'F:\Helical6\Reslices\90kV\0.30mm.tif', slope = 4.11994E-05, offset = -7.00000E-01)[326:]
# corrected035_90kV = FH.im2array2(r'F:\Helical6\Reslices\90kV\0.35mm.tif', slope = 4.11994E-05, offset = -7.00000E-01)[326:]
# corrected040_90kV = FH.im2array2(r'F:\Helical6\Reslices\90kV\0.40mm.tif', slope = 4.11994E-05, offset = -7.00000E-01)[326:]
# corrected045_90kV = FH.im2array2(r'F:\Helical6\Reslices\90kV\0.45mm.tif', slope = 4.11994E-05, offset = -7.00000E-01)[326:]
# corrected050_90kV = FH.im2array2(r'F:\Helical6\Reslices\90kV\0.50mm.tif', slope = 4.11994E-05, offset = -7.00000E-01)[326:]
# corrected055_90kV = FH.im2array2(r'F:\Helical6\Reslices\90kV\0.55mm.tif', slope = 4.11994E-05, offset = -7.00000E-01)[326:]
# corrected060_90kV = FH.im2array2(r'F:\Helical6\Reslices\90kV\0.60mm.tif', slope = 4.11994E-05, offset = -7.00000E-01)[326:]
# corrected065_90kV = FH.im2array2(r'F:\Helical6\Reslices\90kV\0.65mm.tif', slope = 4.11994E-05, offset = -7.00000E-01)[326:]
# corrected070_90kV = FH.im2array2(r'F:\Helical6\Reslices\90kV\0.70mm.tif', slope = 4.11994E-05, offset = -7.00000E-01)[326:]
#
# corrected000_100kV = FH.im2array2(r'F:\Helical6\Reslices\100kV\0.00mm.tif', slope = 4.11994E-05, offset = -7.00000E-01)[326:]
# corrected005_100kV = FH.im2array2(r'F:\Helical6\Reslices\100kV\0.05mm.tif', slope = 4.11994E-05, offset = -7.00000E-01)[326:]
# corrected010_100kV = FH.im2array2(r'F:\Helical6\Reslices\100kV\0.10mm.tif', slope = 4.11994E-05, offset = -7.00000E-01)[326:]
# corrected015_100kV = FH.im2array2(r'F:\Helical6\Reslices\100kV\0.15mm.tif', slope = 4.11994E-05, offset = -7.00000E-01)[326:]
# corrected020_100kV = FH.im2array2(r'F:\Helical6\Reslices\100kV\0.20mm.tif', slope = 4.11994E-05, offset = -7.00000E-01)[326:]
# corrected025_100kV = FH.im2array2(r'F:\Helical6\Reslices\100kV\0.25mm.tif', slope = 4.11994E-05, offset = -7.00000E-01)[326:]
# corrected030_100kV = FH.im2array2(r'F:\Helical6\Reslices\100kV\0.30mm.tif', slope = 4.11994E-05, offset = -7.00000E-01)[326:]
# corrected035_100kV = FH.im2array2(r'F:\Helical6\Reslices\100kV\0.35mm.tif', slope = 4.11994E-05, offset = -7.00000E-01)[326:]
# corrected040_100kV = FH.im2array2(r'F:\Helical6\Reslices\100kV\0.40mm.tif', slope = 4.11994E-05, offset = -7.00000E-01)[326:]
# corrected045_100kV = FH.im2array2(r'F:\Helical6\Reslices\100kV\0.45mm.tif', slope = 4.11994E-05, offset = -7.00000E-01)[326:]
# corrected050_100kV = FH.im2array2(r'F:\Helical6\Reslices\100kV\0.50mm.tif', slope = 4.11994E-05, offset = -7.00000E-01)[326:]
# corrected055_100kV = FH.im2array2(r'F:\Helical6\Reslices\100kV\0.55mm.tif', slope = 4.11994E-05, offset = -7.00000E-01)[326:]
# corrected060_100kV = FH.im2array2(r'F:\Helical6\Reslices\100kV\0.60mm.tif', slope = 4.11994E-05, offset = -7.00000E-01)[326:]
# corrected065_100kV = FH.im2array2(r'F:\Helical6\Reslices\100kV\0.65mm.tif', slope = 4.11994E-05, offset = -7.00000E-01)[326:]
# corrected070_100kV = FH.im2array2(r'F:\Helical6\Reslices\100kV\0.70mm.tif', slope = 4.11994E-05, offset = -7.00000E-01)[326:]
#
# binary = FH.im2array2(r'F:\Helical6\Reslices\Binary.tif', slope=1/255, offset=0)[325:]
# # plt.imshow(binary)
# # plt.show()
#
# lijst_90kV = [corrected000_90kV, corrected005_90kV, corrected010_90kV, corrected015_90kV, corrected020_90kV, corrected025_90kV, corrected030_90kV, corrected035_90kV, corrected040_90kV, corrected045_90kV, corrected050_90kV, corrected055_90kV,
#          corrected060_90kV, corrected065_90kV, corrected070_90kV]
#
# lijst_100kV = [corrected000_100kV, corrected005_100kV, corrected010_100kV, corrected015_100kV, corrected020_100kV, corrected025_100kV, corrected030_100kV, corrected035_100kV, corrected040_100kV, corrected045_100kV, corrected050_100kV, corrected055_100kV,
#          corrected060_100kV, corrected065_100kV, corrected070_100kV]
#
# CNR_notcorrected = CNR(notcorrected, binary)
# CSR_notcorrected = CSR(notcorrected, binary)
# SNR_notcorrected = SNR(notcorrected, binary)
#
# xticks = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
# xticklabels = ['0.00 mm', '0.05 mm', '0.10 mm', '0.15 mm', '0.20 mm', '0.25 mm', '0.30 mm', '0.35 mm', '0.40 mm', '0.45 mm', '0.50 mm', '0.55 mm',
#                '0.60 mm', '0.65 mm', '0.70 mm',]
#
# CSRs_90kV = []
# CNRs_90kV = []
# SNRs_90kV = []
# for i in range(len(lijst_90kV)):
#     reldiff = CSR(lijst_90kV[i], binary)
#     cnr = CNR(lijst_90kV[i], binary)
#     snr = SNR(lijst_90kV[i], binary)
#     CSRs_90kV.append(reldiff)
#     CNRs_90kV.append(cnr)
#     SNRs_90kV.append(snr)
#
# CSRs_100kV = []
# CNRs_100kV = []
# SNRs_100kV = []
# for i in range(len(lijst_100kV)):
#     reldiff = CSR(lijst_100kV[i], binary)
#     cnr = CNR(lijst_100kV[i], binary)
#     snr = SNR(lijst_100kV[i], binary)
#     CSRs_100kV.append(reldiff)
#     CNRs_100kV.append(cnr)
#     SNRs_100kV.append(snr)
#
# seaborn.set_style('whitegrid')
# fontsize = 12
#
# plt.plot(CSRs_90kV, color='slateblue', marker='o',  label='Corrected with a 90 kV spectrum')
# plt.plot(CSRs_100kV, color='crimson', marker='^',  label='Corrected with a 100 kV spectrum')
# plt.axhline(CSR_notcorrected, color='olive', linestyle='--', label='Not corrected')
# plt.title('Contrast to signal ratio (CSR) as function of \n the thickness of the aluminum filter', fontsize=fontsize)
# plt.xticks(xticks, xticklabels, rotation=90, fontsize=fontsize)
# plt.yticks([0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06], fontsize=fontsize)
# plt.xlabel('Thickness of Al filter', fontsize=fontsize)
# plt.ylabel(r'CSR', fontsize=fontsize)
# plt.tight_layout()
# plt.legend()
# plt.show()
#
# plt.plot(CNRs_90kV, color='slateblue', marker='o', label='Corrected with a 90 kV spectrum')
# plt.plot(CNRs_100kV, color='crimson', marker='^', label='Corrected with a 100 kV spectrum')
# plt.axhline(CNR_notcorrected, color='olive', linestyle='--', label='Not corrected')
# plt.title('Contrast to noise ratio (CNR) as function of \n the thickness of the aluminum filter', fontsize=fontsize)
# plt.xticks(xticks, xticklabels, rotation = 90, fontsize=fontsize)
# plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8], fontsize=fontsize)
# plt.xlabel('Thickness of Al filter', fontsize=fontsize)
# plt.ylabel(r'CNR', fontsize=fontsize)
# plt.tight_layout()
# plt.legend()
# plt.show()
#
# plt.plot(SNRs_90kV, color='slateblue', marker='o', label='Corrected with a 90 kV spectrum')
# plt.plot(SNRs_100kV, color='crimson', marker='^', label='Corrected with a 100 kV spectrum')
# plt.axhline(SNR_notcorrected, color='olive', linestyle='--', label='Not corrected')
# plt.title('Signal to noise ratio (SNR) as function of \n the thickness of the aluminum filter', fontsize=fontsize)
# plt.xticks(xticks, xticklabels, rotation = 90, fontsize=fontsize)
# plt.yticks([10.0, 12.0, 14.0, 16.0, 18.0, 20.0], fontsize=fontsize)
# plt.xlabel('Thickness of Al filter', fontsize=fontsize)
# plt.ylabel(r'SNR', fontsize=fontsize)
# plt.tight_layout()
# plt.legend()
# plt.show()
#
# plt.plot(np.array(CNRs_90kV)/np.array(SNRs_90kV), color='slateblue', marker='o', label='Corrected with a 90 kV spectrum')
# plt.plot(np.array(CNRs_100kV)/np.array(SNRs_100kV), color='crimson', marker='^', label='Corrected with a 100 kV spectrum')
# plt.axhline(CNR_notcorrected/SNR_notcorrected, color='olive', linestyle='--', label='Not corrected')
# plt.title('Ratio of CNR and SNR as function of \n the thickness of the aluminum filter', fontsize=fontsize)
# plt.xticks(xticks, xticklabels, rotation = 90, fontsize=fontsize)
# plt.yticks([0.000, 0.01, 0.02, 0.03, 0.04], fontsize=fontsize)
# plt.xlabel('Thickness of Al filter', fontsize=fontsize)
# plt.ylabel(r'$\frac{CNR}{SNR}$', fontsize=fontsize)
# plt.tight_layout()
# plt.legend()
# plt.show()


# capping
# distances_90_010, vals_90_010 = np.loadtxt(r'F:\Helical6\Reslices\capping_90kV_0,10_slice6000.csv', skiprows=1, unpack=True, delimiter=',')
# vals_90_010 = vals_90_010*4.11994E-05 - 7.00000E-01
#
# distances_90_030, vals_90_030 = np.loadtxt(r'F:\Helical6\Reslices\capping_90kV_0,30_slice6000.csv', skiprows=1, unpack=True, delimiter=',')
# vals_90_030 = vals_90_030*4.11994E-05 - 7.00000E-01
#
# distances_90_050, vals_90_050 = np.loadtxt(r'F:\Helical6\Reslices\capping_90kV_0,50_slice6000.csv', skiprows=1, unpack=True, delimiter=',')
# vals_90_050 = vals_90_050*4.11994E-05 - 7.00000E-01
#
# distances_100_010, vals_100_010 = np.loadtxt(r'F:\Helical6\Reslices\capping_100kV_0,10_slice6000.csv', skiprows=1, unpack=True, delimiter=',')
# vals_100_010 = vals_100_010*4.11994E-05 - 7.00000E-01
#
# distances_100_030, vals_100_030 = np.loadtxt(r'F:\Helical6\Reslices\capping_100kV_0,30_slice6000.csv', skiprows=1, unpack=True, delimiter=',')
# vals_100_030 = vals_100_030*4.11994E-05 - 7.00000E-01
#
# distances_100_050, vals_100_050 = np.loadtxt(r'F:\Helical6\Reslices\capping_100kV_0,50_slice6000.csv', skiprows=1, unpack=True, delimiter=',')
# vals_100_050 = vals_100_050*4.11994E-05 - 7.00000E-01
#
#
#
# min = -0.3
# max = 1.5
font = {'size' : 14}
matplotlib.rc('font', **font)
#
# plt.figure(figsize=[6, 6])
# plt.plot(distances_90_010, vals_90_010, linewidth=0.5, color='green', label='90 kV, 0.10 mm Al')
# plt.legend(loc=3)
# plt.ylim([min, max])
# plt.xlabel('Distance along line (cm)')
# plt.ylabel('Voxel Value')
# plt.tight_layout()
#
# plt.figure(figsize=[6, 6])
# plt.plot(distances_90_030, vals_90_030, linewidth=0.5, color='green', label='90 kV, 0.30 mm Al')
# plt.legend(loc=3)
# plt.ylim([min, max])
# plt.xlabel('Distance along line (cm)')
# plt.ylabel('Voxel Value')
# plt.tight_layout()
#
# plt.figure(figsize=[6, 6])
# plt.plot(distances_90_050, vals_90_050, linewidth=0.5, color='green', label='90 kV, 0.50 mm Al')
# plt.legend(loc=3)
# plt.ylim([min, max])
# plt.xlabel('Distance along line (cm)')
# plt.ylabel('Voxel Value')
# plt.tight_layout()
#
# plt.figure(figsize=[6, 6])
# plt.plot(distances_100_010, vals_100_010, linewidth=0.5, color='green', label='100 kV, 0.10 mm Al')
# plt.legend(loc=3)
# plt.ylim([min, max])
# plt.xlabel('Distance along line (cm)')
# plt.ylabel('Voxel Value')
# plt.tight_layout()
#
# plt.figure(figsize=[6, 6])
# plt.plot(distances_100_030, vals_100_030, linewidth=0.5, color='green', label='100 kV, 0.30 mm Al')
# plt.legend(loc=3)
# plt.ylim([min, max])
# plt.xlabel('Distance along line (cm)')
# plt.ylabel('Voxel Value')
# plt.tight_layout()
#
# plt.figure(figsize=[6, 6])
# plt.plot(distances_100_050, vals_100_050, linewidth=0.5, color='green', label='100 kV, 0.50 mm Al')
# plt.legend(loc=3)
# plt.ylim([min, max])
# plt.xlabel('Distance along line (cm)')
# plt.ylabel('Voxel Value')
# plt.tight_layout()
#
# plt.show()


# print(np.mean(vals_90_010[350:600]) - np.mean(vals_90_010[680:1000]))
# print(np.mean(vals_90_030[350:600]) - np.mean(vals_90_010[680:1000]))
# print(np.mean(vals_90_050[350:600]) - np.mean(vals_90_010[680:1000]))
# print(np.mean(vals_100_010[350:600]) - np.mean(vals_90_010[680:1000]))
# print(np.mean(vals_100_030[350:600]) - np.mean(vals_90_010[680:1000]))
# print(np.mean(vals_100_050[350:600]) - np.mean(vals_90_010[680:1000]))

# import os
# import re
# projection = 100
# scanprefix = 'Bristlecone Helical'
# folder0 = r'D:\Jorden(Data2)\transfer_149051_files_05ccbd6e\Helical 6_normalised'
# folder1 = r'F:\Helical6\Helical_6_corrected_simulatedLUT90kV0.10mmAl2.7_07042022'
# folder2 = r'F:\Helical6\Helical_6_corrected_simulatedLUT90kV0.50mmAl2.7_02042022'
# folder3 = r'F:\Helical6\Helical_6_corrected_simulatedLUT100kV0.10mmAl2.7_07042022'
# folder4 = r'F:\Helical6\Helical_6_corrected_simulatedLUT100kV0.50mmAl2.7_28032022'
#
# datafiles0 = [f for f in os.listdir(folder0) if re.match(scanprefix + r'*', f)]
# datafiles1 = [f for f in os.listdir(folder1) if re.match(scanprefix + r'*', f)]
# datafiles2 = [f for f in os.listdir(folder2) if re.match(scanprefix + r'*', f)]
# datafiles3 = [f for f in os.listdir(folder3) if re.match(scanprefix + r'*', f)]
# datafiles4 = [f for f in os.listdir(folder4) if re.match(scanprefix + r'*', f)]
#
# y1 = 200
# y2 = 800
# x1 = 10
# x2 = 20
# val0 = []
# val1 = []
# val2 = []
# val3 = []
# val4 = []
# projection = []
# for i in range(0, len(datafiles0), 50):
#     print(i)
#     val = np.mean(FH.im2array(folder0 + '\\' + datafiles0[i])[y1:y2, x1:x2])
#     val0.append(val)
#     val = np.mean(FH.im2array(folder1 + '\\' + datafiles1[i])[y1:y2, x1:x2])
#     val1.append(val)
#     val = np.mean(FH.im2array(folder2 + '\\' + datafiles2[i])[y1:y2, x1:x2])
#     val2.append(val)
#     val = np.mean(FH.im2array(folder3 + '\\' + datafiles3[i])[y1:y2, x1:x2])
#     val3.append(val)
#     val = np.mean(FH.im2array(folder4 + '\\' + datafiles4[i])[y1:y2, x1:x2])
#     val4.append(val)
#     projection.append(i)
#
# plt.plot(projection, val0, label='original')
# plt.plot(projection, val1, label='90 kV 0.10mm')
# plt.plot(projection, val2, label='90 kV 0.50mm')
# plt.plot(projection, val3, label='100 kV 0.10mm')
# plt.plot(projection, val4, label='100 kV 0.50mm')
# plt.legend()
# plt.show()
#

# original = FH.im2array(r'D:\Jorden(Data2)\transfer_149051_files_05ccbd6e\Helical 6_normalised\Bristlecone Helical_03855.tif')
# corrected = FH.im2array(r'F:\Helical6\Helical_6_corrected_simulatedLUT100kV0.10mmAl2.7_07042022\Bristlecone Helical_03855.tif')
#
# plt.figure(figsize=[8, 4])
# plt.imshow(original, cmap='gray')
# plt.colorbar()
# plt.axhline(y=500, color='r')
# plt.show()
#
# plt.figure(figsize=[8, 4])
# plt.imshow(corrected, cmap='gray')
# plt.colorbar()
# plt.axhline(y=500, color='r')
# plt.show()
#
# plt.plot(corrected[500], label='Corrected with simulated LUT')
# plt.plot(original[500], label='Original projection')
# plt.axhline(y=0.6681, color='g')
# plt.axhline(y=0.3141, color='g')
# plt.xlabel('pixel')
# plt.ylabel('pixel value')
# plt.legend()
# plt.show()

# Experimental__________________________________________________________________________________________________________
# distances_experimental, vals_experimental = np.loadtxt(r'F:\Helical6\Reslices\capping_experimenta_toosmallwedgel_100kV_0,50_slice6000.csv', skiprows=1, unpack=True, delimiter=',')
# vals_experimental = vals_experimental*4.11994E-05 - 7.00000E-01
#
# min = -0.3
# max = 1.5
# font = {'size' : 14}
# matplotlib.rc('font', **font)
#
# plt.figure(figsize=[6, 6])
# plt.plot(distances_experimental, vals_experimental, linewidth=0.5, color='green', label='Experimental LUT, 100 kV, 0.5 mm Al')
# plt.legend(loc=1)
# plt.ylim([min, max])
# plt.xlabel('Distance along line (cm)')
# plt.ylabel('Voxel Value')
# plt.tight_layout()
# plt.show()

# binary = FH.im2array2(r'F:\Helical6\Reslices\Binary.tif', slope=1/255, offset=0)[325:]
# notcorrected = FH.im2array2(r'F:\Helical6\Reslices\Reslice of Helical.tif', slope = 1.67849E-05, offset = -1.00000E-01)[325:]
# corrected_experimental = FH.im2array2(r'F:\Helical6\Reslices\Experimental_100kV_0.5mmAl\Reslice_experimental_100kV_0.5mmAl_toosmallwedge.tif', slope = 4.11994E-05, offset = -7.00000E-01)[326:]
#
# cnr_notcorrected = CNR(notcorrected, binary)
# snr_notcorrected = SNR(notcorrected, binary)
# csr_notcorrected = CSR(notcorrected, binary)
#
# cnr_experimental = CNR(corrected_experimental, binary)
# snr_experimental = SNR(corrected_experimental, binary)
# csr_experimental = CSR(corrected_experimental, binary)
#
# print('cnr_notcorrected={}'.format(cnr_notcorrected))
# print('snr_notcorrected={}'.format(snr_notcorrected))
# print('csr_notcorrected={}'.format(csr_notcorrected))
# print('(cnr/snr)_notcorrected={}'.format(cnr_notcorrected/snr_notcorrected))
#
# print('cnr_experimental={}'.format(cnr_experimental))
# print('snr_experimental={}'.format(snr_experimental))
# print('csr_experimental={}'.format(csr_experimental))
# print('(cnr/snr)_experimental={}'.format(cnr_experimental/snr_experimental))
#

#analysis of ellipse
heelreco = FH.im2array(r'F:\ellipse_heel\reconstructed\ellipse_sinogram_heel.tif')
noheelreco = FH.im2array(r'F:\ellipse_noheel\reconstructed\ellipse_sinogram_noheel.tif')

plt.imshow(heelreco)
plt.show()
plt.imshow(noheelreco)
plt.show()

plt.plot(heelreco[256], label='With heeling')
plt.plot(noheelreco[256], label='Without heeling')
plt.xlabel('Pixels')
plt.ylabel(r'Reconstructed attenuation coefficient (cm$^{-1}$)')
plt.legend()
plt.show()

