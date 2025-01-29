'''
Written by Jorden De Bolle
Last update: 18/07/2022
'''
import FileHandler as FH
import numpy as np
import matplotlib.pyplot as plt
import Functions
import seaborn as sns

datafolder = r'C:\Users\jdebolle\Documents\PythonProjects\thesis_code\Data'
savefolder = r'C:\Users\jdebolle\Documents\PythonProjects\thesis_code\Projections'

def CreateArtificialSpectrum(voltage, number_of_spectra, peak_energy, peak_height, stepsize):
    nbins = int(np.round(voltage/stepsize))
    spec = np.zeros([nbins, number_of_spectra])
    energies = np.linspace(0, voltage, nbins)

    nbins_final = int(np.round(voltage/0.25))
    spec_final = np.zeros([nbins_final, number_of_spectra])
    energies_final = np.linspace(0, voltage, nbins_final)

    for i in range(number_of_spectra):
        #triangle
        peak_index = int(np.round(peak_energy / stepsize - i))
        spec[0:peak_index, i] = np.linspace(0, peak_height-i*peak_height/3000, peak_index)
        spec[peak_index:, i] = np.linspace(peak_height-i*peak_height/3000, 0, nbins-peak_index)

        spec_final[:, i] = Functions.interpolate(energies, energies_final, spec[:, i])

        # add spikes
        spike_index = nbins_final/2
        spec_final[int(spike_index+10), i] = 1e-3
        spec_final[int(spike_index -10), i] = 5e-3
    return spec_final, energies_final

def Smoothspectrum(tube, voltage, sigma):
    file = datafolder + '\\SpectrumFilesHeel\\' + tube + '_W_' + str(voltage) + '.spec'
    data = FH.tex2array(file)
    final = np.zeros(np.shape(data))
    Energies = np.transpose(data)[0]
    data = data[:, 1:]
    smoothed = Functions.gaussianFirstAxis(data, sigma, inplace=False, mode='reflect')
    final[:, 0] = Energies
    final[:, 1:] = smoothed
    file = datafolder + '\\SpectrumFilesHeelSmoothed\\' + tube + '_W_' + str(voltage) + '.spec'
    FH.array2tex(file, final)

def GetSNR(spec):
    rows, columns = np.shape(spec)
    region = spec[np.int(rows/2), np.int(columns/2)-20:np.int(columns/2)+20]
    std = np.std(region)
    mean = np.mean(region)
    return mean/std

######################################################################################################################
volts = np.arange(40, 245, 5)
peaks = volts/5
# SNR_real = []
# SNR_artificial = []
# SNR_nonoise = []
# for i in range(len(volts)):
#     data = FH.tex2array(r'C:\Users\jdebolle\Documents\PythonProjects\thesis_code\Data\SpectrumFilesHeel\Hector_W_' + str(volts[i]) + '.spec')
#     Energies = np.transpose(data)[0]
#     data = data[:,1:]
#     SNR_real.append(GetSNR(data))
#
#     spec, energies = CreateArtificialSpectrum(volts[i], 500, peaks[i], 1e-4, 0.01)
#     SNR_nonoise.append(GetSNR(spec))
#     noise_spec = Functions.addSemiPoissonNoise(spec, 0.0018)
#     SNR_artificial.append(GetSNR(noise_spec))
# plt.plot(volts, SNR_nonoise, 'bx', label='Artificial spectra without noise')
# plt.plot(volts, SNR_real, 'rx', label='Heel spectra from Monte Carlo')
# plt.plot(volts, SNR_artificial, 'gx', label='artificial spectra with noise')
# plt.legend()
# plt.title('SNR of the Monte Carlo heel spectra and \n the artificial spectra for different voltages')
# plt.xlabel('Voltage')
# plt.ylabel('SNR')
# plt.show()
# print('average SNR: {}'.format(SNR/len(volts)))
#
# # data = FH.tex2array(r'C:\Users\jdebolle\Documents\PythonProjects\thesis_code\Data\SpectrumFilesHeel\Hector_W_240.spec')
# # Energies = np.transpose(data)[0]
# # data = data[:,1:]
# # plt.plot(Energies, data[:, 100])
# # plt.yscale('log')
# # plt.show()
#
# print('__________________________________________________________________')
# spec, energies = CreateArtificialSpectrum(60, 500, 15, 1e-4, 0.01)
# noise_spec = Functions.addSemiPoissonNoise(spec, 0.0018)
# print(GetSNR(noise_spec))
# # plt.plot(spec[123, :])
# # plt.show()
#
#
#
# sigmas = np.arange(1, 150, 1)
# best_sigmas = []
# sns.set(style="darkgrid")
# colors = ['olive', 'cadetblue', 'magenta', 'tomato', 'plum', 'lightcoral', 'aqua']
# j = 0
# for i in range(len(volts)):
#     spec, energies = CreateArtificialSpectrum(volts[i], 500, peaks[i], 1e-4, 0.01)
#     noise_spec = Functions.addSemiPoissonNoise(spec, 0.0018)
#     rmses = []
#     for sigma in sigmas:
#         smoothed = Functions.gaussianFirstAxis(noise_spec, sigma=sigma, mode='reflect')
#         rmse = Functions.RMSE(smoothed, spec)
#         rmses.append(rmse)
#     rmses = np.array(rmses)
#     min_index = np.where(rmses==np.min(rmses))[0][0]
#     best_sigmas.append(sigmas[min_index])
#     print('{} kV --> best sigma = {}'.format(volts[i], sigmas[min_index]))
#     if i % 6 == 0:
#         plt.plot(sigmas, rmses, label=str(volts[i])+'kV', color=colors[j])
#         plt.plot(sigmas[min_index], rmses[min_index], color=colors[j], marker='o')
#         j += 1
# plt.yscale('log')
# plt.ylim(7e-7, 1e-5)
# plt.yticks([1e-6, 3e-6, 5e-6, 7e-6, 1e-5], [r'$10^{-6}$', r'$3\times10^{-6}$', r'$5\times10^{-6}$', r'$7\times10^{-6}$', r'$10^{-5}$'])
# plt.legend(loc='best')
# plt.xlabel(r'$sigma$')
# plt.ylabel('RMSE')
# plt.title('RMSE as function of the standard deviation of the Gaussian filter \n kernel for the artificial spectra')
# plt.show()
#
# print('best sigmas: {}'.format(best_sigmas))
# # best sigmas mode reflect: [59, 62, 64, 69, 67, 72, 75, 75, 75, 68, 87, 73, 74, 84, 75, 76, 80, 80, 79, 80, 84, 81, 80, 78, 89, 78, 85, 80, 82, 83, 82, 82, 80, 83, 82, 81, 79, 82, 83, 85, 82]
# # best sigmas mode nearest: [31, 34, 25, 17, 23, 28, 22, 20, 22, 27, 27, 27, 29, 26, 28, 23, 26, 24, 20, 22, 24, 24, 25, 27, 28, 26, 26, 23, 26, 27, 22, 25, 28, 26, 26, 23, 25, 25, 26, 25, 25]
# # best widths for median filter mode reflect: [227, 193, 239, 261, 229, 285, 267, 311, 285, 283, 279, 291, 309, 275, 297, 323, 271, 285, 287, 295, 307, 281, 283, 307, 321, 305, 289, 311, 309, 303, 291, 317, 311, 287, 309, 295, 311, 313, 295, 301, 307]
# # best widths for median filter mode nearest: [51, 63, 61, 51, 49, 63, 65, 49, 61, 61, 55, 57, 51, 47, 59, 53, 57, 43, 51, 63, 61, 55, 59, 51, 55, 55, 55, 55, 57, 59, 63, 51, 57, 59, 59, 53, 53, 55, 53, 59, 57]
# # best widths for mean filter mode reflect: [195, 199, 222, 231, 222, 235, 231, 235, 264, 243, 261, 254, 276, 265, 263, 268, 259, 279, 251, 261, 285, 281, 267, 270, 265, 271, 277, 263, 273, 260, 268, 261, 263, 266, 282, 269, 281, 270, 266, 275, 267]
# # best widths for mean filter mode nearest: [63, 96, 62, 88, 61, 64, 91, 85, 69, 81, 76, 81, 88, 85, 97, 79, 78, 73, 81, 77, 75, 87, 85, 91, 83, 76, 74, 79, 85, 83, 75, 79, 77, 75, 79, 81, 78, 81, 83, 73, 83]
volts = np.arange(40, 245, 5)
best_sigmas = [59, 62, 64, 69, 67, 72, 75, 75, 75, 68, 87, 73, 74, 84, 75, 76, 80, 80, 79, 80, 84, 81, 80, 78, 89, 78, 85, 80, 82, 83, 82, 82, 80, 83, 82, 81, 79, 82, 83, 85, 82]






for i in range(len(volts)):
    print('Smoothing spectrum of {} kV'.format(volts[i]))
    Smoothspectrum('Hector', volts[i], best_sigmas[i])


########################################################################################################################
# Plotting results
sns.set_style('darkgrid')



data = FH.tex2array(r'C:\Users\jdebolle\Documents\PythonProjects\thesis_code\Data\SpectrumFilesHeel\Hector_W_100.spec')
Energies = np.transpose(data)[0]
data = data[:,1:]

plt.plot(data[150, :], label='Photon energy: {:.3f} keV'.format(Energies[150]), linewidth=0.75)
plt.plot(data[250, :], label='Photon energy: {:.3f} keV'.format(Energies[250]), linewidth=0.75)
plt.plot(data[350, :], label='Photon energy: {:.3f} keV'.format(Energies[350]), linewidth=0.75)
plt.legend()
plt.title('Heel spectrum of 100 kV before smoothing')
plt.ylabel('Photons per electron (Sr$^{-1}$ keV$^{-1}$)')
plt.xlabel('Beam angle (mrad)')
plt.show()

plt.plot(Energies, data[:, 50], label='50 mrad', linewidth=0.75)
plt.plot(Energies, data[:, 150], label='150 mrad', linewidth=0.75)
plt.plot(Energies, data[:, 250], label='250 mrad', linewidth=0.75)
plt.plot(Energies, data[:, 350], label='350 mrad', linewidth=0.75)
plt.plot(Energies, data[:, 450], label='450 mrad', linewidth=0.75)
plt.ylim(1e-6, 1e-2)
plt.yscale('log')
plt.title('Heel spectrum of 100 kV before smoothing')
plt.ylabel('Photons per electron (Sr$^{-1}$ keV$^{-1}$)')
plt.xlabel('Energy (keV)')
plt.legend()
plt.show()

data_smoothed = FH.tex2array(r'C:\Users\jdebolle\Documents\PythonProjects\thesis_code\Data\SpectrumFilesHeelSmoothed\Hector_W_100.spec')
Energies = np.transpose(data_smoothed)[0]
data_smoothed = data_smoothed[:,1:]

plt.plot(data_smoothed[150, :], label='Photon energy: {:.3f} keV'.format(Energies[150]), linewidth=0.75)
plt.plot(data_smoothed[250, :], label='Photon energy: {:.3f} keV'.format(Energies[250]), linewidth=0.75)
plt.plot(data_smoothed[350, :], label='Photon energy: {:.3f} keV'.format(Energies[350]), linewidth=0.75)
plt.legend()
plt.title('Heel spectrum of 100 kV after smoothing \n (Gaussian filter, reflection mode)')
plt.ylabel('Photons per electron (Sr$^{-1}$ keV$^{-1}$)')
plt.xlabel('Beam angle (mrad)')
plt.show()

plt.plot(Energies, data_smoothed[:, 50], label='50 mrad', linewidth=0.75)
plt.plot(Energies, data_smoothed[:, 150], label='150 mrad', linewidth=0.75)
plt.plot(Energies, data_smoothed[:, 250], label='250 mrad', linewidth=0.75)
plt.plot(Energies, data_smoothed[:, 350], label='350 mrad', linewidth=0.75)
plt.plot(Energies, data_smoothed[:, 450], label='450 mrad', linewidth=0.75)
plt.yscale('log')
plt.title('Heel spectrum of 100 kV after smoothing \n (Gaussian filter, reflection mode)')
plt.ylabel('Photons per electron (Sr$^{-1}$ keV$^{-1}$)')
plt.xlabel('Energy (keV)')
plt.ylim(1e-6, 1e-2)
plt.legend()
plt.show()

plt.plot(data[150, :], label='Photon energy: {:.3f} keV'.format(Energies[150]), color='olive', linewidth=0.75)
plt.plot(data[250, :], label='Photon energy: {:.3f} keV'.format(Energies[250]), color='skyblue', linewidth=0.75)
plt.plot(data[350, :], label='Photon energy: {:.3f} keV'.format(Energies[350]), color='salmon', linewidth=0.75)
plt.plot(data_smoothed[150, :], color='red', label='Smoothed')
plt.plot(data_smoothed[250, :], color='red')
plt.plot(data_smoothed[350, :], color='red')
plt.legend()
plt.title('Heel spectrum of 100 kV before and after smoothing \n (Gaussian filter, reflection mode)')
plt.ylabel('Photons per electron (Sr$^{-1}$ keV$^{-1}$)')
plt.xlabel('Beam angle (mrad)')
plt.show()
#################################################################################################
# artificial spectra
spec, energies = CreateArtificialSpectrum(100, 500, 20, 1e-4, 0.01)
noise_spec = Functions.addSemiPoissonNoise(spec, 0.0018)
plt.plot(spec[150, :], label='Photon energy: {:.3f} keV'.format(energies[150]), linewidth=0.75)
plt.plot(spec[250, :], label='Photon energy: {:.3f} keV'.format(energies[250]), linewidth=0.75)
plt.plot(spec[350, :], label='Photon energy: {:.3f} keV'.format(energies[350]), linewidth=0.75)
plt.legend()
plt.title('Artificial heel spectrum of 100 kV')
plt.ylabel('Photons per electron (Sr$^{-1}$ keV$^{-1}$)')
plt.xlabel('Beam angle (mrad)')
plt.show()

plt.plot(energies, spec[:, 50], label='50 mrad', linewidth=0.75)
plt.plot(energies, spec[:, 150], label='150 mrad', linewidth=0.75)
plt.plot(energies, spec[:, 250], label='250 mrad', linewidth=0.75)
plt.plot(energies, spec[:, 350], label='350 mrad', linewidth=0.75)
plt.plot(energies, spec[:, 450], label='450 mrad', linewidth=0.75)
plt.yscale('log')
plt.title('Artificial heel spectrum of 100 kV')
plt.ylabel('Photons per electron (Sr$^{-1}$ keV$^{-1}$)')
plt.xlabel('Energy (keV)')
plt.ylim(1e-6, 1e-2)
plt.legend()
plt.show()

plt.plot(noise_spec[150, :], label='Photon energy: {:.3f} keV'.format(energies[150]), linewidth=0.75)
plt.plot(noise_spec[250, :], label='Photon energy: {:.3f} keV'.format(energies[250]), linewidth=0.75)
plt.plot(noise_spec[350, :], label='Photon energy: {:.3f} keV'.format(energies[350]), linewidth=0.75)
plt.legend()
plt.title('Artificial heel spectrum of 100 kV after adding semi-Poisson noise')
plt.ylabel('Photons per electron (Sr$^{-1}$ keV$^{-1}$)')
plt.xlabel('Beam angle (mrad)')
plt.show()

plt.plot(energies, noise_spec[:, 50], label='50 mrad', linewidth=0.75)
plt.plot(energies, noise_spec[:, 150], label='150 mrad', linewidth=0.75)
plt.plot(energies, noise_spec[:, 250], label='250 mrad', linewidth=0.75)
plt.plot(energies, noise_spec[:, 350], label='350 mrad', linewidth=0.75)
plt.plot(energies, noise_spec[:, 450], label='450 mrad', linewidth=0.75)
plt.yscale('log')
plt.title('Artificial heel spectrum of 100 kV after adding semi-Poisson noise')
plt.ylabel('Photons per electron (Sr$^{-1}$ keV$^{-1}$)')
plt.xlabel('Energy (keV)')
plt.ylim(1e-6, 1e-2)
plt.legend()
plt.show()

smoothed = Functions.gaussianFirstAxis(noise_spec, sigma=74, mode='reflect')

plt.plot(smoothed[150, :], label='Photon energy: {:.3f} keV'.format(energies[150]), linewidth=0.75)
plt.plot(smoothed[250, :], label='Photon energy: {:.3f} keV'.format(energies[250]), linewidth=0.75)
plt.plot(smoothed[350, :], label='Photon energy: {:.3f} keV'.format(energies[350]), linewidth=0.75)
plt.legend()
plt.title('Artificial heel spectrum of 100 kV after smoothing \n (Gaussian filter, reflection mode)')
plt.ylabel('Photons per electron (Sr$^{-1}$ keV$^{-1}$)')
plt.xlabel('Beam angle (mrad)')
plt.show()

plt.plot(noise_spec[150, :], label='Photon energy: {:.3f} keV'.format(energies[150]), color='olive', linewidth=0.75)
plt.plot(noise_spec[250, :], label='Photon energy: {:.3f} keV'.format(energies[250]), color='skyblue', linewidth=0.75)
plt.plot(noise_spec[350, :], label='Photon energy: {:.3f} keV'.format(energies[350]), color='salmon', linewidth=0.75)
plt.plot(smoothed[150, :], color='red', label='Smoothed')
plt.plot(smoothed[250, :], color='red')
plt.plot(smoothed[350, :], color='red')
plt.legend()
plt.title('Artificial heel spectrum of 100 kV before and after smoothing \n (Gaussian filter, reflection mode)')
plt.ylabel('Photons per electron (Sr$^{-1}$ keV$^{-1}$)')
plt.xlabel('Beam angle (mrad)')
plt.show()

plt.plot(spec[150, :], label='Photon energy: {:.3f} keV'.format(energies[150]), color='olive', linewidth=0.75)
plt.plot(spec[250, :], label='Photon energy: {:.3f} keV'.format(energies[250]), color='skyblue', linewidth=0.75)
plt.plot(spec[350, :], label='Photon energy: {:.3f} keV'.format(energies[350]), color='salmon', linewidth=0.75)
plt.plot(smoothed[150, :], color='red', label='Smoothed', linewidth=0.75)
plt.plot(smoothed[250, :], color='red', linewidth=0.75)
plt.plot(smoothed[350, :], color='red', linewidth=0.75)
plt.legend()
plt.title('Artificial heel spectrum of 100 kV before adding noise and after smoothing \n (Gaussian filter, reflection mode)')
plt.ylabel('Photons per electron (Sr$^{-1}$ keV$^{-1}$)')
plt.xlabel('Beam angle (mrad)')
plt.show()

plt.plot(energies, smoothed[:, 50], label='50 mrad', linewidth=0.75)
plt.plot(energies, smoothed[:, 150], label='150 mrad', linewidth=0.75)
plt.plot(energies, smoothed[:, 250], label='250 mrad', linewidth=0.75)
plt.plot(energies, smoothed[:, 350], label='350 mrad', linewidth=0.75)
plt.plot(energies, smoothed[:, 450], label='450 mrad', linewidth=0.75)
plt.yscale('log')
plt.title('Artificial heel spectrum of 100 kV after smoothing \n (Gaussian filter, reflection mode)')
plt.ylabel('Photons per electron (Sr$^{-1}$ keV$^{-1}$)')
plt.xlabel('Energy (keV)')
plt.ylim(1e-6, 1e-2)
plt.legend()
plt.show()

plt.plot(energies, np.abs((smoothed - spec))[:, 50], color='red', linewidth=0.5)
plt.xlabel('Energy (keV)')
plt.ylabel('Absolute error (Photons per electron (Sr$^{-1}$ keV$^{-1}$)')
plt.title('Absolute error of smoothed spectrum (Gaussian filter, reflection mode) \n with respect to original artificial spectrum of 100 kV \n and beam angle of 50 mrad')
plt.yscale('log')
plt.show()
plt.plot(energies, np.abs((smoothed - spec))[:, 150], color='red', linewidth=0.5)
plt.xlabel('Energy (keV)')
plt.ylabel('Absolute error (Photons per electron (Sr$^{-1}$ keV$^{-1}$)')
plt.title('Absolute error of smoothed spectrum (Gaussian filter, reflection mode) \n with respect to original artificial spectrum of 100 kV \n and beam angle of 150 mrad')
plt.yscale('log')
plt.show()
plt.plot(energies, np.abs((smoothed - spec))[:, 250], color='red', linewidth=0.5)
plt.xlabel('Energy (keV)')
plt.ylabel('Absolute error (Photons per electron (Sr$^{-1}$ keV$^{-1}$)')
plt.title('Absolute error of smoothed spectrum (Gaussian filter, reflection mode) \n with respect to original artificial spectrum of 100 kV \n and beam angle of 250 mrad')
plt.yscale('log')
plt.show()
plt.plot(energies, np.abs((smoothed - spec))[:, 350], color='red', linewidth=0.5)
plt.xlabel('Energy (keV)')
plt.ylabel('Absolute error (Photons per electron (Sr$^{-1}$ keV$^{-1}$)')
plt.title('Absolute error of smoothed spectrum (Gaussian filter, reflection mode) \n with respect to original artificial spectrum of 100 kV \n and beam angle of 350 mrad')
plt.yscale('log')
plt.show()
plt.plot(energies, np.abs((smoothed - spec))[:, 450], color='red', linewidth=0.5)
plt.xlabel('Energy (keV)')
plt.ylabel('Absolute error (Photons per electron (Sr$^{-1}$ keV$^{-1}$)')
plt.title('Absolute error of smoothed spectrum (Gaussian filter, reflection mode) \n with respect to original artificial spectrum of 100 kV \n and beam angle of 450 mrad')
plt.yscale('log')
plt.show()

maxerrors = np.max(np.abs((smoothed - spec)), axis=0)
maxerror_indices = np.argmax(np.abs((smoothed - spec)), axis=0)
maxerror_specvalues = []
for i in range(len(maxerror_indices)):
    maxerror_specvalues.append(spec[maxerror_indices[i], i])
maxerror_specvalues = np.array(maxerror_specvalues)

plt.plot(maxerrors)
plt.xlabel('Beam angle (mrad)')
plt.ylabel('Maximal absolute error (Photons per electron (Sr$^{-1}$ keV$^{-1}$)')
plt.title('Maximal absolute error for every beam angle after \n smoothing of a 100 kV artificial spectrum \n (Gaussian filter, reflection mode)')
plt.yscale('log')
plt.show()

plt.plot(maxerrors/maxerror_specvalues)
plt.xlabel('Beam angle (mrad)')
plt.ylabel('Relative maximal error')
plt.title('Relative maximal error for every beam angle after \n smoothing of a 100 kV artificial spectrum \n (Gaussian filter, reflection mode)')
plt.show()

plt.plot(energies, spec[:, 250], label='Original spectrum (without noise)', color='red', linewidth=0.5)
plt.plot(energies, smoothed[:, 250], color='green', linewidth=0.5, label='After smoothing')
plt.yscale('log')
plt.title('Comparison between original artificial spectrum (without noise) \n of 100 kV and smoothed spectrum \n (Gaussian filter, reflection mode)')
plt.ylabel('Photons per electron (Sr$^{-1}$ keV$^{-1}$)')
plt.xlabel('Energy (keV)')
plt.show()
