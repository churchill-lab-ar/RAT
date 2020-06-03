# -*- coding: utf-8 -*-
from numpy import pi, exp, sin, arcsin, cos, dot, real, complex, conjugate, sqrt, empty
import numpy as np
from scipy import interpolate
from scipy.constants import *
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def make_2x2_array(a, b, c, d, dtype=float):
    """
    Makes a 2x2 numpy array of [[a,b],[c,d]]
    Same as "numpy.array([[a,b],[c,d]], dtype=float)", but ten times faster
    """
    my_array = empty((2, 2), dtype=dtype)
    my_array[0, 0] = a
    my_array[0, 1] = b
    my_array[1, 0] = c
    my_array[1, 1] = d
    return my_array


def make_2x1_array(a, b, dtype=float):
    """
    Makes a 2x1 numpy array of [[a],[b]]
    Same as "numpy.array([[a],[b]], dtype=float)", but ten times faster
    """
    my_array = empty((2, 1), dtype=dtype)
    my_array[0, 0] = a
    my_array[1, 0] = b
    return my_array


def loadData(file_name, delimiter, **Dtype):
    if(Dtype['unit'] == 'nm' and Dtype['columnNumber'] == '2'):
        wavelength_nm, n_k = np.loadtxt(
            file_name, delimiter=delimiter, skiprows=0, unpack=True)
        n_k_interpolate = interpolate.interp1d(
            wavelength_nm, n_k, kind="linear")
        return wavelength_nm, n_k_interpolate
    elif(Dtype['unit'] == 'um' and Dtype['columnNumber'] == '2'):
        wavelength_um, n_k = np.loadtxt(
            file_name, delimiter=delimiter, skiprows=0, unpack=True)
        wavelength_nm = 1000. * wavelength_um
        n_k_interpolate = interpolate.interp1d(
            wavelength_nm, n_k, kind="linear")
        return wavelength_nm, n_k_interpolate
    elif(Dtype['unit'] == 'eV' and Dtype['columnNumber'] == '2'):
        energy_eV, n_k = np.loadtxt(
            file_name, delimiter=delimiter, skiprows=0, unpack=True)
        wavelength_nm = 1240./energy_eV
        n_k_interpolate = interpolate.interp1d(
            wavelength_nm, n_k, kind="linear")
        return wavelength_nm, n_k_interpolate
    elif(Dtype['unit'] == 'nm' and Dtype['columnNumber'] == '3'):
        wavelength_nm, n, k = np.loadtxt(
            file_name, delimiter=delimiter, skiprows=0, unpack=True)
        n_interpolate = interpolate.interp1d(wavelength_nm, n, kind="linear")
        k_interpolate = interpolate.interp1d(wavelength_nm, k, kind="linear")
        return wavelength_nm, n_interpolate, k_interpolate
    elif(Dtype['unit'] == 'um' and Dtype['columnNumber'] == '3'):
        wavelength_um, n, k = np.loadtxt(
            file_name, delimiter=delimiter, skiprows=0, unpack=True)
        wavelength_nm = 1000. * wavelength_um
        n_interpolate = interpolate.interp1d(wavelength_nm, n, kind="linear")
        k_interpolate = interpolate.interp1d(wavelength_nm, k, kind="linear")
        return wavelength_nm, n_interpolate, k_interpolate
    elif(Dtype['unit'] == 'eV' and Dtype['columnNumber'] == '3'):
        energy_eV, n, k = np.loadtxt(
            file_name, delimiter=delimiter, skiprows=0, unpack=True)
        wavelength_nm = 1240./energy_ev
        n_interpolate = interpolate.interp1d(wavelength_nm, n, kind="linear")
        k_interpolate = interpolate.interp1d(wavelength_nm, k, kind="linear")
        return wavelength_nm, n_interpolate, k_interpolate


def load_n_kappa_data(file_name, delimiter, **Dtype):
    if(Dtype['columnNumber'] == '2'):
        if(Dtype['unit'] == 'nm'):
            wavelength_nm, n_kappa = np.loadtxt(file_name, delimiter=delimiter, skiprows=0, unpack=True)
            n_kappa_interpolate = interpolate.interp1d(wavelength_nm, n_kappa, kind="linear")
        elif(Dtype['unit'] == 'um'):
            wavelength_um, n_kappa = np.loadtxt(file_name, delimiter=delimiter, skiprows=0, unpack=True)
            wavelength_nm = 1000. * wavelength_um
            n_kappa_interpolate = interpolate.interp1d(wavelength_nm, n_kappa, kind="linear")
        elif(Dtype['unit'] == 'eV'):
            energy_eV, n_kappa = np.loadtxt(
                file_name, delimiter=delimiter, skiprows=0, unpack=True)
            wavelength_nm = 1240./energy_eV
            n_kappa_interpolate = interpolate.interp1d(
                wavelength_nm, n_kappa, kind="linear")
        return wavelength_nm, n_kappa_interpolate
    elif(Dtype['columnNumber'] == '3'):
        if(Dtype['unit'] == 'nm'):
            wavelength_nm, n, kappa = np.loadtxt(
                file_name, delimiter=delimiter, skiprows=0, unpack=True)
            n_interpolate = interpolate.interp1d(
                wavelength_nm, n, kind="linear")
            kappa_interpolate = interpolate.interp1d(
                wavelength_nm, kappa, kind="linear")
        elif(Dtype['unit'] == 'um'):
            wavelength_um, n, kappa = np.loadtxt(
                file_name, delimiter=delimiter, skiprows=0, unpack=True)
            wavelength_nm = 1000. * wavelength_um
            n_interpolate = interpolate.interp1d(
                wavelength_nm, n, kind="linear")
            kappa_interpolate = interpolate.interp1d(
                wavelength_nm, kappa, kind="linear")
        elif(Dtype['unit'] == 'eV'):
            energy_eV, n, kappa = np.loadtxt(
                file_name, delimiter=delimiter, skiprows=0, unpack=True)
            wavelength_nm = 1240./energy_ev
            n_interpolate = interpolate.interp1d(
                wavelength_nm, n, kind="linear")
            kappa_interpolate = interpolate.interp1d(
                wavelength_nm, kappa, kind="linear")
        return wavelength_nm, n_interpolate, kappa_interpolate


def load_epsilons_data(file_name, delimiter, **Dtype):
    if(Dtype['columnNumber'] == '2'):
        if(Dtype['unit'] == 'nm'):
            wavelength_nm, epsilon1_2 = np.loadtxt(file_name, delimiter=delimiter, skiprows=0, unpack=True)
            epsilon1_2_interpolate = interpolate.interp1d(wavelength_nm, n_k, kind="linear")
        elif(Dtype['unit'] == 'um'):
            wavelength_um, epsilon1_2 = np.loadtxt(file_name, delimiter=delimiter, skiprows=0, unpack=True)
            wavelength_nm = 1000. * wavelength_um
            epsilon1_2_interpolate = interpolate.interp1d(wavelength_nm, epsilon1_2, kind="linear")
        elif(Dtype['unit'] == 'eV'):
            energy_eV, epsilon1_2 = np.loadtxt(file_name, delimiter=delimiter, skiprows=0, unpack=True)
            wavelength_nm = 1240./energy_eV
            epsilon1_2_interpolate = interpolate.interp1d(wavelength_nm, epsilon1_2, kind="linear")
        elif(Dtype['unit'] == 'Ryd'):
            energy_Ryd, epsilon1_2 = np.loadtxt(file_name, delimiter=delimiter, skiprows=0, unpack=True)
            # 1 Ryd = 13.6056980659 eV
            wavelength_nm = 1240./(13.6056980659 * energy_Ryd)
            epsilon1_2_interpolate = interpolate.interp1d(wavelength_nm, epsilon1_2, kind="linear")
        return wavelength_nm, epsilon1_2_interpolate
    elif(Dtype['columnNumber'] == '3'):
        if(Dtype['unit'] == 'nm'):
            wavelength_nm, epsilon1, epsilon2 = np.loadtxt(file_name, delimiter=delimiter, skiprows=0, unpack=True)
            epsilon1_interpolate = interpolate.interp1d(wavelength_nm, epsilon1, kind="linear")
            epsilon2_interpolate = interpolate.interp1d(wavelength_nm, epsilon2, kind="linear")
        elif(Dtype['unit'] == 'um'):
            wavelength_um, epsilon1, epsilon2 = np.loadtxt(
                file_name, delimiter=delimiter, skiprows=0, unpack=True)
            wavelength_nm = 1000. * wavelength_um
            epsilon1_interpolate = interpolate.interp1d(
                wavelength_nm, epsilon1, kind="linear")
            epsilon2_interpolate = interpolate.interp1d(
                wavelength_nm, epsilon2, kind="linear")
        elif(Dtype['unit'] == 'eV'):
            energy_eV, epsilon1, epsilon2 = np.loadtxt(
                file_name, delimiter=delimiter, skiprows=0, unpack=True)
            wavelength_nm = 1240./energy_ev
            epsilon1_interpolate = interpolate.interp1d(
                wavelength_nm, epsilon1, kind="linear")
            epsilon2_interpolate = interpolate.interp1d(
                wavelength_nm, epsilon2, kind="linear")
        elif(Dtype['unit'] == 'Ryd'):
            energy_Ryd, epsilon1, epsilon2 = np.loadtxt(
                file_name, delimiter=delimiter, skiprows=0, unpack=True)
            wavelength_nm = 1240./(13.6056980659 * energy_Ryd)
            epsilon1_interpolate = interpolate.interp1d(
                wavelength_nm, epsilon1, kind="linear")
            epsilon2_interpolate = interpolate.interp1d(
                wavelength_nm, epsilon2, kind="linear")
        return wavelength_nm, epsilon1_interpolate, epsilon2_interpolate


def epsilon_to_n(epsilon1, epsilon2):
    complex_epsilon = epsilon1 - 1j * epsilon2
    complex_n = sqrt(complex_epsilon)
    return complex_n


def snell(complex_index_1, complex_index_2, incident_angle):
    n1 = complex_index_1
    n2 = complex_index_2
    theta_i = incident_angle
    theta_t = arcsin((n1/n2) * sin(theta_i))
    return theta_t


def phase(thickness, complex_index, angle, wavelength):
    phase = (2.*pi*thickness*complex_index*cos(angle))/wavelength
    return phase


def waveVector(permeability, complex_index, incident_angle, wavelength):
    # The Optical Admittance of The Medium, y :: S (siemens)
    y = 1.0/(c * mu_0 * permeability)
    # The Wave Vector, k ::
    k = (2 * pi * c * mu_0 * permeability * y)/wavelength
    # The Tilted Optical Admittance of The Medium, \eta :: S (siemens)
    eta_s = y * complex_index * cos(incident_angle)
    eta_p = y * complex_index / cos(incident_angle)
    # The Wave Vector, k ::
    k_s = (2 * pi * c * mu_0 * permeability * eta_s) / \
        (cos(incident_angle) * wavelength)
    k_p = (2 * pi * c * mu_0 * permeability *
           eta_p * cos(incident_angle)) / wavelength
    return k, k_s, k_p


def LayerMatrix(thickness, permeability, complex_index, incident_angle, wavelength, **layer_position):
    # The Optical Admittance of The Medium, y :: S (siemens)
    y = 1.0/(c * mu_0 * permeability)
    # The Wave Vector, k ::
    k = (2 * pi * c * mu_0 * permeability * y)/wavelength
    # The Tilted Optical Admittance of The Medium, \eta :: S (siemens)
    eta_s = y * complex_index * cos(incident_angle)
    eta_p = y * complex_index / cos(incident_angle)
    # The Wave Vector, k ::
    k_s = (2 * pi * c * mu_0 * permeability * eta_s) / \
        (cos(incident_angle) * wavelength)
    k_p = (2 * pi * c * mu_0 * permeability *
           eta_p * cos(incident_angle)) / wavelength
    if(layer_position['position'] == '0' or layer_position['position'] == 'last'):
        phase = 0
    else:
        phase = (2.*pi*thickness*complex_index*cos(incident_angle))/wavelength
    if(layer_position['position'] == 'last'):
        Ms = make_2x1_array(1, eta_s, dtype=complex)
        Mp = make_2x1_array(1, eta_p, dtype=complex)
    else:
        Ms = make_2x2_array(cos(phase), (1j * sin(phase) / eta_s),
                            (1j * sin(phase) * eta_s), cos(phase), dtype=complex)
        Mp = make_2x2_array(cos(phase), (1j * sin(phase) / eta_p),
                            (1j * sin(phase) * eta_p), cos(phase), dtype=complex)
    return eta_s, eta_p, Ms, Mp


def TransferMatrix(*layers):
    LayerNumber = len(layers)
    totalMatrix = make_2x2_array(1, 0, 0, 1, dtype=complex)
    for i in range(0, LayerNumber):
        totalMatrix = dot(totalMatrix, layers[i])
    return totalMatrix


def Reflectance(n0, TransferMatrix):
    B = TransferMatrix[0][0]
    C = TransferMatrix[1][0]
    # Reflectance
    numerator = n0*B - C
    denominator = n0*B + C
    r = numerator/denominator
    R = r * conjugate(r)
    return R


def Transmittance(n0, n_substrate, TransferMatrix):
    B = TransferMatrix[0][0]
    C = TransferMatrix[1][0]
    # Transmittance
    numerator = 4*n0*real(n_substrate)
    denominator = (n0*B + C) * conjugate((n0*B + C))
    T = numerator/denominator
    return T


def Absorbance(n0, n_substrate, TransferMatrix):
    B = TransferMatrix[0][0]
    C = TransferMatrix[1][0]
    # Absorbance
    numerator = 4*n0*real(((B*conjugate(C)) - n_substrate))
    denominator = (n0*B + C) * conjugate((n0*B + C))
    A = numerator/denominator
    return A


def Contrast(R_flake, R_substrate):
    # Contrast
    numerator = R_flake - R_substrate
    denominator = R_flake + R_substrate
    contrast = numerator/denominator
    return contrast


def plotFigure(FigureNumber, xData, yData, **labels):
    plt.figure(FigureNumber)
    for i in range(0, len(yData)):
        plt.plot(xData, yData[i], label=labels['label'][i])
    plt.xlim(min(xData), max(xData))
    plt.xlabel(labels['xAxis'], fontweight='bold')
    plt.ylabel(labels['yAxis'], fontweight='bold')
    plt.title(labels['title'], fontweight='bold')
    plt.legend(loc=8, bbox_to_anchor=(0.65, 0.1))
    plt.savefig(labels['FigureFileName'])
    # plt.show()

def plotContrastFigure(FigureNumber, xData, yData, **labels):
    plt.figure(FigureNumber)
    for i in range(0, len(yData)):
        plt.plot(xData, yData[i], label=labels['label'][i])
    plt.xlim(min(xData), max(xData))
    plt.ylim(-1, 1)
    plt.xlabel(labels['xAxis'], fontweight='bold')
    plt.ylabel(labels['yAxis'], fontweight='bold')
    plt.title(labels['title'], fontweight='bold')
    plt.legend(loc=8, bbox_to_anchor=(0.65, 0.1))
    plt.savefig(labels['FigureFileName'])
    # plt.show()

def plotImage(FigureNumber, xData, yData, colorData, **labels):
    plt.figure(FigureNumber)
    plt.imshow(colorData, origin='lower', interpolation='none', cmap="RdBu")
    cbar = plt.colorbar()
    cbar.set_label(labels['colorBar'], rotation=270, labelpad=15, fontweight='bold')
    plt.xticks([0, 40, 90, 140, 190, 240, 290], ["10", "50", "100", "150", "200", "250", "300"])
    plt.yticks([0, 40, 90, 140, 190, 240, 290], ["10", "50", "100", "150", "200", "250", "300"])
    plt.xlabel(labels['xAxis'], fontweight='bold')
    plt.ylabel(labels['yAxis'], fontweight='bold')
    plt.title(labels['title'], fontweight='bold')
    plt.savefig(labels['FigureFileName'])
    # plt.show()
