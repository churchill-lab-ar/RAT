# -*- coding: utf-8 -*-
import sys
import numpy as np
from numpy import cos, inf, zeros, array, exp, conj, nan, isnan, pi, sqrt, sin, arcsin, dot, real, complex, empty
from scipy.constants import *
import pandas as pd
sys.path.append("..")
from RAT_core import RAT as RAT

################################################################################
################################################################################
wavelength = 780  # nm
# wavelength = 1550  # nm
################################################################################
################################################################################
# Incident Angle, theta :: degree
incident_angle_value = 0
if (incident_angle_value >= 90):
    incident_angle_value = incident_angle_value - 90
else:
    incident_angle_value = incident_angle_value
incident_angle = incident_angle_value * degree
################################################################################
################################################################################
# Air #
Air_thickness_value = inf
Air_epsilon_r_value = 1
Air_mu_r_value = 1


# hBN #
hBN_epsilon_r_value = 1
hBN_mu_r_value = 1
hBN_thickness = 150
hBN_thickness_value = hBN_thickness
################################################################################
# Journal Name::    Physical Review Letters
# Paper Title::     Refractive Index Dispersion of Hexagonal Boron Nitride in the Visible and Near‐Infrared
# Authors::         Seong‐Yeon Lee, Tae‐Young Jeong, Suyong Jung, and Ki‐Ju Yee
# DOI::             https://doi.org/10.1002/pssb.201800417
# Year::            2019
hBN_index_real_value = 2.2


# BP #
BP_epsilon_r_value = 1
BP_mu_r_value = 1
BP_thickness = 40
BP_thickness_value = BP_thickness


# Au #
Au_thickness_value = 60
Au_epsilon_r_value = 1
Au_mu_r_value = 1


# SiO2 #
# SiO2_thickness_value = 90 # nm
SiO2_thickness_value = 300 # nm
SiO2_epsilon_r_value = 1
SiO2_mu_r_value = 1


# Si #
Si_thickness_value = inf
Si_epsilon_r_value = 1
Si_mu_r_value = 1
################################################################################
################################################################################
### Air ###
Air_thickness = Air_thickness_value
Air_mu_r = Air_mu_r_value
Air_complex_index = 1 - 1j * 0
Air_incident_angle = incident_angle

### hBN ###
hBN_thickness = hBN_thickness_value
hBN_epsilon_r = hBN_epsilon_r_value
hBN_mu_r = hBN_mu_r_value
hBN_complex_index = hBN_index_real_value - 1j * 0
hBN_incident_angle = RAT.snell(Air_complex_index, hBN_complex_index, Air_incident_angle)

### Black Phosphorous, BP ###
BP_thickness = BP_thickness_value
BP_epsilon_r = BP_epsilon_r_value
BP_mu_r = BP_mu_r_value

################################################################################
# Journal Name::    Physical Review Letters
# Paper Title::     Anisotropic Particle-Hole Excitations in Black Phosphorus
# Authors::         R. Schuster, J. Trinckauf, C. Habenicht, M. Knupfer, and B. Büchner
# DOI::             https://doi.org/10.1103/PhysRevLett.115.026404
# Year::            2015
BP_epsilon1_ac_fileName = '../Data/import_data/BP_ac_epsilon1_NIR_VIS_Schuter.txt'
BP_wavelength_epsilon1_ac_nm, BP_epsilon1_ac_real_interpolate = RAT.load_epsilons_data(BP_epsilon1_ac_fileName, delimiter=', ', unit='eV', columnNumber='2')
BP_epsilon1_zz_fileName = '../Data/import_data/BP_zz_epsilon1_NIR_VIS_Schuter.txt'
BP_wavelength_epsilon1_zz_nm, BP_epsilon1_zz_real_interpolate = RAT.load_epsilons_data(BP_epsilon1_zz_fileName, delimiter=', ', unit='eV', columnNumber='2')
BP_epsilon2_ac_fileName = '../Data/import_data/BP_ac_epsilon2_NIR_VIS_Schuter.txt'
BP_wavelength_epsilon2_ac_nm, BP_epsilon2_ac_imag_interpolate = RAT.load_epsilons_data(BP_epsilon2_ac_fileName, delimiter=', ', unit='eV', columnNumber='2')
BP_epsilon2_zz_fileName = '../Data/import_data/BP_zz_epsilon2_NIR_VIS_Schuter.txt'
BP_wavelength_epsilon2_zz_nm, BP_epsilon2_zz_imag_interpolate = RAT.load_epsilons_data(BP_epsilon2_zz_fileName, delimiter=', ', unit='eV', columnNumber='2')
BP_epsilon1_ac = BP_epsilon1_ac_real_interpolate(wavelength)
BP_epsilon2_ac = BP_epsilon2_ac_imag_interpolate(wavelength)
BP_epsilon1_zz = BP_epsilon1_zz_real_interpolate(wavelength)
BP_epsilon2_zz = BP_epsilon2_zz_imag_interpolate(wavelength)
BP_complex_index_ac = RAT.epsilon_to_n(BP_epsilon1_ac, BP_epsilon2_ac)
BP_complex_index_zz = RAT.epsilon_to_n(BP_epsilon1_zz, BP_epsilon2_zz)
################################################################################
BP_incident_angle_ac = RAT.snell(hBN_complex_index, BP_complex_index_ac, hBN_incident_angle)
BP_incident_angle_zz = RAT.snell(hBN_complex_index, BP_complex_index_zz, hBN_incident_angle)

### Gold, Au ###
Au_thickness = Au_thickness_value
Au_epsilon_r = Au_epsilon_r_value
Au_mu_r = Au_mu_r_value
Au_fileName = '../Data/import_data/Au_n_k.txt'
wavelength_nm_Au, Au_index_real_interpolate, Au_index_imag_interpolate = RAT.loadData(Au_fileName, delimiter='\t', unit='nm', columnNumber='3')
Au_complex_index = Au_index_real_interpolate(wavelength) - 1j * Au_index_imag_interpolate(wavelength)
Au_incident_angle_ac = RAT.snell(BP_complex_index_ac, Au_complex_index, BP_incident_angle_ac)
Au_incident_angle_zz = RAT.snell(BP_complex_index_zz, Au_complex_index, BP_incident_angle_zz)
Au_incident_angle = RAT.snell(Air_complex_index, Au_complex_index, Air_incident_angle)

### SiO2 ###
SiO2_thickness = SiO2_thickness_value
SiO2_epsilon_r = SiO2_epsilon_r_value
SiO2_mu_r = SiO2_mu_r_value
SiO2_fileName = '../Data/import_data/SiO2_n.txt'
wavelength_nm_SiO2, SiO2_index_real_interpolate = RAT.loadData(SiO2_fileName, delimiter='\t', unit='eV', columnNumber='2')
SiO2_index_Imag = 0
SiO2_complex_index = SiO2_index_real_interpolate(wavelength) - 1j * SiO2_index_Imag
SiO2_incident_angle_ac = RAT.snell(Au_complex_index, SiO2_complex_index, Au_incident_angle_ac)
SiO2_incident_angle_zz = RAT.snell(Au_complex_index, SiO2_complex_index, Au_incident_angle_zz)
SiO2_incident_angle = RAT.snell(Au_complex_index, SiO2_complex_index, Au_incident_angle)

### Silicon, Si ###
Si_thickness = Si_thickness_value
Si_epsilon_r = Si_epsilon_r_value
Si_mu_r = Si_mu_r_value
Si_fileName = '../Data/import_data/Si_n_k.txt'
Si_wavelength_nm, Si_index_real_interpolate, Si_index_imag_interpolate = RAT.loadData(Si_fileName, delimiter='\t', unit='nm', columnNumber='3')
Si_complex_index = Si_index_real_interpolate(wavelength) - 1j * Si_index_imag_interpolate(wavelength)
Si_incident_angle_ac = RAT.snell(SiO2_complex_index, Si_complex_index, SiO2_incident_angle_ac)
Si_incident_angle_zz = RAT.snell(SiO2_complex_index, Si_complex_index, SiO2_incident_angle_zz)
Si_incident_angle = RAT.snell(SiO2_complex_index, Si_complex_index, SiO2_incident_angle)
################################################################################
################################################################################
def CalculateRAT_A(BP_thickness, hBN_thickness):
    #   Air #
    Air_eta_s, Air_eta_p, Air_M_s, Air_M_p = RAT.LayerMatrix(Air_thickness, Air_mu_r, Air_complex_index, Air_incident_angle, wavelength, position='0')
    #   hBN #
    hBN_eta_s, hBN_eta_p, hBN_M_s, hBN_M_p = RAT.LayerMatrix(hBN_thickness, hBN_mu_r, hBN_complex_index, hBN_incident_angle, wavelength, position='1')
    #   BP #
    BP_eta_ac_s, BP_eta_ac_p, BP_M_ac_s, BP_M_ac_p = RAT.LayerMatrix(BP_thickness, BP_mu_r, BP_complex_index_ac, BP_incident_angle_ac, wavelength, position='2')
    BP_eta_zz_s, BP_eta_zz_p, BP_M_zz_s, BP_M_zz_p = RAT.LayerMatrix(BP_thickness, BP_mu_r, BP_complex_index_zz, BP_incident_angle_zz, wavelength, position='2')
    #   Au #
    Au_eta_ac_s, Au_eta_ac_p, Au_M_ac_s, Au_M_ac_p = RAT.LayerMatrix(Au_thickness, Au_mu_r, Au_complex_index, Au_incident_angle_ac, wavelength, position='3')
    Au_eta_zz_s, Au_eta_zz_p, Au_M_zz_s, Au_M_zz_p = RAT.LayerMatrix(Au_thickness, Au_mu_r, Au_complex_index, Au_incident_angle_zz, wavelength, position='3')
    Au_eta_s, Au_eta_p, Au_M_s, Au_M_p = RAT.LayerMatrix(Au_thickness, Au_mu_r, Au_complex_index, Au_incident_angle, wavelength, position='3')
    #   SiO2  #
    SiO2_eta_ac_s, SiO2_eta_ac_p, SiO2_M_ac_s, SiO2_M_ac_p = RAT.LayerMatrix(SiO2_thickness, SiO2_mu_r, SiO2_complex_index, SiO2_incident_angle_ac, wavelength, position='4')
    SiO2_eta_zz_s, SiO2_eta_zz_p, SiO2_M_zz_s, SiO2_M_zz_p = RAT.LayerMatrix(SiO2_thickness, SiO2_mu_r, SiO2_complex_index, SiO2_incident_angle_zz, wavelength, position='4')
    SiO2_eta_s, SiO2_eta_p, SiO2_M_s, SiO2_M_p = RAT.LayerMatrix(SiO2_thickness, SiO2_mu_r, SiO2_complex_index, SiO2_incident_angle, wavelength, position='4')
    #   Si  #
    Si_eta_ac_s, Si_eta_ac_p, Si_M_ac_s, Si_M_ac_p = RAT.LayerMatrix(Si_thickness, Si_mu_r, Si_complex_index, Si_incident_angle_ac, wavelength, position='last')
    Si_eta_zz_s, Si_eta_zz_p, Si_M_zz_s, Si_M_zz_p = RAT.LayerMatrix(Si_thickness, Si_mu_r, Si_complex_index, Si_incident_angle_zz, wavelength, position='last')
    Si_eta_s, Si_eta_p, Si_M_s, Si_M_p = RAT.LayerMatrix(Si_thickness, Si_mu_r, Si_complex_index, Si_incident_angle, wavelength, position='last')
    ################################################################################
    ################################################################################
    # s-polarized Light ::
    #   Air/hBN/BP_ac/Au/SiO2/Si
    totalMatrix_ac_s = RAT.TransferMatrix(Air_M_s, hBN_M_s, BP_M_ac_s, Au_M_ac_s, SiO2_M_ac_s, Si_M_ac_s)
    Rs_ac = real(RAT.Reflectance(Air_eta_s, totalMatrix_ac_s))
    As_ac = real(RAT.Absorbance(Air_eta_s, Si_eta_ac_s, totalMatrix_ac_s))
    Ts_ac = real(RAT.Transmittance(Air_eta_s, Si_eta_ac_s, totalMatrix_ac_s))
    #   Air/hBN/BP_zz/Au/SiO2/Si
    totalMatrix_zz_s = RAT.TransferMatrix(Air_M_s, hBN_M_s, BP_M_zz_s, Au_M_ac_s, SiO2_M_zz_s, Si_M_zz_s)
    Rs_zz = real(RAT.Reflectance(Air_eta_s, totalMatrix_zz_s))
    As_zz = real(RAT.Absorbance(Air_eta_s, Si_eta_zz_s, totalMatrix_zz_s))
    Ts_zz = real(RAT.Transmittance(Air_eta_s, Si_eta_zz_s, totalMatrix_zz_s))
    #   Air/hBN/BP_ac/Au/SiO2/Si
    totalMatrix_ac_p = RAT.TransferMatrix(Air_M_p, hBN_M_p, BP_M_ac_p, Au_M_ac_p, SiO2_M_ac_p, Si_M_ac_p)
    Rp_ac = real(RAT.Reflectance(Air_eta_p, totalMatrix_ac_p))
    Ap_ac = real(RAT.Absorbance(Air_eta_p, Si_eta_ac_p, totalMatrix_ac_p))
    Tp_ac = real(RAT.Transmittance(Air_eta_p, Si_eta_ac_p, totalMatrix_ac_p))
    #   Air/hBN/BP_zz/Au/SiO2/Si
    totalMatrix_zz_p = RAT.TransferMatrix(Air_M_p, hBN_M_p, BP_M_zz_p, Au_M_zz_p, SiO2_M_zz_s, Si_M_zz_p)
    Rp_zz = real(RAT.Reflectance(Air_eta_p, totalMatrix_zz_p))
    Ap_zz = real(RAT.Absorbance(Air_eta_p, Si_eta_zz_p, totalMatrix_zz_p))
    Tp_zz = real(RAT.Transmittance(Air_eta_p, Si_eta_zz_p, totalMatrix_zz_p))
    ################################################################################
    ################################################################################
    Rs = ((Rs_ac + Rs_zz) / 2)
    As = ((As_ac + As_zz) / 2)
    Ts = ((Ts_ac + Ts_zz) / 2)
    ################################################################################
    ################################################################################
    Rp = ((Rp_ac + Rp_zz) / 2)
    Ap = ((Ap_ac + Ap_zz) / 2)
    Tp = ((Tp_ac + Tp_zz) / 2)
    ################################################################################
    ################################################################################
    R_ac = ((Rs_ac + Rp_ac) / 2)
    A_ac = ((As_ac + Ap_ac) / 2)
    T_ac = ((Ts_ac + Tp_ac) / 2)
    ################################################################################
    ################################################################################
    R_zz = ((Rs_zz + Rp_zz) / 2)
    A_zz = ((As_zz + Ap_zz) / 2)
    T_zz = ((Ts_zz + Tp_zz) / 2)
    ################################################################################
    ################################################################################
    R = ((R_ac + R_zz) / 2)
    A = ((A_ac + A_zz) / 2)
    T = ((T_ac + T_zz) / 2)

    return A

################################################################################
################################################################################
BP_thickness = 40
hBN_thickness = 150
A = CalculateRAT_A(BP_thickness, hBN_thickness)
print("For BP thickness " + str(BP_thickness) + " nm and hBN thickness " + str(hBN_thickness) + " nm \n Absorbance for unpolarized light at " + str(wavelength) + " nm = ", A)
