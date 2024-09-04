"""
Author: 2nd Lt. Patrick Carattini
Created: 4 Feb 2023
Title: gbs_otf
"""

import numpy as np
from scipy.fft import fft2, ifft2, fftshift
from numpy import squeeze, ceil, real
import matplotlib.pyplot as plt
import os
from simulate_coronagraph import *

# def simulate_moon(D,F,lam,dt, gen_dim_obj):
#
#     path = os.getcwd() + '/source_files/moon_img.txt'
#     opo9914d = np.genfromtxt(path, delimiter=",").astype('float')
#     img = np.zeros((1200, 1200, 3))
#     img[:, :1200, 0] = opo9914d[:, :1200]
#     img[:, :1200, 1] = opo9914d[:, 1200:2400]
#     img[:, :1200, 2] = opo9914d[:, 2400:3600]
#
#     dpix = lam * F / (2 * D)
#     dtheta = lam / (2 * D)
#     Moon_dist = 384400000.0                   # meters
#     Moon_diameter = 3474800.0                 # meters
#     Moon_pixels = Moon_diameter / (Moon_dist * dtheta)
#
#     pixels = round(Moon_pixels)
#     f_interp_moon = np.zeros([pixels, pixels]).astype('complex')
#     fmoon = fftshift(fft2(squeeze(img[:,:, 2])))
#     half_pix = ceil(pixels / 2).astype('int')
#     f_interp_moon[half_pix - 600: half_pix + 600,
#                     half_pix - 600: half_pix + 600] = fmoon
#     # f_interp_moon[:600,:600] = fmoon[:600,:600]
#     moon = real(ifft2(fftshift(f_interp_moon)))
#     Source_img = np.multiply(np.ones([3000, 3000]), moon[1, 1])
#     Source_img[1501 - half_pix + 1: 1501 + half_pix,
#                 1501 - half_pix + 1: 1501 + half_pix] = moon
#
#     # Calculate energy from the moon and
#     Intensity = 1000.0                          # w / m ^ 2 power per unit area hitting the moon
#     h = 6.62607015e-34                          # plancks constant
#     c = 3.0e8                                   # speed of light in meters
#     v = c / lam                                 # frequency of light
#     moon_reflectivity = 0.10                    # moon's reflectivity is 10%
#     photons_moon = (Intensity * ((dtheta*Moon_dist)**2) * dt * moon_reflectivity) / (h * v)
#     # energy = (photons / (4.0 * np.pi * Moon_dist ** 2.0)) * np.pi * (D / 2.0)**2.0
#     photons_telescope = (photons_moon*np.pi*(D/2)**2)/(2*np.pi*(Moon_dist)**2)
#
#     # Make Image reflect real energy values
#
#     # moon_max = np.max(np.max(Source_img))
#     moon_max = Source_img.max()
#
#     norm_moon = np.divide(Source_img, moon_max)
#
#     photons_img = np.multiply(norm_moon, photons_telescope)
#
#     if gen_dim_obj:
#         # Add dim objects
#         object_size = 10        # area of object in m^2
#         obj_reflectivity = 1
#         obj_photons = (Intensity * (object_size) * dt * obj_reflectivity) / (h * v)
#         obj_photons_telescope = (obj_photons*np.pi*(D/2)**2)/(2*np.pi*(Moon_dist)**2)
#
#         num_objects = 1500
#
#         images = np.zeros([num_objects,3000,3000])
#         obj_pos = np.arange(0,3000,2,dtype=int)
#
#         img_index = np.arange(0,num_objects,dtype=int)
#         images[img_index] = photons_img
#         images[img_index,1500,obj_pos] = obj_photons_telescope
#         # obj_pos= [0,999,1999,2999]
#
#         # img_index = np.arange(0,num_objects,dtype=int)
#         # for i in img_index:
#         #     images[i] = photons_img
#         #     images[i,1500,obj_pos[i]] = obj_photons_telescope
#
#
#     else:
#         images = photons_img
#
#     return images


def simulate_moon(D, F, lam, dt, background):
    path = os.getcwd() + '/source_files/moon_img.txt'
    opo9914d = np.genfromtxt(path, delimiter=",").astype('float')
    img = np.zeros((1200, 1200, 3))
    img[:, :1200, 0] = opo9914d[:, :1200]
    img[:, :1200, 1] = opo9914d[:, 1200:2400]
    img[:, :1200, 2] = opo9914d[:, 2400:3600]

    dpix = lam * F / (2 * D)
    dtheta = lam / (2 * D)
    Moon_dist = 384400000.0  # meters
    Moon_diameter = 3474800.0  # meters
    Moon_pixels = Moon_diameter / (Moon_dist * dtheta)

    pixels = round(Moon_pixels)
    f_interp_moon = np.zeros([pixels, pixels]).astype('complex')
    fmoon = fftshift(fft2(squeeze(img[:, :, 2])))
    half_pix = ceil(pixels / 2).astype('int')
    f_interp_moon[half_pix - 600: half_pix + 600,
    half_pix - 600: half_pix + 600] = fmoon
    moon_outline = make_pupil(1000, 0, 3000)
    moon = np.abs(ifft2(fftshift(f_interp_moon)))
    Source_img = np.zeros((3000, 3000))

    Source_img[1501 - half_pix + 1: 1501 + half_pix,
    1501 - half_pix + 1: 1501 + half_pix] = moon

    # Calculate energy from the moon and
    Intensity = 1000.0  # w / m ^ 2 power per unit area hitting the moon
    h = 6.62607015e-34  # plancks constant
    c = 3.0e8  # speed of light in meters
    v = c / lam  # frequency of light
    moon_reflectivity = 0.10  # moon's reflectivity is 10%
    photons = (D * D * Intensity * (Moon_dist * dtheta) * (Moon_dist * dtheta) * dt * moon_reflectivity) / (
                4 * Moon_dist * Moon_dist * (h * v))
    # energy = (photons / (4.0 * np.pi * Moon_dist ** 2.0)) * np.pi * (D / 2.0)**2.0

    # moon_max = np.max(np.max(Source_img))
    moon_max = Source_img.max()

    norm_moon = np.divide(Source_img, moon_max)

    photons_img = photons * np.multiply(norm_moon, moon_outline) + background

    # Add dim objects
    obj_reflectivity = 1
    obj_photons = (Intensity * dpix * dt * obj_reflectivity) / (h * v)
    # obj_photons = 5e20

    images = np.zeros([3000, 3000])
    images = photons_img

    return images

# photon_img = simulate_moon(0.07, 0.4, 610.0*10**-9,1)

# plt.imshow(photon_img)
# plt.savefig('photon_moon')

# # Telescope Parameters
# D = 0.07                        # diameter of telescope in meters
# lamb = 610*10**(-9)             # wavelength of ligth in meters
# f = 0.4                         # focal length in meters
# dt = 100.0e-3                   # CCD integration time

# photon_img = simulate_moon(D, f, lamb, dt, True)

# source_file_path = os.getcwd() + '/source_files/'

# np.save(source_file_path + 'photon_img_obj_full', photon_img)