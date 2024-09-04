"""
Author: 2nd Lt. Patrick Carattini
Created: 6 May 2024
Title: 2024 Spring Project
"""

import numpy as np
from numpy import squeeze, ceil, real
from scipy.fft import fft2, ifft2, fftshift
import matplotlib.pyplot as plt

# from skimage.util import random_noise
import os
from scipy.stats import norm


from make_otf2 import make_otf, make_pupil
from simulate_moon import simulate_moon
from make_long_otf2 import make_long_otf
from detector_blur import detector_blur
from simulate_coronagraph import *

from make_short_otf import *

# TODO: check for existince of otfs in the file system to automate generation of OTFs

# Telescope Parameters
# D = 0.07                      # diameter of telescope in meters
D = 0.07
obs = 0  # obscuration diameter
lamb = 610 * 10 ** (-9)  # wavelength of ligth in meters
f = 0.4  # focal length in meters
si = 3000  # ccd pixel dimensions
pixle_per_meter = si / (2 * D)  # array is 5x5 meter
dm = 1 / pixle_per_meter  # distance between pixels in meters
r1 = (D / 2) * pixle_per_meter  # radius of telescope in pixels
r2 = (obs / 2) * pixle_per_meter  # radius of obscuration in pixels
scale = 1  # value at dc
phase = np.zeros([si, si])  # zero for non-abberated system
dt = 100.0e-3  # CCD integration time

# Atmosphere Parameters
z = 100 * 10**3  # Karman line ~ 100km
ro = 0.02  # seeing parameter
r1a = D / 2  # radius used for turbulence calculations
dx = 4 * r1a / si  # x steps for turbulence calculations

# paths TODO: make OS agnostic
source_file_path = os.getcwd() + "\\source_files\\100_frame_test\\"
figures_path = os.getcwd() + "\\figures\\"


generate = False  # flag to generate data or load from memory
gen_objects = False  # flag to generate dim objects around the moon

# Model Telescope
if generate or not (os.path.isfile(source_file_path + "tele_pupil.npy")):
    tele_pupil = make_pupil(r1, r2, si)
    np.save(source_file_path + "tele_pupil", tele_pupil)
else:
    tele_pupil = np.load(source_file_path + "tele_pupil.npy")

if generate or (
    (not (os.path.isfile(source_file_path + "tele_otf.npy")))
    or (not (os.path.isfile(source_file_path + "tele_psf.npy")))
):
    [tele_otf, tele_psf] = make_otf(scale, tele_pupil)
    np.save(source_file_path + "tele_otf", tele_otf)
    np.save(source_file_path + "tele_psf", tele_psf)
else:
    tele_psf = np.load(source_file_path + "tele_psf.npy")
    tele_otf = np.load(source_file_path + "tele_otf.npy")

# Model Atmosphere
if generate or (not (os.path.isfile(source_file_path + "atmosphere_otf.npy"))):
    atmosphere_otf = make_long_otf(r1a, dm, si, ro)
    np.save(source_file_path + "atmosphere_otf", atmosphere_otf)
else:
    atmosphere_otf = np.load(source_file_path + "atmosphere_otf.npy")

# Model Telescope + Atmosphere
if generate or (not (os.path.isfile(source_file_path + "turbulent_otf.npy"))):
    turbulent_otf = np.multiply(tele_otf, atmosphere_otf)
    np.save(source_file_path + "turbulent_otf", turbulent_otf)
else:
    turbulent_otf = np.load(source_file_path + "turbulent_otf.npy")

if generate or (not (os.path.isfile(source_file_path + "turbulent_psf.npy"))):
    turbulent_psf = ifft2(fftshift(turbulent_otf))
    np.save(source_file_path + "turbulent_psf", turbulent_psf)
else:
    turbulent_psf = np.load(source_file_path + "turbulent_psf.npy")

# Detector model ( adds blur so the niquist image is represented at 1/2 niquist)
if generate or (not (os.path.isfile(source_file_path + "detector_otf.npy"))):
    detector_otf = detector_blur(2, 2, si)
    np.save(source_file_path + "detector_otf", detector_otf)
else:
    detector_otf = np.load(source_file_path + "detector_otf.npy")

# Telescope + Atmosphere + Detector Model
if generate or (not (os.path.isfile(source_file_path + "total_otf.npy"))):
    total_otf = np.multiply(turbulent_otf, detector_otf)
    np.save(source_file_path + "total_otf", total_otf)
else:
    total_otf = np.load(source_file_path + "total_otf.npy")

if generate or (not (os.path.isfile(source_file_path + "total_psf.npy"))):
    total_psf = ifft2(fftshift(total_otf))
    np.save(source_file_path + "total_psf", total_psf)
else:
    total_psf = np.load(source_file_path + "total_psf.npy")


## OBJ location
obj_intensity = 1e4
obj_y = 2801
obj_x = 2801
# Simulate Moon
if generate or (not (os.path.isfile(source_file_path + "photon_img.npy"))):
    photon_img = simulate_moon(D, f, lamb, dt, 100)

    ## save copy for background statistics
    background_img = photon_img

    ## Add Dim Object
    photon_img[obj_y, obj_x] = obj_intensity

    np.save(source_file_path + "photon_img", photon_img)
    np.save(source_file_path + "background_img", background_img)
else:
    photon_img = np.load(source_file_path + "photon_img.npy")
    background_img = np.load(source_file_path + "background_img.npy")


# Model Coronagraph
sys_otf = np.multiply(tele_otf, detector_otf)
if generate or (
    (not (os.path.isfile(source_file_path + "coronagraph_input.npy")))
    or (not (os.path.isfile(source_file_path + "coronagraph.npy")))
    or (not (os.path.isfile(source_file_path + "coronagraph_output.npy")))
    or (not (os.path.isfile(source_file_path + "coronagraph_output_img.npy")))
    or (not (os.path.isfile(source_file_path + "bg_coronagraph_input.npy")))
    or (not (os.path.isfile(source_file_path + "bg_coronagraph_output.npy")))
    or (not (os.path.isfile(source_file_path + "bg_coronagraph_output_img.npy")))
):

    ## obj img
    coronagraph_input = real(ifft2(np.multiply(turbulent_otf, fft2(photon_img))))
    coronagraph = make_coronagraph(1100, si, 0, 0)
    coronagraph_output = np.multiply(coronagraph, coronagraph_input)
    coronagraph_output_img = real(ifft2(np.multiply(sys_otf, fft2(coronagraph_output))))

    ## save obj corona files
    np.save(source_file_path + "coronagraph_input", coronagraph_input)
    np.save(source_file_path + "coronagraph", coronagraph)
    np.save(source_file_path + "coronagraph_output", coronagraph_output)
    np.save(source_file_path + "coronagraph_output_img", coronagraph_output_img)

    ## background img
    bg_coronagraph_input = real(ifft2(np.multiply(turbulent_otf, fft2(background_img))))
    # coronagraph = make_coronagraph(1100, si,0,0)
    bg_coronagraph_output = np.multiply(coronagraph, bg_coronagraph_input)
    bg_coronagraph_output_img = real(
        ifft2(np.multiply(total_otf, fft2(bg_coronagraph_output)))
    )

    ## save background corona files
    np.save(source_file_path + "bg_coronagraph_input", bg_coronagraph_input)
    np.save(source_file_path + "bg_coronagraph_output", bg_coronagraph_output)
    np.save(source_file_path + "bg_coronagraph_output_img", bg_coronagraph_output_img)


else:
    ## load obj corona imgs
    coronagraph_input = np.load(source_file_path + "coronagraph_input.npy")
    coronagraph = np.load(source_file_path + "coronagraph.npy")
    coronagraph_output = np.load(source_file_path + "coronagraph_output.npy")
    coronagraph_output_img = np.load(source_file_path + "coronagraph_output_img.npy")

    ## load background corona imgs
    bg_coronagraph_input = np.load(source_file_path + "bg_coronagraph_input.npy")
    bg_coronagraph_output = np.load(source_file_path + "bg_coronagraph_output.npy")
    bg_coronagraph_output_img = np.load(
        source_file_path + "bg_coronagraph_output_img.npy"
    )

## Model no coronagraph
if generate or (
    (not (os.path.isfile(source_file_path + "output_img.npy")))
    or (not (os.path.isfile(source_file_path + "bg_output_img.npy")))
):
    ## obj img
    output_img = real(ifft2(np.multiply(total_otf, fft2(photon_img))))
    np.save(source_file_path + "output_img", output_img)
    ## bg img
    bg_output_img = real(ifft2(np.multiply(total_otf, fft2(background_img))))
    np.save(source_file_path + "bg_output_img", bg_output_img)
else:
    output_img = np.load(source_file_path + "output_img.npy")
    bg_output_img = np.load(source_file_path + "bg_output_img.npy")

# Down Sample to match IRL detector size
downscale_factor = 2
if generate or (
    (not (os.path.isfile(source_file_path + "down_sample_img.npy")))
    or (not (os.path.isfile(source_file_path + "coronagraph_down_sample_img.npy")))
    or (not (os.path.isfile(source_file_path + "bg_down_sample_img.npy")))
    or (not (os.path.isfile(source_file_path + "bg_coronagraph_down_sample_img.npy")))
):
    if gen_objects:
        down_sample_img = output_img[:, ::downscale_factor, ::downscale_factor]
        coronagraph_down_sample_img = coronagraph_output_img[
            :, ::downscale_factor, ::downscale_factor
        ]

        bg_down_sample_img = bg_output_img[:, ::downscale_factor, ::downscale_factor]
        bg_coronagraph_down_sample_img = bg_coronagraph_output_img[
            :, ::downscale_factor, ::downscale_factor
        ]

    else:
        down_sample_img = output_img[::downscale_factor, ::downscale_factor]
        coronagraph_down_sample_img = coronagraph_output_img[
            ::downscale_factor, ::downscale_factor
        ]

        bg_down_sample_img = bg_output_img[::downscale_factor, ::downscale_factor]
        bg_coronagraph_down_sample_img = bg_coronagraph_output_img[
            ::downscale_factor, ::downscale_factor
        ]

    np.save(source_file_path + "down_sample_img", down_sample_img)
    np.save(
        source_file_path + "coronagraph_down_sample_img", coronagraph_down_sample_img
    )

    np.save(source_file_path + "bg_down_sample_img", bg_down_sample_img)
    np.save(
        source_file_path + "bg_coronagraph_down_sample_img",
        bg_coronagraph_down_sample_img,
    )

else:
    down_sample_img = np.load(source_file_path + "down_sample_img.npy")
    coronagraph_down_sample_img = np.load(
        source_file_path + "coronagraph_down_sample_img.npy"
    )
    bg_down_sample_img = np.load(source_file_path + "bg_down_sample_img.npy")
    bg_coronagraph_down_sample_img = np.load(
        source_file_path + "bg_coronagraph_down_sample_img.npy"
    )

# Add Poison & Readout Noise
num_frames = 100
readout_std = 4
if generate or (
    not (os.path.isfile(source_file_path + "noisy_img.npy"))
    or (not (os.path.isfile(source_file_path + "coronagraph_noisy_img.npy")))
    or (not (os.path.isfile(source_file_path + "bg_noisy_img.npy")))
    or (not (os.path.isfile(source_file_path + "bg_coronagraph_noisy_img.npy")))
):
    # noisy_img = np.random.poisson(down_sample_img) ## singe img gen
    noisy_img = np.random.normal(
        np.random.poisson(np.array([down_sample_img] * num_frames)), readout_std
    )  ## generate independant 100 frames
    coronagraph_noisy_img = np.random.normal(
        np.random.poisson(np.array([coronagraph_down_sample_img] * num_frames)),
        readout_std,
    )
    np.save(source_file_path + "noisy_img", noisy_img)
    np.save(source_file_path + "coronagraph_noisy_img", coronagraph_noisy_img)

    bg_noisy_img = np.random.normal(
        np.random.poisson(np.array([bg_down_sample_img] * num_frames)), readout_std
    )  ## generate independant 100 frames
    bg_coronagraph_noisy_img = np.random.normal(
        np.random.poisson(np.array([bg_coronagraph_down_sample_img] * num_frames)),
        readout_std,
    )
    np.save(source_file_path + "bg_noisy_img", bg_noisy_img)
    np.save(source_file_path + "bg_coronagraph_noisy_img", bg_coronagraph_noisy_img)

else:
    noisy_img = np.load(source_file_path + "noisy_img.npy")
    coronagraph_noisy_img = np.load(source_file_path + "coronagraph_noisy_img.npy")

    bg_noisy_img = np.load(source_file_path + "bg_noisy_img.npy")
    bg_coronagraph_noisy_img = np.load(
        source_file_path + "bg_coronagraph_noisy_img.npy"
    )


## Calculate Step Response of the Edge of the Moon
diff_img = -(
    noisy_img[0, 0 : int((si / 2) - 1), int(si / 4)]
    - noisy_img[0, 1 : int(si / 2), int(si / 4)]
)
xx = np.argmax(diff_img)
diff_img_trimmed = diff_img[:xx]
# oned_imp = []
# for i in range(len(xx)):
#     diff_img_trimmed = diff_img[i,:xx[i]]
#     oned_imp.append((diff_img_trimmed-np.mean(diff_img_trimmed))/np.std(diff_img_trimmed))
oned_imp = (diff_img_trimmed - np.mean(diff_img_trimmed)) / np.std(diff_img_trimmed)


##Estimate PSF
num_ro = 20
corr_vals = np.zeros(num_ro)
for i in range(num_ro):
    ro_temp = i * 0.001 + 0.001
    r1 = D / 2
    dx = 4 * r1 / si
    atmosphere_otf_temp = make_long_otf(r1, dx, si, ro_temp)
    total_otf_temp = np.multiply(sys_otf, atmosphere_otf_temp)
    total_psf_temp = fftshift(ifft2(total_otf_temp))
    samp_psf = total_psf_temp[::downscale_factor, ::downscale_factor]
    slice = samp_psf[int(si / 4), int(si / 4) - xx : int(si / 4)].real
    imp_est = (slice - np.mean(slice)) / np.std(slice)
    corr_vals[i] = np.mean(np.multiply(oned_imp, imp_est.real))


coronagraph_sys_otf = np.multiply(tele_otf, sys_otf)
ro_est = np.argmax(corr_vals) / 1e3 + 0.001
atmosphere_otf_est = make_long_otf(r1, dx, si, ro_est)
total_otf_est = np.multiply(coronagraph_sys_otf, atmosphere_otf_est)
total_psf_est = fftshift((ifft2(total_otf_est)))
detector_psf_est = total_psf_est[::downscale_factor, ::downscale_factor]
detector_otf_est = fft2(fftshift(detector_psf_est))

print("r0 est: " + str(ro_est))
print("r0 actual = " + str(ro))
print("error = " + str(ro - ro_est))

##Wiener Filter
NSR = 0.01
ROTF = np.conjugate(detector_otf_est) / (abs(detector_otf_est) ** 2 + NSR)
img_WF = real(ifft2(fft2(noisy_img) * ROTF))
bg_img_WF = real(ifft2(fft2(bg_noisy_img) * ROTF))
coronagraph_img_WF = real(ifft2(fft2(coronagraph_noisy_img) * ROTF))
bg_coronagraph_img_WF = real(ifft2(fft2(bg_coronagraph_noisy_img) * ROTF))


## Point Detector ##TODO: take median from top row since psf is almost 50:50
patch = img_WF[
    :,
    int(obj_y / downscale_factor) - 25 : int(obj_y / downscale_factor) + 25,
    int(obj_x / downscale_factor) - 25 : int(obj_x / downscale_factor) + 25,
]
B = np.median(patch, (1, 2))
sigma = np.std(patch, (1, 2))
D = patch[:, 25, 25]
gamma_base = (D - B) / sigma

result = (np.sum(np.ones(len(gamma_base[gamma_base > 6]))) / num_frames) * 100
print("Base Detector is " + str(result) + "% accurate")

##Point detector w/ coronagraph
patch = coronagraph_img_WF[
    :,
    int(obj_y / downscale_factor) - 25 : int(obj_y / downscale_factor) + 25,
    int(obj_x / downscale_factor) - 25 : int(obj_x / downscale_factor) + 25,
]
B = np.median(patch, (1, 2))
sigma = np.std(patch, (1, 2))
D = patch[:, 25, 25]
coronagraph_gamma = (D - B) / sigma

result = (
    np.sum(np.ones(len(coronagraph_gamma[coronagraph_gamma > 6]))) / num_frames
) * 100
print("Coronagraph Detector is " + str(result) + "% accurate")

no_obj_y, no_obj_x = si - obj_y, si - obj_x

## False alarm rate stats no coronagraph ##TODO: calculate gammas of patches with no object and make cdf
patch = bg_img_WF[
    :,
    int(no_obj_y / downscale_factor) - 25 : int(no_obj_y / downscale_factor) + 25,
    int(no_obj_x / downscale_factor) - 25 : int(no_obj_x / downscale_factor) + 25,
]
B = np.median(patch, (1, 2))
sigma = np.std(patch, (1, 2))
D = patch[:, 25, 25]
bg_gamma_base = (D - B) / sigma

mean = np.mean(bg_gamma_base)
sigma = np.std(bg_gamma_base)
background_stats = norm.cdf(6, loc=mean, scale=sigma)
false_alarm_rate = 1 - background_stats

print("Background Mean = " + str(mean))
print("Background Variance = " + str(sigma))
print("False alarm rate for base detector: " + str(false_alarm_rate * 100) + "%")

## False alarm rate with coronagraph
patch = bg_coronagraph_img_WF[
    :,
    int(no_obj_y / downscale_factor) - 25 : int(no_obj_y / downscale_factor) + 25,
    int(no_obj_x / downscale_factor) - 25 : int(no_obj_x / downscale_factor) + 25,
]
B = np.median(patch, (1, 2))
sigma = np.std(patch, (1, 2))
D = patch[:, 25, 25]
bg_coronagraph_gamma_base = (D - B) / sigma

mean = np.mean(bg_coronagraph_gamma_base)
sigma = np.std(bg_coronagraph_gamma_base)
background_stats = norm.cdf(6, loc=mean, scale=sigma)
false_alarm_rate = 1 - background_stats

print("Background Mean = " + str(mean))
print("Background Variance = " + str(sigma))
print("False alarm rate for coronagraph: " + str(false_alarm_rate * 100) + "%")


## Plot Base Detector
f, ax = plt.subplots(2, 2)
ax[0, 0].imshow(noisy_img[0])
ax[0, 0].set_title("Original img")
ax[0, 0].tick_params(
    left=False, right=False, labelleft=False, labelbottom=False, bottom=False
)
ax[0, 1].imshow(img_WF[0])
ax[0, 1].set_title("Deconvolved img")
ax[0, 1].tick_params(
    left=False, right=False, labelleft=False, labelbottom=False, bottom=False
)

ax[1, 0].plot(noisy_img[0, int(obj_y / 2), int(obj_x / 2) - 50 : int(obj_x / 2) + 50])
ax[1, 0].set_title("Original img")

ax[1, 1].plot(img_WF[0, int(obj_y / 2), int(obj_x / 2) - 50 : int(obj_x / 2) + 50])
ax[1, 1].set_title("Deconv img")
plt.show()

## Plot Coronagraph Detector
f, ax = plt.subplots(2, 2)
ax[0, 0].imshow(coronagraph_noisy_img[0])
ax[0, 0].set_title("Original Coronagraph img")
ax[0, 0].tick_params(
    left=False, right=False, labelleft=False, labelbottom=False, bottom=False
)
ax[0, 1].imshow(coronagraph_img_WF[0])
ax[0, 1].set_title("Deconvolved Coronagraph img")
ax[0, 1].tick_params(
    left=False, right=False, labelleft=False, labelbottom=False, bottom=False
)

ax[1, 0].plot(
    coronagraph_noisy_img[0, int(obj_y / 2), int(obj_x / 2) - 50 : int(obj_x / 2) + 50]
)
ax[1, 0].set_title("Original Coronagraph img")

ax[1, 1].plot(
    coronagraph_img_WF[0, int(obj_y / 2), int(obj_x / 2) - 50 : int(obj_x / 2) + 50]
)
ax[1, 1].set_title("Deconv Coronagraph img")
plt.show()

# # Plots of Sim Chain
# f, axarr = plt.subplots(1,3)
# axarr[0].imshow(photon_img[1], norm='linear')
# axarr[0].set_xlabel('Simulated Moon')
# axarr[0].tick_params(left = False, right = False , labelleft = False ,
#                 labelbottom = False, bottom = False)
# axarr[1].imshow(output_img[1], norm='linear')
# axarr[1].set_xlabel('Telescope x Moon')
# axarr[1].tick_params(left = False, right = False , labelleft = False ,
#                 labelbottom = False, bottom = False)
# axarr[2].imshow(noisy_img[1], norm='linear')
# axarr[2].set_xlabel('Noisy Img')
# axarr[2].tick_params(left = False, right = False , labelleft = False ,
#                 labelbottom = False, bottom = False)
# plt.savefig('plots')

# # Plots of Objects
# f, axarr = plt.subplots(2,2)
# axarr[0,0].imshow(noisy_img[0])
# axarr[0,0].set_xlabel('left 1/4')
# axarr[0,0].tick_params(left = False, right = False , labelleft = False ,
#                 labelbottom = False, bottom = False)

# axarr[0,1].imshow(noisy_img[1])
# axarr[0,1].set_xlabel('left 2/4')
# axarr[0,1].tick_params(left = False, right = False , labelleft = False ,
#                 labelbottom = False, bottom = False)

# axarr[1,0].imshow(noisy_img[2])
# axarr[1,0].set_xlabel('right 2/4')
# axarr[1,0].tick_params(left = False, right = False , labelleft = False ,
#                 labelbottom = False, bottom = False)

# axarr[1,1].imshow(noisy_img[3])
# axarr[1,1].set_xlabel('right 4/4')
# axarr[1,1].tick_params(left = False, right = False , labelleft = False ,
#                 labelbottom = False, bottom = False)


# plt.savefig(figures_path + 'detector_model_imgs')
