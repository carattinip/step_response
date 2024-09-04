"""
Author: 2nd Lt. Patrick Carattini
Created: 6 May 2024
Title: 2024 Spring Project
"""

import numpy as np
from numpy import squeeze, ceil, real
from scipy.fft import fft2, ifft2, fftshift
import matplotlib.pyplot as plt
from matplotlib import cm
from pathlib import Path


# from skimage.util import random_noise
import os
from scipy.stats import norm


from make_otf2 import make_otf, make_pupil
from simulate_moon import simulate_moon
from make_long_otf2 import make_long_otf
from detector_blur import detector_blur
from simulate_coronagraph import *

from make_short_otf import *

# paths TODO: make OS agnostic
source_file_path = Path(os.getcwd()) / "source_files"
figures_path = Path(os.getcwd()) / "figures"


# TODO: check for existence of OTFs in the file system to automate generation of OTFs
## Flags to Generate or Pull From File
generate = False  # flag to generate data or load from memory
gen_objects = False  # flag to generate dim objects around the moon

## Flag for generating plots
gen_plots = True

# Telescope Parameters
D = 0.07  # diameter of telescope in meters
obs = 0  # obscuration diameter
lamb = 610 * 10 ** (-9)  # wavelength of length in meters
f = 0.4  # focal length in meters
si = 3000  # ccd pixel dimensions
scale = 1  # value at dc
phase = np.zeros([si, si])  # zero for non-abberated system
dt = 100.0e-3  # CCD integration time

# Atmosphere Parameters
z = 100 * 10**3  # Karman line ~ 100km
ro = 0.02  # seeing parameter
r1a = D / 2  # radius used for turbulence calculations


def plot_OTF(OTF, name):

    if gen_plots:
        # Plot Tele OTF
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        # Make data.
        fmax = (D - obs) / (2 * lamb * f)
        fs = fmax * 2
        df = fs / si
        X = np.arange(-fmax, fmax, df) / 1e3
        Y = X
        X, Y = np.meshgrid(X, Y)

        # Plot the surface.
        ax.set_xlabel("x Spatial Frequency (kilocycles/m)")
        ax.set_ylabel("y Spatial Frequency (kilocycles/m)")
        ax.set_zlabel("Magnitude")
        surf = ax.plot_surface(X, Y, abs(fftshift(OTF)), cmap=cm.winter)
        plt.title(name + " OTF")

        fig_name = name + "_OTF"
        plt.savefig(figures_path / fig_name)
        plt.close()
    else:
        pass


def plot_img(img, name):
    if gen_plots:
        plt.imshow(img)
        # plt.xlabel('Meters')
        # plt.ylabel('Meters')
        plt.title(name + " Image")
        plt.tick_params(
            left=False, right=False, labelleft=False, labelbottom=False, bottom=False
        )

        fig_name = name + "_img"
        plt.savefig(figures_path / fig_name)
        plt.close()
    else:
        pass


class my_model:

    def gen_tele_OTF(self):
        tele_pupil = make_pupil(self.r1, self.r2, self.si)
        [tele_OTF, tele_psf] = make_otf(self.scale, tele_pupil)

        plot_OTF(tele_OTF, "Telescope")

        return tele_OTF

    def gen_detector_OTF(self):
        detector_OTF = detector_blur(2, 2, self.si)
        plot_OTF(detector_OTF, "Detector")
        return detector_OTF

    def gen_atm_OTF(self):
        atm_OTF = make_long_otf(self.r1a, self.dx, self.si, self.ro)
        plot_OTF(atm_OTF, "Atmosphere")
        return atm_OTF

    def gen_short_OTF(self):
        short_OTF = make_short_otf2(self.r1a, self.dx, self.si, self.ro)
        plot_OTF(short_OTF, "Short")
        return short_OTF

    def gen_sys_OTF(self):
        sys_OTF = self.detector_OTF * self.tele_OTF
        plot_OTF(sys_OTF, "System")
        return sys_OTF

    def gen_turb_OTF(self):
        turb_otf = self.atm_OTF * self.tele_OTF
        plot_OTF(turb_otf, "Turbulent")
        return turb_otf

    def gen_total_OTF(self):
        tot_OTF = self.turb_OTF * self.detector_OTF
        plot_OTF(tot_OTF, "Total")
        return tot_OTF

    def gen_moon(self):
        photon_img = simulate_moon(self.D, self.f, self.lamb, self.dt, 100)
        plot_img(photon_img, "Photon")
        return photon_img

    ### Moon Sim
    def gen_output_img(self):
        output_img = real(ifft2(np.multiply(self.total_OTF, fft2(self.photon_img))))
        plot_img(output_img, "Output")
        return output_img

    def down_sample_img(self, downscale_factor):
        down_sample_img = self.output_img[::downscale_factor, ::downscale_factor]
        plot_img(down_sample_img, "Down Sampled")
        return down_sample_img

    def gen_noisy_img(self, readout_std):
        noisy_img = np.random.normal(
            np.random.poisson(self.down_sample_image), readout_std
        )

        plot_img(noisy_img, "Noisy")
        return noisy_img

    ### Star Sim
    def gen_star_out_img(self, readout_std):
        star = np.ones([5, self.si, self.si])
        star[:, int(si / 2), int(si / 2)] = 10e3
        tot_short_OTF = self.sys_OTF * self.short_OTF
        star = real(ifft2(np.multiply(tot_short_OTF, fft2(star))))
        noisy_star = np.random.normal(
            np.random.poisson(
                star[:, :: self.down_sample_factor, :: self.down_sample_factor]
            ),
            readout_std,
        )
        plot_img(noisy_star[0], "Noisy Star")
        return noisy_star

    # def gen_tele_otf(self):
    #     tele_OTF = np.multiply(self.optic_OTF, self.detector_OTF)
    #     plot_OTF(tele_OTF, "Tele")
    #     return tele_OTF

    def __init__(
        self,
        Diameter,
        Obscuration,
        lamb,
        focal_length,
        si,
        Phase,
        dt,
        scale,
        z,
        ro,
        r1a,
    ):
        self.D = Diameter
        self.obs = Obscuration
        self.lamb = lamb
        self.f = focal_length
        self.si = si
        self.phase = Phase
        self.dt = dt
        self.scale = scale
        self.pixel_per_meter = self.si / (2 * self.D)  # array is 5x5 meter
        self.dm = 1 / self.pixel_per_meter  # distance between pixels in meters
        self.r1 = (D / 2) * self.pixel_per_meter  # radius of telescope in pixels
        self.r2 = (obs / 2) * self.pixel_per_meter  # radius of obscuration in pixels

        # Atmosphere params
        self.z = z  # Karman Line
        self.ro = ro  # seeing parameter
        self.r1a = r1a  # radius in m for turbulence calcs
        self.dx = 4 * self.r1a / self.si  # x steps for turbulence calculations

        ## image params
        self.down_sample_factor = 2
        self.std_readOut = 4

        self.tele_OTF = self.gen_tele_OTF()
        self.detector_OTF = self.gen_detector_OTF()
        self.sys_OTF = self.gen_sys_OTF()

        self.atm_OTF = self.gen_atm_OTF()
        self.short_OTF = self.gen_short_OTF()
        self.turb_OTF = self.gen_turb_OTF()

        self.total_OTF = self.gen_total_OTF()

        self.photon_img = self.gen_moon()
        self.output_img = self.gen_output_img()
        self.down_sample_image = self.down_sample_img(self.down_sample_factor)
        self.noisy_img = self.gen_noisy_img(self.std_readOut)

        self.star_sim = self.gen_star_out_img(self.std_readOut)
        # self.tele_OTF = self.gen_tele_otf()


model = my_model(D, obs, lamb, f, si, phase, dt, scale, z, ro, r1a)

sys_otf = model.sys_OTF
patch = model.star_sim[:, 740:760, 740:760].mean(axis=0)
## Calculate Step Response of the Edge of the star
width = int(len(patch))
diff_img = -(
    patch[0 : int(width - 1), int(width / 2)] - patch[1 : int(width), int(width / 2)]
)
xx = np.argmax(diff_img)
diff_img_trimmed = diff_img[:xx]
oned_imp = (diff_img_trimmed - np.mean(diff_img_trimmed)) / np.std(diff_img_trimmed)

downscale_factor = model.down_sample_factor
##Estimate PSF
num_ro = 20
corr_vals = np.zeros(num_ro)
for i in range(num_ro):
    ro_temp = i * 0.001 + 0.001
    r1 = D / 2
    dx = 4 * r1 / si
    atmosphere_otf_temp = make_short_otf2(r1, dx, si, ro_temp)
    total_otf_temp = np.multiply(sys_otf, atmosphere_otf_temp)
    total_psf_temp = fftshift(ifft2(total_otf_temp))
    samp_psf = total_psf_temp[::downscale_factor, ::downscale_factor]
    # test_patch = samp_psf[740:760, 740:760]
    # width = len(test_patch)
    samp = samp_psf[int(si / 4), int(si / 4) - xx : int(si / 4)].real
    imp_est = (samp - np.mean(samp)) / np.std(samp)
    corr_vals[i] = np.mean(np.multiply(oned_imp, imp_est.real))

ro_est = np.argmax(corr_vals) / 1e3 + 0.001
print(ro_est)

lim = int(0)

plot_img(
    model.star_sim[0, 740:760, 740:760],
    "cropped_star",
)
# from matplotlib import cm

# # Plot Tele OTF
# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# # Make data.
# fmax = (D - obs) / (2 * lamb * f)
# fs = fmax * 2
# df = fs / si
# X = np.arange(-fmax, fmax, df) / 1e3
# Y = X
# X, Y = np.meshgrid(X, Y)


# # Plot the surface.
# ax.set_xlabel("x Spatial Frequency (kilocycles/m)")
# ax.set_ylabel("y Spatial Frequency (kilocycles/m)")
# ax.set_zlabel("Magnitude")
# ax.set_title("Telescope OTF")
# surf = ax.plot_surface(X, Y, abs(fftshift(tele.optic_OTF)), cmap=cm.winter)

# plt.show()
