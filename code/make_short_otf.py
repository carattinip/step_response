def make_short_otf2(r1, dx, si, ro):

    import numpy as np
    from scipy.fft import fftshift

    pupilx = np.zeros((si, si))

    otf = np.zeros((si, si))

    if 2 * np.floor(si / 2) == si:

        mi = int(np.floor(si / 2))

        pupilx = np.zeros([si, si])

        for i in range(0, si - 1):

            pupilx[i] = range(-mi, mi)

    if 2 * np.floor(si / 2) != si:

        mi = int(np.floor(si / 2))

        pupilx = np.zeros([si, si])

        for i in range(0, si - 1):

            pupilx[i] = range(-mi, mi + 1)

    pupily = np.transpose(pupilx)

    dist2 = np.multiply(pupilx, pupilx) + np.multiply(pupily, pupily)

    dist = np.sqrt(dist2)

    temp = np.power((dx * dist / ro), (5 / 3))

    temp3 = np.ones((si, si)) - dist / (2 * r1 / dx) + 0.0000001

    binmap = temp3 > 0

    temp4 = np.power(np.multiply(temp3, binmap), (1 / 3))

    temp2 = -3.44 * np.multiply(temp, temp4)

    otf = np.exp(temp2)

    otf2 = fftshift(np.multiply(otf, binmap))

    return otf2


# import numpy as np

# si = 50
# ## for temp testing sys_otf is 1s
# sys_otf = np.ones([si, si])
# downscale_factor = 1


# def imp_fit(input):
#     import numpy as np
#     from scipy.fft import fft, fft2, fftshift

#     si = 50
#     ## Calculate Step Response of the Edge of the star
#     diff_img = -(
#         input[0, 0 : int((si / 2) - 1), int(si / 4)]
#         - input[0, 1 : int(si / 2), int(si / 4)]
#     )
#     xx = np.argmax(diff_img)
#     diff_img_trimmed = diff_img[:xx]
#     oned_imp = (diff_img_trimmed - np.mean(diff_img_trimmed)) / np.std(diff_img_trimmed)

#     ##Estimate PSF
#     num_ro = 20
#     corr_vals = np.zeros(num_ro)
#     for i in range(num_ro):
#         ro_temp = i * 0.001 + 0.001
#         r1 = D / 2
#         dx = 4 * r1 / si
#         atmosphere_otf_temp = make_short_otf2(r1, dx, si, ro_temp)
#         total_otf_temp = np.multiply(sys_otf, atmosphere_otf_temp)
#         total_psf_temp = fftshift(ifft2(total_otf_temp))
#         samp_psf = total_psf_temp
#         samp = samp_psf[int(si / 2), int(si / 2) - xx : int(si / 2)].real

#         imp_est = (samp - np.mean(samp)) / np.std(samp)
#         corr_vals[i] = np.mean(np.multiply(oned_imp, imp_est.real))

#     ro_est = np.argmax(corr_vals) / 1e3 + 0.001

#     return ro_est


# import matplotlib.pyplot as plt

# from scipy.fft import fft2, ifft2, fftshift
# from matplotlib import cm
# from numpy import real
# import numpy as np


# # Telescope Parameters
# D = 0.07  # diameter of telescope in meters
# obs = 0  # obscuration diameter
# lamb = 610 * 10 ** (-9)  # wavelength of length in meters
# f = 0.4  # focal length in meters

# scale = 1  # value at dc
# phase = np.zeros([si, si])  # zero for non-abberated system
# dt = 100.0e-3  # CCD integration time

# # Atmosphere Parameters
# z = 100 * 10**3  # Karman line ~ 100km
# ro = 0.02  # seeing parameter
# r1a = D / 2  # radius used for turbulence calculations
# dx = 4 * r1a / si


# short_otf = [make_short_otf2(r1a, dx, si, ro)] * 10

# obj = np.ones([si, si])
# obj[int(si / 2), int(si / 2)] = 5e3


# sim_obj = real(ifft2((short_otf * fft2(obj))))

# # Plot sim PSF

# # # plt.imshow(real(tele_psf), extent=[0,(si/pixle_per_meter), 0, (si/pixle_per_meter)])
# plt.imshow(sim_obj.mean(axis=0))
# # plt.xlim(1500, 1500 + 15)
# # plt.ylim(150015, 1500 - 15)
# plt.xlabel("x (pixles)")
# plt.ylabel("y (pixles)")
# plt.title("Sim Obj PSF")
# plt.show()


# # Plot reconstructed PSF

# ro_est = imp_fit(sim_obj)
# print(f"Estimated r_o = {ro_est:.2f}")

# # # # Plot Tele OTF
# # # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# # # # Make data.
# # # fmax = (D - obs) / (2 * lamb * f)
# # # fs = fmax * 2
# # # df = fs / si
# # # X = np.arange(-fmax, fmax, df) / 1e3
# # # Y = X
# # # X, Y = np.meshgrid(X, Y)

# # # # Plot the surface.
# # # ax.set_xlabel("x Spatial Frequency (kilocycles/m)")
# # # ax.set_ylabel("y Spatial Frequency (kilocycles/m)")
# # # ax.set_zlabel("Magnitude")
# # # surf = ax.plot_surface(X, Y, abs(fftshift(short_otf)), cmap=cm.winter)
# # # plt.title("Short OTF")

# # # fig_name = "short_OTF"
# # # plt.show()
