import numpy as np
import os
import matplotlib.pyplot as plt
source_file_path = os.getcwd() + '\\source_files\\100_frame_test\\'
noisy_img = np.load(source_file_path + 'noisy_img.npy')
bg_noisy_img = np.load(source_file_path + 'bg_noisy_img.npy')

print(noisy_img.max())

obj_y = 3000-2801
obj_x = 3000-2801
f, ax = plt.subplots(1,2)
ax[0].plot(noisy_img[0,int(2801/2),int(2801/2)-50:int(2801/2)+50])
ax[0].set_title('Obj img')
ax[1].plot(bg_noisy_img[0,int(obj_y/2),int(obj_x/2)-50:int(obj_x/2)+50])
ax[1].set_title('Background img')
plt.show()

# noisy_img
# ## Calculate Step Response of the Edge of the Moon
# diff_img = -(noisy_img[0,0:int((si/2)-1),int(si/4)]-noisy_img[0,1:int(si/2),int(si/4)])
# xx = np.argmax(diff_img)
# diff_img_trimmed = diff_img[:xx]
# # oned_imp = []
# # for i in range(len(xx)):
# #     diff_img_trimmed = diff_img[i,:xx[i]]
# #     oned_imp.append((diff_img_trimmed-np.mean(diff_img_trimmed))/np.std(diff_img_trimmed))
# oned_imp = (diff_img_trimmed-np.mean(diff_img_trimmed))/np.std(diff_img_trimmed)
# sys_otf = np.multiply(tele_otf,detector_otf)
#
# ##Estimate PSF
# num_ro = 20
# corr_vals = np.zeros(num_ro)
# for i in range(num_ro):
#     ro_temp = i*.001+.001
#     r1 = D/2
#     dx = 4*r1/si
#     atmosphere_otf_temp = make_long_otf(r1,dx,si,ro_temp)
#     total_otf_temp = np.multiply(sys_otf,atmosphere_otf_temp)
#     total_psf_temp = fftshift(ifft2(total_otf_temp))
#     samp_psf = total_psf_temp[::downscale_factor, ::downscale_factor]
#     slice = samp_psf[int(si/4),int(si/4)-xx:int(si/4)].real
#     imp_est = (slice-np.mean(slice))/np.std(slice)
#     corr_vals[i] = np.mean(np.multiply(oned_imp,imp_est.real))
#
#
#
# ro_est = np.argmax(corr_vals)/1e3+0.001
# atmosphere_otf_est = make_long_otf(r1,dx,si,ro_est)
# total_otf_est = np.multiply(sys_otf,atmosphere_otf_est)
# total_psf_est = fftshift((ifft2(total_otf_est)))
# detector_psf_est = total_psf_est[::downscale_factor,::downscale_factor]
# detector_otf_est = fft2(fftshift(detector_psf_est))
#
# print('r0 est: '+str(ro_est))
# print('r0 actual = '+str(ro))
# print('error = '+str(ro-ro_est))
#
# ##Wiener Filter
# NSR = 0.01
# ROTF = np.conjugate(detector_otf_est)/(abs(detector_otf_est)**2+NSR)
# img_WF=real(ifft2(fft2(noisy_img)*ROTF))
# bg_img_WF=real(ifft2(fft2(bg_noisy_img)*ROTF))
# coronagraph_img_WF=real(ifft2(fft2(coronagraph_noisy_img)*ROTF))
# bg_coronagraph_img_WF=real(ifft2(fft2(bg_coronagraph_noisy_img)*ROTF))
#
#
# ## Point Detector ##TODO: take median from top row since psf is almost 50:50
# patch = img_WF[:,int(obj_y/downscale_factor)-25:int(obj_y/downscale_factor)+25,
#         int(obj_x/downscale_factor)-25:int(obj_x/downscale_factor)+25]
# B = np.median(patch,(1,2))
# sigma = np.std(patch,(1,2))
# D = patch[:,25,25]
# gamma_base=(D-B)/sigma
#
# result = (np.sum(np.ones(len(gamma_base[gamma_base>6])))/num_frames)*100
# print('Base Detector is '+str(result)+'% accurate')
#
# ##Point detector w/ coronagraph
# patch = coronagraph_img_WF[:,int(obj_y/downscale_factor)-25:int(obj_y/downscale_factor)+25,
#         int(obj_x/downscale_factor)-25:int(obj_x/downscale_factor)+25]
# B = np.median(patch,(1,2))
# sigma = np.std(patch,(1,2))
# D = patch[:,25,25]
# coronagraph_gamma=(D-B)/sigma
#
# result = (np.sum(np.ones(len(coronagraph_gamma[coronagraph_gamma>6])))/num_frames)*100
# print('Coronagraph Detector is '+str(result)+'% accurate')
#
# ## False alarm rate stats no coronagraph ##TODO: calculate gammas of patches with no object and make cdf
# patch = bg_img_WF[:,int(obj_y/downscale_factor)-25:int(obj_y/downscale_factor)+25,
#         int(obj_x/downscale_factor)-25:int(obj_x/downscale_factor)+25]
# B = np.median(patch,(1,2))
# sigma = np.std(patch,(1,2))
# D = patch[:,25,25]
# bg_gamma_base=(D-B)/sigma
#
# mean = np.mean(bg_gamma_base)
# sigma = np.std(bg_gamma_base)
# background_stats = norm.cdf(6,loc=mean, scale=sigma)
# false_alarm_rate = 1-background_stats
#
# print('Background Mean = ' +str(mean))
# print('Background Variance = ' +str(sigma))
# print('False alarm rate for base detector: ' + str(false_alarm_rate*100)+'%')
#
# ## False alarm rate with coronagraph
# patch = bg_coronagraph_img_WF[:,int(obj_y/downscale_factor)-25:int(obj_y/downscale_factor)+25,
#         int(obj_x/downscale_factor)-25:int(obj_x/downscale_factor)+25]
# B = np.median(patch,(1,2))
# sigma = np.std(patch,(1,2))
# D = patch[:,25,25]
# bg_coronagraph_gamma_base=(D-B)/sigma
#
# mean = np.mean(bg_coronagraph_gamma_base)
# sigma = np.std(bg_coronagraph_gamma_base)
# background_stats = norm.cdf(6,loc=mean, scale=sigma)
# false_alarm_rate = 1-background_stats