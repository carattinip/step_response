# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 10:58:44 2024

@author: swordsman
"""

import numpy as np
import os
from scipy.fft import fft2, ifft2, fftshift
from numpy import squeeze, ceil, real
import matplotlib.pyplot as plt
def detector_blur(blurx,blury,si):
    import numpy as np
    from scipy.fft import fft2
    psf=np.zeros((si,si));
    for ii in range(0,blurx-1):
        for jj in range(0,blury-1):
            ycord=int(np.round(jj+si/2))
            xcord=int(np.round(ii+si/2))
            psf[ycord][xcord]=1
    otf=np.abs(fft2(psf))
    return(otf)
def make_long_otf2(r1,dx,si,ro):
    import numpy as np
    from scipy.fft import fftshift
    pupilx=np.zeros((si,si));
    otf=np.zeros((si,si));
    if(2*np.floor(si/2)==si):
        mi=int(np.floor(si/2));
        pupilx=np.zeros([si,si])
        
        for i in range(0,si-1):
            pupilx[i]=range(-mi,mi)
           
    if(2*np.floor(si/2)!=si):
         mi=int(np.floor(si/2));
         pupilx=np.zeros([si,si])
         for i in range(0,si-1):
             pupilx[i]=range(-mi,mi+1)
    pupily=np.transpose(pupilx)
    dist2=np.multiply(pupilx,pupilx)+np.multiply(pupily,pupily)
    dist=np.sqrt(dist2)
    temp=np.power((dx*dist/ro),5/3)
    temp2=-3.44*temp
    otf=np.exp(temp2)
    otf2=fftshift(otf)
    return(otf2)
def make_pupil(r1,r2,si):
    import matplotlib.pyplot as plt
    import numpy as np
    if(2*np.floor(si/2)==si):
        mi=int(np.floor(si/2));
        pupilx=np.zeros([si,si])
        for i in range(0,si-1):
            pupilx[i]=range(-mi,mi)
           
    if(2*np.floor(si/2)!=si):
         mi=int(np.floor(si/2));
         pupilx=np.zeros([si,si])
         for i in range(0,si-1):
             pupilx[i]=range(-mi,mi+1)
    pupily=np.transpose(pupilx)
    dist2=np.multiply(pupilx,pupilx)+np.multiply(pupily,pupily)
    dist=np.sqrt(dist2)
    pupil2=(dist<r1)
    pupil3=(dist>r2)
    pupil=np.multiply(pupil2.astype(int),pupil3.astype(int))
    return(pupil)
def make_coronagraph(r1,si,xs,ys):
    import numpy as np
    if(2*np.floor(si/2)==si):
        mi=int(np.floor(si/2));
        pupilx=np.zeros([si,si])
        for i in range(0,si-1):
            pupilx[i]=range(-mi,mi)
            
    if(2*np.floor(si/2)!=si):
         mi=int(np.floor(si/2));
         pupilx=np.zeros([si,si])
         for i in range(0,si-1):
             pupilx[i]=range(-mi,mi+1)
    pupily=np.transpose(pupilx)
    dist2=np.multiply(pupilx,pupilx)+np.multiply(pupily,pupily)
    dist=np.sqrt(dist2)
    pupil2=(dist>r1)
    pupil=np.multiply(pupil2.astype(int),np.ones((si,si)))
    pupil2=np.roll(pupil,ys,xs)
    return(pupil2)



def make_otf2(scale,cpupil):
    from scipy.fft import fft2
    import numpy as np
    
    psf=fft2(cpupil)
    psf=abs(psf)
    psf=np.multiply(psf, psf)
    spsf=np.sum(psf)
    norm_psf=scale*psf/spsf;
    otf=fft2(norm_psf)        
    return(otf,norm_psf)
def simulate_moon(D,F,lam,dt,background):

    path = os.getcwd() + '/source_files/moon_img.txt'
    opo9914d = np.genfromtxt(path, delimiter=",").astype('float')
    img = np.zeros((1200, 1200, 3))
    img[:, :1200, 0] = opo9914d[:, :1200]
    img[:, :1200, 1] = opo9914d[:, 1200:2400]
    img[:, :1200, 2] = opo9914d[:, 2400:3600]

    dpix = lam * F / (2 * D)
    dtheta = lam / (2 * D)
    Moon_dist = 384400000.0                   # meters
    Moon_diameter = 3474800.0                 # meters
    Moon_pixels = Moon_diameter / (Moon_dist * dtheta)
    
    pixels = round(Moon_pixels)
    f_interp_moon = np.zeros([pixels, pixels]).astype('complex')
    fmoon = fftshift(fft2(squeeze(img[:,:, 2])))
    half_pix = ceil(pixels / 2).astype('int')
    f_interp_moon[half_pix - 600: half_pix + 600,
                    half_pix - 600: half_pix + 600] = fmoon
    moon_outline=make_pupil(1000, 0, 3000)
    moon = np.abs(ifft2(fftshift(f_interp_moon)))
    Source_img = np.zeros((3000,3000))
    
    Source_img[1501 - half_pix + 1: 1501 + half_pix,
                1501 - half_pix + 1: 1501 + half_pix] = moon

    # Calculate energy from the moon and
    Intensity = 1000.0                          # w / m ^ 2 power per unit area hitting the moon
    h = 6.62607015e-34                          # plancks constant
    c = 3.0e8                                     # speed of light in meters
    v = c / lam                                 # frequency of light
    moon_reflectivity = 0.10                    # moon's reflectivity is 10%
    photons = (D*D*Intensity * (Moon_dist * dtheta) *(Moon_dist * dtheta)* dt * moon_reflectivity) /(4*Moon_dist*Moon_dist* (h * v))
    # energy = (photons / (4.0 * np.pi * Moon_dist ** 2.0)) * np.pi * (D / 2.0)**2.0


    # Make Image reflect real energy values

    # moon_max = np.max(np.max(Source_img))
    moon_max = Source_img.max()

    norm_moon = np.divide(Source_img, moon_max)

    photons_img = photons*np.multiply(norm_moon, moon_outline)+background

    # Add dim objects
    obj_reflectivity = 1
    obj_photons = (Intensity * dpix * dt * obj_reflectivity) / (h * v)
    # obj_photons = 5e20

    images = np.zeros([3000,3000])
    images = photons_img
    

    return images

photon_img = simulate_moon(0.07, 0.4, 610.0*10**-9,.1,100)
D = 0.07                        # diameter of telescope in meters
obs = 0                         # obscuration diameter
lamb = 610*10**(-9)             # wavelength of ligth in meters
f = 0.4                         # focal length in meters
si = 3000                       # ccd pixel dimensions
pixle_per_meter = si/(2*D)          # For Nyquist sampling
          # distance between pixels in meters
r1 = (D/2)*pixle_per_meter      # radius of telescope in pixels
r2 = (obs/2)*pixle_per_meter    # radius of obscuration in pixels
scale = 1                       # value at dc
pupil_test=make_pupil(r1,r2,si)
[tele_otf,norm_psf2]=make_otf2(1,pupil_test)

                 
ro = 0.02
r1 = D/2
dx = 4*r1/si

atmosphere_otf = make_long_otf2(r1,dx,si,ro)
detector_otf=detector_blur(2,2,si)
otfTurbulent = np.multiply(tele_otf, atmosphere_otf)
tot_otf=np.multiply(tele_otf,detector_otf)
coronagraph_input = real(ifft2(np.multiply(otfTurbulent, fft2(photon_img))))
coronagraph=make_coronagraph(1100,si,0,0)
coronagraph_output=np.multiply(coronagraph,coronagraph_input)
output_img = real(ifft2(np.multiply(tot_otf, fft2(coronagraph_output))))
output_img2=output_img[0:si:2,0:si:2]
# Add Poison Noise
# noisy_img = random_noise(real(output_img), mode='poisson')
noisy_img = np.random.poisson(output_img2)