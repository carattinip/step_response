# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 09:50:00 2022

@author: thecainfamily
"""
import numpy as np
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

# pupil_test=make_pupil(750,100,3000)

def make_otf(scale,cpupil):
    from scipy.fft import fft2
    import numpy as np
    
    psf=fft2(cpupil)
    psf=abs(psf)
    psf=np.multiply(psf, psf)
    spsf=np.sum(psf)
    norm_psf=scale*psf/spsf;
    otf=fft2(norm_psf)        
    return(otf,norm_psf)
# pupil_test=make_pupil(750,0,3000)
# [otf,psf]=make_otf2(1,pupil_test)
# otf_amp=np.sqrt(np.multiply(otf,np.conj(otf)))
# otf_img=otf_amp.astype(float)