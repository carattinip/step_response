# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 09:50:00 2022

@author: thecainfamily
"""
import numpy as np
from scipy.fft import fft2

def detector_blur(blurx,blury,si):

    psf=np.zeros((si,si));
    for ii in range(0,blurx-1):
        for jj in range(0,blury-1):
            ycord=int(np.round(jj+si/2))
            xcord=int(np.round(ii+si/2))
            psf[ycord][xcord]=1
    otf=np.abs(fft2(psf))
    return(otf)


# detector_otf=detector_blur(5,5,256)
