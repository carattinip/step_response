# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 09:50:00 2022

@author: thecainfamily
"""



def make_long_otf(r1,dx,si,ro):
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
    if isinstance(ro, np.ndarray):
        # proceed -> is an np array
        otf2 = np.zeros((len(ro), si, si))
        for i in range(len(ro)):
            temp = (dx * dist / ro[i]) ** (5 / 3)
            temp2 = -3.44 * temp
            otf = np.exp(temp2)
            otf2[i] = fftshift(otf)
    else:
        # Not an np.ndarray
        temp = (dx * dist / ro) ** (5 / 3)
        temp2 = -3.44 * temp
        otf = np.exp(temp2)
        otf2 = fftshift(otf)
    return(otf2)
# long_otf=make_long_otf2(5,20/3000,3000,2)
