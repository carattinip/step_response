# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 14:16:37 2024

@author: thecainfamily
"""

import numpy as np
from scipy.fft import fft2, ifft2, fftshift

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


def make_coronagraph(r1, si, xs, ys):
    import numpy as np
    if (2 * np.floor(si / 2) == si):
        mi = int(np.floor(si / 2));
        pupilx = np.zeros([si, si])
        for i in range(0, si - 1):
            pupilx[i] = range(-mi, mi)

    if (2 * np.floor(si / 2) != si):
        mi = int(np.floor(si / 2));
        pupilx = np.zeros([si, si])
        for i in range(0, si - 1):
            pupilx[i] = range(-mi, mi + 1)
    pupily = np.transpose(pupilx)
    dist2 = np.multiply(pupilx, pupilx) + np.multiply(pupily, pupily)
    dist = np.sqrt(dist2)
    pupil2 = (dist > r1)
    pupil = np.multiply(pupil2.astype(int), np.ones((si, si)))
    pupil2 = np.roll(pupil, ys, xs)
    return (pupil2)



# sz=1001
# source_array=np.zeros((sz,sz))
# source_array[501][481]=1
#
# pupil_field=fftshift(fft2(fftshift(source_array)))
# entrance_pupil=make_pupil(250,0,sz)
# exit_pupil=np.multiply(pupil_field,entrance_pupil)
#
# coronagraph_plane=fftshift(fft2(fftshift(exit_pupil)))/(sz*sz)
# temp=np.abs(coronagraph_plane)
# star_intensity=np.multiply(temp,temp)
# coronagraph_stop=make_coronagraph(50,sz)
# coronagraph_exit=np.multiply(coronagraph_plane,coronagraph_stop)
# second_pupil=make_pupil(250,0,sz)
# pupil_field2=fftshift(fft2(fftshift(coronagraph_exit)))
# second_pupil_exit=np.multiply(second_pupil,pupil_field2)
# image_plane=fftshift(fft2(fftshift(second_pupil_exit)))/(sz*sz)
# temp=np.abs(image_plane)
# final_intensity=np.multiply(temp,temp)
# temp2=final_intensity[501]

