# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 18:35:20 2024

@author: thecainfamily
"""
import numpy
import numpy as np
from PIL import Image, ImageSequence
object=np.zeros((100,100,100));
im = Image.open("1ms_54offset_gain120.tif")
for i, page in enumerate(ImageSequence.Iterator(im)):
    object[i]=np.array(page)
            
numpy.save('ds1',object )

object=np.zeros((100,100,100));
im = Image.open("2ms_54offset_gain120.tif")
for i, page in enumerate(ImageSequence.Iterator(im)):
    object[i]=np.array(page)

numpy.save('ds2', object)

object=np.zeros((100,100,100));
im = Image.open("20ms_54offset_gain120.tif")
for i, page in enumerate(ImageSequence.Iterator(im)):
    object[i]=np.array(page)

numpy.save('ds3', object)