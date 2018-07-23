# -*- coding: utf-8 -*-
"""
SpytLab speckle python lab
Author: Helene Labriet & Emmanuel Brun
Date: April 2018
"""

#import EdfFile as edf
import fabio
import fabio.edfimage as edf
import fabio.tifimage as tif
#import edfimage
from PIL import Image
import numpy as np
import scipy
from scipy.misc import imsave as imsave
import glob
import skimage.io



def openImage(filename):
    filename=str(filename)
    im=fabio.open(filename)
    imarray=im.data
    return imarray



def saveEdf(data,filename):
    print(filename)
    dataToStore=data.astype(np.float32)
    edf.EdfImage(data=dataToStore).write(filename)


