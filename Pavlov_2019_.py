import os
import sys


from spytIO import openImage, saveEdf
from numpy.fft import fftshift as fftshift
from numpy.fft import ifftshift as ifftshift
from numpy.fft import fft2 as fft2
from numpy.fft import ifft2 as ifft2
from math import pi as pi
import numpy as np

import glob
# from NoiseTracking.OpticalFlow import pavlovThread
import sys


def kevToLambda(energyInKev):
    energy = energyInKev * 1e3
    waveLengthInNanometer = 1240. / energy

    return waveLengthInNanometer * 1e-9


def tie_Pavlovetal2019(Is, Ir, energy, z1, z2, pix_size, delta, beta, bg_val, scale):
    """ Phase retrieval using the Transport of Intensity Equation for homogeneous samples in the case of speckles
        Parameters
       ----------
       arg1 : Is
          Image with the sample and the membrane
       arg2 : Ir
          Image with the membrane only


       arg3 : str
           energy = mean x-ray energy (keV)
       arg4: float
           z1 = source to sample distance (m)
       arg5: float
           z2 = sample to detector distance (m)
       arg6: float
           pix_size = size of the image pixels (m)

       arg7: float
           delta = refractive index

       arg8: float
           beta = absorption par    t of the refactive index

       arg9: float
           bg_val = background intensity in img_in (e.g. 0, 1, 100...)

       arg10: float
           scale = parameter in the low-frequency filter (e.g., 2) (added by KMP)

       Returns
       -------
       float
           img_thickness = image of sample thickness (m)

       """

    lambda_energy = kevToLambda(energy)
    waveNumber = (2 * pi) / lambda_energy
    mu = 2 * waveNumber * beta

    magnificationFactor = (z1 + z2) / z1
    pix_size=pix_size*magnificationFactor
    #pix_size = pix_size * magnificationFactor

    sigmaSource = 150.e-6

    gamma = delta / beta

    is_divided_by_Ir = np.true_divide(Is, Ir)

    numerator = 1 - is_divided_by_Ir

    # average_image = np.mean(numerator)
    # Correction on the average image. Now the average of the new array is ~0
    # numerator = numerator - average_image

    saveEdf(numerator, 'ImageNew.edf')

    padCol = 1600
    padRow = 1600
    width, height = numerator.shape
    numerator = np.pad(numerator, ((padRow, padRow), (padCol, padCol)), 'reflect')

    fftNumerator = fftshift(fft2(numerator))

    Nx, Ny = fftNumerator.shape
    print('Nx:'+str(Nx)+' Ny:'+str(Ny))
    u, v = np.meshgrid(np.arange(0, Nx), np.arange(0, Ny))
    u = (u - (Nx / 2))
    v = (v - (Ny / 2))

    u_m=  u / (Nx * pix_size)
    v_m = v / (Ny * pix_size)
    uv_sqr=  np.transpose(u_m ** 2 + v_m ** 2)  # ie (u2+v2)
    # without taking care of source size
    # denominator = 1 + pi * gamma * z2 * lambda_energy * k_sqr

    # Beltran et al method to deblur with source
    denominator = 1 + pi * (gamma * z2 - waveNumber * sigmaSource * sigmaSource) * lambda_energy * uv_sqr

#    denominator *= magnificationFactor
    tmp = fftNumerator / denominator

    # Low pass filter
    sigma_x = ((1/ (Nx * pix_size*1.6)) * scale) ** 2
    sigma_y = ((1/ (Ny * pix_size*1.6)) * scale) ** 2
    f = (1. - np.exp(-(u_m ** 2 / (2. * sigma_x) + v_m ** 2 / (2. * sigma_y))))  # ie f(x,y)
    lff = np.transpose(f)  # ie LFF

    # Application of the Low pass filter
    tmp = lff * tmp

    # inverse fourier transform
    tmpThickness = ifft2(ifftshift(tmp))  # F-1
    img_thickness = np.real(tmpThickness)
    # Division by mu
    img_thickness = img_thickness / mu
    # multiplication to be in micron
    img_thickness = img_thickness * 1e6
    # unpadding
    img_thickness = img_thickness[padRow:padRow + width, padCol:padCol + height]
    img_thickness += bg_val

    return img_thickness


if __name__ == "__main__":
    # testOneImage()
    irName = 'Ref0367.edf'

    isName = 'im0007.edf'
    imageSample = openImage(isName)
    imageReference = openImage(irName)
    imageSample=imageSample[411:411+400,768:768+400]
    imageReference = imageReference[411:411 + 400, 768:768 + 400]
    # darkName='/VOLUMES/ID17/broncho/IHR_April2018/HA800_Patte21_3um_Gap90_75_Speckle02_/dark.edf'

    # result=processOneImage(isName,irName)

    beta = 2.7274690492888E-12
    delta = 1.0430137117588E-07
    z1 = 142  # (m) distance from source to object
    z2 = 11.0  # (m) distance from the object to the detector
    pix_size = 6.1e-6

    result = tie_Pavlovetal2019(imageSample, imageReference, 52, z1, z2, pix_size, delta, beta, 0, 12)

    saveEdf(result, 'thickness.edf')
