# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 11:45:14 2023

@author: Portatil PC 7
"""
from skimage import filters, feature, segmentation, exposure
import bm4d
import numpy as np
import scipy.io as sp
import matplotlib.pyplot as plt

mat_data = sp.loadmat('C:/Users/Portatil PC 7/Documents/12_Guillermo/Foreleg/Left/mat/RAREprotocols_T1_TRA.2023.05.31.12.03.48.164.mat')
image_data = mat_data['image3D']
image_data = np.abs(image_data)  # prendre la magnitude pour avoir une image réelle
image_data = image_data.astype(np.float)  # conversion en float

# Estimate the noise level using the MAD method
#   sigma_psd = np.median(image_data)/0.6745

#noise = np.std(image_data)


med = np.median(image_data)
mad = np.median(np.abs(image_data - med))
noise = 1.4826 * mad
sigma_psd = noise / np.sqrt(2 * np.log(image_data.size))

print(f"Noise level (MAD): {sigma_psd:.2e}")

# Débruitage de l'image
denoised_image = bm4d.bm4d(image_data, sigma_psd=sigma_psd)

slice_index = 9  # Modify this to the desired slice index

normalized_image = exposure.rescale_intensity(denoised_image[slice_index, :, :])
image_edges = feature.canny(normalized_image, sigma=0.1)
marked_denoised = segmentation.mark_boundaries(normalized_image, image_edges)

plt.subplot(1, 3, 1)
plt.imshow(image_data[slice_index,:,:], cmap='gray')
plt.title(f'Original Image (Slice {slice_index+1})')

plt.subplot(1, 3, 2)
plt.imshow(denoised_image[slice_index,:,:], cmap='gray')
plt.title(f'Denoised Image (Slice {slice_index+1})')

plt.subplot(1, 3, 3)
plt.imshow(marked_denoised, cmap='gray')
plt.title(f'Contour Detection - Denoised Image (Slice {slice_index+1})')
