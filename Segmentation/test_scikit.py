# -*- coding: utf-8 -*-
"""
Created on Wed May 31 09:21:50 2023

@author: S.Mouhamadi
"""

from skimage import filters, feature, segmentation, exposure
import bm4d
import numpy as np
import scipy.io as sp
import matplotlib.pyplot as plt

# Charger l'image Matlab 3D
mat_data = sp.loadmat('C:/Users/Portatil PC 7/Documents/PostProcessingTool/MotoGP-RawDatas/20221105/5_Elena/RARE.2022.11.05.17.22.29.768.mat')
image_data = mat_data['image3D']
image_data = np.abs(image_data)  # prendre la magnitude pour avoir une image réelle
image_data = image_data.astype(np.float)  # conversion en float

# Estimate the noise level using the MAD method
#   sigma_psd = np.median(image_data)/0.6745

# noise = np.std(image_data)


med = np.median(image_data)
mad = np.median(np.abs(image_data - med))
noise = 1.4826 * mad
sigma_psd = noise / np.sqrt(2 * np.log(image_data.size))

print(f"Noise level (MAD): {sigma_psd:.2e}")

# Débruitage de l'image
denoised_image = bm4d.bm4d(image_data, sigma_psd=sigma_psd)

# Affichage de l'image originale
plt.subplot(1, 2, 1)
plt.title('Normalized image')plt.imshow(denoised_image[20, :, :], cmap='gray')
plt.axis('off')
plt.title('Denoised image')

#Test of filters.sobel
edges = filters.sobel(denoised_image[20, :, :])
plt.subplot(1, 2, 2)
plt.imshow(edges, cmap='gray')
plt.axis('off')
plt.title('Contours (Sobel)')

#Test of all thresholding filter
filters.try_all_threshold(denoised_image[20, :, :])
plt.show()

# Affichage de l'image débruitée, de l'image normalisée et du résultat de mark_boundaries
plt.figure()
plt.subplot(1, 3, 1)
plt.imshow(denoised_image[20, :, :], cmap='gray')
plt.axis('off')
plt.title('Denoised mage')

# Normalisation des intensités de l'image
normalized_image = exposure.rescale_intensity(denoised_image[20, :, :])

plt.subplot(1, 3, 2)
plt.imshow(normalized_image, cmap='gray')
plt.axis('off')


#Test of mark_boundaries
image_edges = feature.canny(normalized_image, sigma=0.5)
image = segmentation.mark_boundaries(normalized_image, image_edges)
plt.subplot(1, 3, 3)
plt.imshow(image, cmap='gray')
plt.axis('off')
plt.title('mark_boundaries Result')
plt.show()
