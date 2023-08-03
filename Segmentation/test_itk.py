# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 11:31:12 2023

@author: S.Mouhamadi
"""

import bm4d
import numpy as np
import scipy.io as sp
import matplotlib.pyplot as plt
from ipywidgets import interact, IntSlider
import itk

# Charger l'image Matlab 3D
mat_data = sp.loadmat('C:/Users/Portatil PC 7/Documents/12_Guillermo/Foreleg/Left/mat/RAREprotocols_T1_TRA.2023.05.31.12.03.48.164.mat')
image_data = mat_data['image3D']
image_data = np.abs(image_data)
image_data = image_data.astype(np.float64)

# Estimate the noise level using the MAD method
# sigma_psd = np.median(image_data) / 0.6745
noise = np.std(image_data)
sigma_psd = noise / np.sqrt(2 * np.log(image_data.size))
print(f"Noise level (MAD): {sigma_psd:.2e}")

# Débruitage de l'image
denoised_image = bm4d.bm4d(image_data, sigma_psd=sigma_psd)

# Convert the denoised image to ITK format
denoised_image_itk = itk.GetImageFromArray(denoised_image)

# Perform image segmentation using ITK filters
seg_filter = itk.ConnectedThresholdImageFilter.New(denoised_image_itk)
seg_edge = itk.CannyEdgeDetectionImageFilter(denoised_image_itk)
seed_point = itk.Index[3]()
seed_point[0] = 0
seed_point[1] = 5
seed_point[2] = 15
seg_filter.SetSeed(seed_point)  # Set the seed point for segmentation
seg_filter.SetLower(0)  # Set the lower threshold for segmentation
seg_filter.SetUpper(200)  # Set the upper threshold for segmentation
segmented_image = seg_filter.GetOutput()
segmented_image.Update()

# Convert the segmented image back to NumPy array format
segmented_image_np = itk.GetArrayFromImage(segmented_image)

# Rapport signal sur bruit
snr = np.mean(image_data) / np.std(image_data)
print(f"SNR: {snr:.2f}")

# Afficher les images sur la même figure
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

# Image originale
axs[0, 0].imshow(image_data[10, :, :], cmap='gray')
axs[0, 0].set_title('Image originale')

# Image débruitée avec BM4D
axs[0, 1].imshow(denoised_image[10, :, :], cmap='gray')
axs[0, 1].set_title('Image débruitée avec BM4D')

# Segmented image
axs[1, 0].imshow(segmented_image_np[10, :, :], cmap='gray')
axs[1, 0].set_title('Segmented Image')

# edge detector image
axs[1, 1].imshow(itk.CannyEdgeDetectionImageFilter(denoised_image_itk)[10, :, :], cmap='gray')
axs[1, 1].set_title('Edges')
plt.tight_layout()
plt.show()
