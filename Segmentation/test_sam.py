# -*- coding: utf-8 -*-
"""
Created on Wed May 24 15:31:51 2023

@author: S.Mouhamadi
"""

import torch
import cv2
import supervision as sv
import scipy.io as sio
import numpy as np

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_h"


from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

# Charger le fichier .mat
mat_data = sio.loadmat('C:/Users/Portatil PC 7/Documents/Test_BM4D/rodillaBUENO_image3D.mat')

# Accéder à l'objet contenant les données de l'image
image_data = mat_data['image3D']

# Maintenant, vous pouvez utiliser les données de l'image dans votre code de segmentation
# Assurez-vous de comprendre la structure des données et d'effectuer les opérations appropriées.

# Par exemple, si vous voulez convertir les données en une image BGR pour une utilisation avec OpenCV,
# vous pouvez faire ce qui suit :
image_data_real = image_data.real.astype(np.uint8)
image_data_imag = image_data.imag.astype(np.uint8)
image_bgr = cv2.merge((image_data_real, image_data_imag, image_data_imag))


CHECKPOINT_PATH = "./sam_vit_h_4b8939.pth"
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)

mask_generator = SamAutomaticMaskGenerator(sam)
IMAGE_PATH ='C:/Users/Portatil PC 7/Documents/Test_BM4D/rodillaBUENO_image3D.mat'

image_bgr = cv2.imread(IMAGE_PATH)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_RGB2BGR)

sam_result = mask_generator.generate(image_rgb)

mask_annotator = sv.MaskAnnotator()

detections = sv.Detections.from_sam(sam_result=sam_result)

annotated_image = mask_annotator.annotate(scene=image_bgr.copy(), detections=detections)

sv.plot_images_grid(
    images=[image_bgr, annotated_image],
    grid_size=(1, 2),
    titles=['source image', 'segmented image']
)