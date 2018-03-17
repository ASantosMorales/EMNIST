#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 19:40:54 2018

@author: a_santos
"""

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt 
import cv2

#%% Load data
file = 'C:/Users/A_Santos/OneDrive/Documentos/Python Scripts/EMNIST_byclass_test.mat'
data = sio.loadmat(file, squeeze_me=True, struct_as_record=False)
emnist_images = data['images_test']
emnist_labels = data['labels_test']
del(data, file) #These varible are no more necessaries

#%% Show some random images
random_indexes = np.random.randint(0, len(emnist_labels), 25)

f, axes = plt.subplots(5, 5)
k = 0
for i in range(5):
    for j in range(5):
        axes[i, j].imshow(np.reshape(emnist_images[k, :], (28, 28)), cmap='gray')
        k+=1
cv2.waitKey(0)
cv2.destroyAllWindows()
     
#%% Test of showing images
img = np.reshape(emnist_images[2, :], (28, 28))
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()