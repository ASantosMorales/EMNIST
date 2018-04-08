#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 19:40:54 2018

This script only shows some random characters of the
EMNIST database to prove that the catalog reading is
alredy mastered.

@author: a_santos
"""

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

#%% Load data
file = '/home/a_santos/Documents/Research/EMNIST/EMNIST_byclass_train.mat'
#file = 'C:/Users/A_Santos/OneDrive/Documentos/Python Scripts/EMNIST_byclass_test.mat'
data = sio.loadmat(file, squeeze_me=True, struct_as_record=False)
emnist_images = data['images_train']
emnist_labels = data['labels_train']
del(data, file) #These varible are no more necessaries

#%% Make labels
labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 'A', 'B', 'C', 'D', 'E', 'F', 
          'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 
          'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 
          'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 
          'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
   
#%% Show some random images
rows_to_show = 10
columns_to_show = 10
random_indexes = np.random.randint(0, len(emnist_labels), rows_to_show*columns_to_show)

f, axes = plt.subplots(rows_to_show, columns_to_show)
k = 0
for i in range(rows_to_show):
    for j in range(columns_to_show):
        character = np.flip(np.rot90(np.reshape(emnist_images[random_indexes[k], :], (28, 28)), 3), 1)
        axes[i, j].imshow(character, cmap='gray')
        axes[i, j].set_title(labels[emnist_labels[random_indexes[k]]])
        axes[i, j].set_axis_off()
        k+=1