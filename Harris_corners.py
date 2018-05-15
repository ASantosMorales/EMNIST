#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 11 19:25:04

This script obtains the Harris corners of each image.
First, the script shows how the outcome of the Harris detector algorithm
looks like.
After that, it also construct the features vector corresponding for Harris corners.

@author: a_santos
"""
# Importing libreries and tools
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Loading data
file = '/media/a_santos/DATA/Shared_Windows_Ubuntu/Python_projects/EMNIST/database/EMNIST_byclass_train.mat'
#file = 'C:/Users/A_Santos/OneDrive/Documentos/Python Scripts/EMNIST_byclass_test.mat'
data = sio.loadmat(file, squeeze_me=True, struct_as_record=False)
emnist_images = data['images_train']
emnist_labels = data['labels_train']
del(data, file) #These varible are no more necessaries

# Making labels
variables = ['_0', '_1', '_2', '_3', '_4', '_5', '_6', '_7', '_8', '_9', '_A', 
          '_B', '_C', '_D', '_E', '_F', '_G', '_H', '_I', '_J', '_K', '_L',
          '_M', '_N', '_O', '_P', '_Q', '_R', '_S', '_T', '_U', '_V', '_W',
          '_X', '_Y', '_Z', '_a', '_b', '_c', '_d', '_e', '_f', '_g', '_h',
          '__i', '_j', '_k', '_l', '_m', '_n', '_o', '_p', '_q', '_r', '_s',
          '_t', '_u', '_v', '_w', '_x', '_y', '_z']

labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 'A', 'B', 'C', 'D', 'E', 'F', 
          'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 
          'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 
          'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 
          'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

# Creation of lists. A list for each character with all their images
for i in range(len(labels)):
    temp = []
    for j in range(len(emnist_labels)):
        if  labels[i] == labels[emnist_labels[j]]:
            temp.append(np.flip(np.rot90(np.reshape(emnist_images[j, :], (28, 28)), 3), 1))
    globals()[variables[i]] = temp
    print(str(i))
          
# Harris-corner-feature-vector creation
Harris_corners_feature_vector = []
for character in labels:
    character_index = labels.index(character)
    index = 0
    for index_emnist_label in emnist_labels:
        if character_index == index_emnist_label:
            character_image = eval(variables[character_index])[index]               # Choose the character (image 28 x 28)
            try:
                dst = cv2.cornerHarris(character_image, 2, 3, 0.04)                 # Try to get the Harris corners
            except:
                dst = np.zeros([28, 28])                                            # If is not possible to get the corners, return a zero matrix
                print('Not possible')
            Harris_corners_feature_vector.append(len(np.where(dst > 0.025)[0]))     # Store the number o Harris corners
            index += 1
    print(character)

# Save data
np.save('Harris_corners_feature_vector.npy', Harris_corners_feature_vector)

#%% Show some random images focusing on the Harris corners
rows_to_show = 5
columns_to_show = 10
random_indexes = np.random.randint(0, len(labels), columns_to_show)
f, axes = plt.subplots(rows_to_show, columns_to_show)
k = 0
for i in range(rows_to_show):
    for j in range(columns_to_show):
        character = eval(variables[random_indexes[j]])[k]           #Getting a random image
        img_color = cv2.cvtColor(character, cv2.COLOR_GRAY2RGB)     #Changing to rgb image
        dst = cv2.cornerHarris(character, 2, 3, 0.04)               #Applying Harris detector
        img_color[dst > 0.025] = [255, 0, 0]                        #Focusing in the Harris corners (red_color = [255, 0, 0])
        axes[i, j].imshow(img_color)                                #Showing the image
        axes[i, j].set_axis_off()
        k += 1
