#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 16 20:56:37 2018

@author: a_santos
"""

# Importing libreries and tools
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Loading data
file = '/media/a_santos/DATA/Shared_Windows_Ubuntu/Python_projects/EMNIST/database/EMNIST_byclass_train.mat'
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

# Holes-detection-vector creation
Holes_number = []
for character in labels:
    character_index = labels.index(character)
    index = 0
    for index_emnist_label in emnist_labels:
        if character_index == index_emnist_label:
            character_image = eval(variables[character_index])[index]                                       # Getting the character (image 28 x 28)
            ret, thresh = cv2.threshold(character_image, 100, 255, 0)                                       # Image binarization
            im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)     # Gettting the contours (contours - 1 = holes)
            Holes_number.append(len(contours) - 1)                                                          # Vector creation
            index += 1
    print(character)

# Save data
np.save('Holes_feature_vector.npy', Holes_number)

#%% 
# This section is to show some examples of the hole-detection outcome. In this case de zero
# number is analyzed. 

# Getting the indexes where the holes number is greater than expected. 
many_holes_indexes = []                                                                             # Creation of the empty list to store the indexes
for index in range(len(eval(variables[labels.index(0)]))):
    img = eval(variables[labels.index(0)])[index]                                                   # Getting the particular zero image
    ret, thresh = cv2.threshold(img, 100,255,0)                                                     # Image binarization
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)     # Getting the contours (contours - 1 = holes)
    if (len(contours) -1) > 1:                                                                      # The holes-number is greater than expected?
        many_holes_indexes.append(index)                                                            # Store the index in the list
    if 10 == len(many_holes_indexes):                                                               # Getting only the 10 first images that fulfill the condition
        break
    
# Image creation    
rows_to_show = 1                                                                                    # Specifying the outcome-image-shape
columns_to_show = 10
f, axes = plt.subplots(rows_to_show, columns_to_show)
k = 0
for j in range(columns_to_show):
    img = eval(variables[labels.index(0)])[many_holes_indexes[k]]                                   # Getting the particular zero image
    ret, thresh = cv2.threshold(img, 100, 255, 0)                                                   # Image binarization
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)     # Getting the contours (contours - 1 = holes)
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)                                               # Becoming the gray-scale-image into color-image
    for contour in range(1, len(contours)):
        for (y, x) in contours[contour].reshape(-1, 2):
            img_color[x, y, :] = 0                                                                  # Drawing the contours
            img_color[x, y, contour%3] = 255
    axes[j].imshow(img_color)                                                                       # Outcome-image creation
    axes[j].set_axis_off()
    axes[j].set_title('Holes = {}'.format(len(contours) - 1))
    k += 1