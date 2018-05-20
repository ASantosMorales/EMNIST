#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 19 15:42:07 2018

@author: a_santos

Support resources: https://stackoverflow.com/questions/32793703/how-can-i-get-ellipse-coefficient-from-fitellipse-function-of-opencv
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

labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B',
          'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
          'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
          'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
          'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 
          'y', 'z']

# Creation of lists. A list for each character with all their images
for i in range(len(labels)):
    temp = []
    for j in range(len(emnist_labels)):
        if  labels[i] == labels[emnist_labels[j]]:
            temp.append(np.flip(np.rot90(np.reshape(emnist_images[j, :], (28, 28)), 3), 1))
    globals()[variables[i]] = temp
    print(str(i))

#%%
# Holes-detection-vector creation
elongation = []                                                                                             # Creation of the empty list to store the elongation values
errors = []                                                                                                 # Creation of the empty list to store the errors (normally with the 1 number)
for character in labels:
    character_index = labels.index(character)
    index = 0
    for index_emnist_label in emnist_labels:
        if character_index == index_emnist_label:
            img = eval(variables[character_index])[index]                                                   # Getting the character (image 28 x 28)
            ret, thresh = cv2.threshold(img, 100, 255, 0)                                                   # Image binarization
            im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)     # Gettting all the contours
            contour = max(contours, key = len)                                                              # Gettting the contour with more points (The shape more important in the hierarchy)
            try:
                ellipse = cv2.fitEllipse(contour)                                                           # Getting the ellipse parameters
                elongation.append(round(max(ellipse[1])/min(ellipse[1]), 2))                                # Vector creation
            except:
                elongation.append(0)                                                                        # If error, write a zero
                errors.append([img, contours])                                                              # Store the error to future analysis
            index += 1
    print(character)

# Save data
np.save('Elongation.npy', elongation)

#%%
rows_to_show = 1                                                                                    # Specifying the outcome-image-shape
columns_to_show = 10
random_indexes = np.random.randint(0, len(labels), columns_to_show)
f, axes = plt.subplots(rows_to_show, columns_to_show)
for j in range(columns_to_show):
    img = eval(variables[random_indexes[j]])[0]                                                     # Getting a random image to analyze
    et, thresh = cv2.threshold(img, 100, 255, 0)                                                    # Image binarization
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)     # Gettting all the contours
    contour = max(contours, key = len)                                                              # Gettting the contour with more points (The shape more important in the hierarchy)
    ellipse = cv2.fitEllipse(contour)                                                               # Getting the ellipse parameters
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)                                               # Becoming the gray-scale-image into color-image
    ellipse_image = cv2.ellipse(img_color, ellipse, (0, 255, 0), 1)                                 # Drawing the contours
    ellipse_image[round(ellipse[0][0]), round(ellipse[0][1]), 0] = 255                              # Drawing the center of the image
    ellipse_image[round(ellipse[0][0]), round(ellipse[0][1]), 1] = 0
    ellipse_image[round(ellipse[0][0]), round(ellipse[0][1]), 2] = 0
    
    # Axis determination
    mayor_axe = round(max(ellipse[1]), 2)
    minor_axe = round(min(ellipse[1]), 2)
    ratio = round(mayor_axe/minor_axe, 2)
    
    # Plotting
    axes[j].imshow(ellipse_image)                                                                       # Outcome-image creation
    axes[j].set_axis_off()
    axes[j].set_title('{}'.format(labels[random_indexes[j]]) + '\nmayor axis = {}'.format(mayor_axe) +
        '\nminor axis = {}'.format(minor_axe) + '\nratio = {}'.format(ratio),
        fontsize = 10)