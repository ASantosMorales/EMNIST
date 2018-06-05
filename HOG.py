#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 30 19:35:35 2018

@author: a_santos
"""

# Importing libreries and tools
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Function to obtain the Histogram of Oriented Gradients (HOG) description
def HOG(img):
    bin_n = 12                                                                                  # Number of bins 
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)                                                       # Getting the X gradients
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)                                                       # Getting the Y gradients
    mag, ang = cv2.cartToPolar(gx, gy)                                                          # Getting the gradients in polar description
    bins = np.int32(bin_n*ang/(2*np.pi))                                                        # Calculating the bins
    bin_cells = bins[:14,:14], bins[14:,:14], bins[:14,14:], bins[14:,14:]                      # Divide the matrices in 4 sub-squares
    mag_cells = mag[:14,:14], mag[14:,:14], mag[:14,14:], mag[14:,14:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]    # Getting the histograms
    hist = np.hstack(hists)
    return(hist)

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
HOG_description = []                                            # Creation of the empty list to store the HOG description
for character in labels:
    character_index = labels.index(character)
    index = 0
    for index_emnist_label in emnist_labels:
        if character_index == index_emnist_label:
            img = eval(variables[character_index])[index]       # Getting the character (image 28 x 28) 
            HOG_description.append(np.transpose(HOG(img)))      # Applying HOG
            index += 1
    print(character)

# Save data
np.save('HOG.npy', HOG_description)