#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 14:36:39 2018

@author: a_santos
"""

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import cv2

#%% Load data
file = '/home/a_santos/Documents/Research/EMNIST/EMNIST_byclass_train.mat'
#file = 'C:/Users/A_Santos/OneDrive/Documentos/Python Scripts/EMNIST_byclass_test.mat'
data = sio.loadmat(file, squeeze_me=True, struct_as_record=False)
emnist_images = data['images_train']
emnist_labels = data['labels_train']
del(data, file) #These varible are no more necessaries

#%% Make labels
variables = ['_0', '_1', '_2', '_3', '_4', '_5', '_6', '_7', '_8', '_9', '_A', 
          '_B', '_C', '_D', '_E', '_F', '_G', '_H', '_I', '_J', '_K', '_L',
          '_M', '_N', '_O', '_P', '_Q', '_R', '_S', '_T', '_U', '_V', '_W',
          '_X', '_Y', '_Z', '_a', '_b', '_c', '_d', '_e', '_f', '_g', '_h',
          '_i', '_j', '_k', '_l', '_m', '_n', '_o', '_p', '_q', '_r', '_s',
          '_t', '_u', '_v', '_w', '_x', '_y', '_z']

labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 'A', 'B', 'C', 'D', 'E', 'F', 
          'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 
          'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 
          'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 
          'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

#%% Creation of lists. A list for each character
count = []
for i in range(len(labels)):
    temp = []
    k = 0
    for j in range(len(emnist_labels)):
        if  labels[i] == labels[emnist_labels[j]]:
            temp.append(np.flip(np.rot90(np.reshape(emnist_images[j, :], (28, 28)), 3), 1))
            k += 1
    count.append(k)
    globals()[variables[i]] = temp
    print(str(i))
#%%
img = None
char = eval(variables[8])
for i in range(100):
    im = char[i]
    if img is None:
        img = plt.imshow(im)
    else:
        img.set_data(im)
    plt.pause(1)
    plt.draw()

#%%
img = None
char = eval(variables[8])
for i in range(100):
    im = char[i]
    img_color = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
    dst = cv2.cornerHarris(im, 2, 3, 0.04)                                 #Apply Harris corner detector
    img_color[dst > 0.03] = [0, 0, 255] 
    if img is None:
        #img = plt.imshow(im, cmap = 'gray')
        img = plt.imshow(img_color)
    else:
        img.set_data(img_color)
    plt.pause(1)
    plt.draw()

#%%
chars = []
count_2 = []
for i in range(len(labels)):
    k = 0
    temp = eval(variables[i])
    for j in range(len(temp)):
        chars.append(labels[i])
        k += 1
    count_2.append(k)