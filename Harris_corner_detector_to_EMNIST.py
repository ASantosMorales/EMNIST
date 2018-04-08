#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 14:22:17 2018

This script applies the Harris Corner Detector to the handwritten 
characters. The obtective is to do the feature extraction.

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
labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 'A', 'B', 'C', 'D', 'E', 'F', 
          'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 
          'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 
          'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 
          'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
     
#%% Show images emphasizing on the corners
img = np.flip(np.rot90(np.reshape(emnist_images[4, :], (28, 28)), 3), 1)#Import image
img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)                       #Change the grayscale image to color image
dst = cv2.cornerHarris(img, 2, 3, 0.04)                                 #Apply Harris corner detector
img_color[dst > 0.03] = [0, 0, 255]                                     #Emphasys on the corners (rgb_red = [0, 0, 255])

cv2.imshow('image',img_color)
cv2.waitKey(0)
cv2.destroyAllWindows() 