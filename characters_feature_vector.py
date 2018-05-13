#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 12 20:13:56 2018

@author: a_santos
"""
# Importing libreries and tools
import scipy.io as sio
import numpy as np

# Loading data
file = '/media/a_santos/DATA/Shared_Windows_Ubuntu/Python_projects/EMNIST/database/EMNIST_byclass_train.mat'
data = sio.loadmat(file, squeeze_me=True, struct_as_record=False)
emnist_labels = data['labels_train']

labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 'A', 'B', 'C', 'D', 'E', 'F', 
          'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 
          'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 
          'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 
          'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

#%% Characters feature vector creation
characters_feature_vector = []
for character in labels:
    character_index = labels.index(character)
    for index_emnist_label in emnist_labels:
        if character_index == index_emnist_label:
            characters_feature_vector.append(character)
    print(character)
    
# Save data
np.save('characters_feature_vector.npy', characters_feature_vector)