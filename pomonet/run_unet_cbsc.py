# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 13:24:16 2021

@author: Mihai Boldeanu
"""

import os
import gc
import time

import tensorflow as tf
from tensorflow.keras import backend as K



from ccbysc.device import  Photo_fake
from unet_ccbsca import Unet_ccbsca
#from pympler.tracker import SummaryTracker


# Free up RAM in case the model definition cells were run multiple times
tf.keras.backend.clear_session()

print("Loading images")
path = r"F:\pollen_clas\new data\different cases"
img_paths = os.listdir(path)
img_paths = [img for img in img_paths if ".png" in img]


print("Starting main")
new_classifier = Unet_ccbsca()
final_list = []

img_paths = img_paths 

t0 = time.time()

for i,file_name  in enumerate(img_paths):
    
    file_path = os.path.join(path,file_name)
    ### Created fake photo class to test the classifier
    ### Works like Photo class but read from 
    fake_image = Photo_fake(file_path)
    result = new_classifier.classify(fake_image)
    final_list.append(result)
    K.clear_session()
    _ = gc.collect()
t1 = time.time()

total = t1-t0 
print(total)


print("Starting main with validation")
newer_classifier = Unet_ccbsca(validation=True)
final_list = []
t0 = time.time()

for i,file_name  in enumerate(img_paths):
    
    file_path = os.path.join(path,file_name)
    ### Created fake photo class to test the classifier
    ### Works like Photo class but read from 
    fake_image = Photo_fake(file_path)
    result = newer_classifier.classify(fake_image)
    final_list.append(result)
    K.clear_session()
    _ = gc.collect()
t1 = time.time()
total = t1-t0 
print(total)
