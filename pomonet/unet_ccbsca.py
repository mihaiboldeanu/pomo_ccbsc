# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 18:27:40 2021

@author: 40737
"""
import gc
import os
from typing import List
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras import backend as K

from skimage.morphology import dilation, disk
import matplotlib.pyplot as plt



import ccbsc_api
from ccbsc_api.classification import Classification,Coordinates,Pollen
from ccbsc_api.device import Photo

import pomonet.utils as utils

class Unet_ccbsca(ccbsc_api.CCBSCAlgorithm):
    def __init__(self,validation=False):
        dependencies = {'mean_IOU': utils.mean_IOU}
        my_path = os.path.abspath(os.path.dirname(__file__))
        path_unet = os.path.join(my_path, r"models/u_net_classical-0.00299.hdf5")
        self.model_u_net = load_model(path_unet,custom_objects=dependencies)
        path_cnn = os.path.join(my_path, r"models/cnn_model-0-aug-0.93770.hdf5")
        self.model_cnn = load_model(path_cnn,custom_objects=dependencies)
        self.identifier = "u_net_v0.299_cnn_model_v0.937"
        self.identifier = "u_net_v0.299_cnn_model_v0.937"
        self.validation = validation
    
    def classify(self, photo: Photo) -> List[Classification]:
        """ Takes the given photo and return a list of classifications found in the photo. Override 
            this method in you algorithm. """
            
        numpy_array_image = np.array(photo.image)
        
        temp_x = numpy_array_image/255.0
        temp_x = np.reshape(temp_x,(1,temp_x.shape[0],temp_x.shape[1],1))
        
        temp_x_tensor = tf.convert_to_tensor(temp_x, dtype=tf.float32)
        
        result_u_net = self.model_u_net(temp_x_tensor)
        result_u_net = np.array(result_u_net)
        
        expanded_mask = utils.expand_mask(result_u_net)
        expanded_mask = np.reshape(expanded_mask,(1,960,1280,1))
        
        if (expanded_mask==2).any():
            alternaria = utils.get_one_class(expanded_mask)
            coord_list = utils.segment_image(alternaria)
            coord_list_valid,expanded_mask = utils.validate_coords(coord_list,expanded_mask,alternaria)
        else:
            coord_list_valid = []

            
        
        if coord_list_valid:
            
            x=np.zeros((len(coord_list_valid),360,360,1))
            ret_list = []
            
            for j,corners in enumerate(coord_list_valid):
                crop_for_cnn = utils.get_crop(numpy_array_image,corners)
                
                mask_for_cnn = utils.get_crop(alternaria[0,:,:,0],corners)
                mask_for_cnn[:,:,0] = dilation(mask_for_cnn[:,:,0],selem=disk(6))
                mask_for_cnn = mask_for_cnn > 0
                
                clean_crop = crop_for_cnn * mask_for_cnn
                x[j,:,:,:] = clean_crop[0:360,0:360,:] 
            
            x = x/255.
            
            x_cnn_tensor = tf.convert_to_tensor(x, dtype=tf.float32) 
            confirmation = self.model_cnn(x_cnn_tensor)
            confirmation = np.argmax(confirmation,axis=1)
            if self.validation:
                directory = r".\\validation"
                if not os.path.exists(directory):
                    os.makedirs(directory)
                name = photo.id
                f = plt.figure()
                plt.imshow(numpy_array_image,cmap="gray")
                for coord in coord_list_valid:
                    plt.scatter(coord[2],coord[0])
                    plt.scatter(coord[3],coord[0])
                    plt.scatter(coord[2],coord[1])
                    plt.scatter(coord[3],coord[1])
                file_path = os.path.join(directory,name+".png")
                plt.tight_layout()
                plt.savefig(file_path,bbox_inches='tight')
                f.clear()
                plt.close(f)
                plt.close("all")
               

            
            for c,conf in enumerate(confirmation):
                crop_target = coord_list_valid[c]
                x = crop_target[2]
                width = crop_target[3] - crop_target[2]
                y = crop_target[0]
                height =  crop_target[1] - crop_target[0]
                if conf == 1:
                    
                    features = "confirmed by CNN"
                else:
                    features = None
                coordinates = Coordinates(x, width, y, height)
                pollen_class = Classification(coordinates, Pollen.ALTERNARIA, photo,features)
                ret_list.append(pollen_class)
        else:
    
            ret_list = []
        
        K.clear_session()
        _ = gc.collect()
        return ret_list
    
    def identifier(self):
        """ Returns the identifier of the actual algorithm. """
        return self.identifier