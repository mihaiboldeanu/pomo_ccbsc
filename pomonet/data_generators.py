# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 07:45:38 2021

@author: Mihai Boldeanu
"""


import os
import random

import numpy as np

from skimage.morphology import erosion, disk
from scipy.ndimage import rotate

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img




class Pollen(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size,
                 img_size, input_img_paths,
                 target_img_paths1,target_img_paths2,
                 augment=True,junk_value=1):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths1 = target_img_paths1
        self.target_img_paths2 = target_img_paths2
        self.augment = augment
        self.junk_value = junk_value

    def __len__(self):
        return len(self.target_img_paths1) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_img_paths1 = self.target_img_paths1[i : i + self.batch_size]
        batch_target_img_paths2 = self.target_img_paths2[i : i + self.batch_size]

        x, y, w = self.__data_generation(batch_input_img_paths,
                                      batch_target_img_paths1,
                                      batch_target_img_paths2)
        
        return x, y
    def __data_generation(self,
                          batch_input_path,
                          batch_target_img_paths1,
                          batch_target_img_paths2):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        x = np.zeros((self.batch_size,self.img_size[0],self.img_size[1],1))
        y = np.zeros((self.batch_size,self.img_size[0],self.img_size[1],1))
        w = np.zeros((self.batch_size,self.img_size[0],self.img_size[1],1))
        for i, _ in enumerate(batch_input_path):
            img = load_img(batch_input_path[i], target_size=self.img_size,color_mode="grayscale")
            img1 = load_img(batch_target_img_paths1[i], target_size=self.img_size, color_mode="grayscale")
            img2 = load_img(batch_target_img_paths2[i], target_size=self.img_size, color_mode="grayscale")
            flipud, fliplr, rotate_angle  = 0, 0 ,0
        
            if self.augment:
                flipud = np.random.random(1) > 0.5
                fliplr = np.random.random(1) > 0.5
                if np.random.random(1) > 0.5:
                    rotate_angle = np.random.randint(0,360,1)[0]
                else:
                    rotate_angle = 0
            
            temp_x = self.augment_f(img,flipud,fliplr,rotate_angle)
            temp_y1 = self.augment_f(img1,flipud,fliplr,rotate_angle)
            temp_y2 = self.augment_f(img2,flipud,fliplr,rotate_angle)
            
            temp_y1 = temp_y1 > 128
            temp_y2 = temp_y2 > 128

            temp_y = temp_y1 * 2 + temp_y2 * self.junk_value
            x[i,:,:,0] = temp_x
            y[i,:,:,0] = temp_y
            
        w += 0.1
        w[np.where(y>0)]=1
        w[np.where(y>1)]=2
        
        return tf.convert_to_tensor(x/255.), tf.convert_to_tensor(y), tf.convert_to_tensor(w)

    def augment_f(self,img,flipud,fliplr,rotate_angle):
        temp_x = np.array(img)
        if rotate_angle:
                temp_x = np.around(rotate(temp_x,rotate_angle,reshape=False))
                temp_x[np.where(temp_x<0)] = 0
    
        if flipud:
            temp_x = np.flip(temp_x,axis=0)
        if fliplr:
            temp_x = np.flip(temp_x,axis=1)
            
        return temp_x
    
    def on_epoch_end(self):
        seed = np.random.randint(12345)
        random.Random(seed).shuffle(self.input_img_paths)
        random.Random(seed).shuffle(self.target_img_paths1)
        random.Random(seed).shuffle(self.target_img_paths2)
        
    
class Pollen_synthetic(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self,
                 batch_size,
                 step_per_epoch,
                 img_size,
                 input_img_paths,
                 value_dict,
                 validation=False):
        self.batch_size = batch_size
        self.step_per_epoch = step_per_epoch
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.value_dict = value_dict
        self.validation = validation

    def __len__(self):
        return self.step_per_epoch

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""

        # Generate data
        if self.validation:
            random.seed(idx)
        else:
            random.seed(np.random.randint(0,913829128))
            
        x, y, w = self.__data_generation(idx)
        
        return x, y, w
    
    def __data_generation(self, idx):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        i = 0
        x = np.zeros((self.batch_size,self.img_size[0],self.img_size[1],1))
        y = np.zeros((self.batch_size,self.img_size[0],self.img_size[1],1))
        w = np.zeros((self.batch_size,self.img_size[0],self.img_size[1],1))
        
        while i < self.batch_size:
            
            part = random.randint(20,48)
            
            image = np.zeros((1320,1640))    
            mask =  np.zeros((1320,1640)) 
            
            selection = self.get_pollens(self.input_img_paths,number_examples = part)
            
            image,mask = self.add_pollen(image,mask,selection,self.value_dict)
                
                
            x[i,:,:,0] = image[180:1320-180,180:1640-180]
            y[i,:,:,0] = mask[180:1320-180,180:1640-180]
            # y[i] = tf.keras.utils.to_categorical(mask[180:1320-180,180:1640-180], num_classes=20)
            i+=1
        w += 0.1
        w[np.where(y>0)]=1
        # w[np.where(y>1)]=2
        return tf.convert_to_tensor(x/255.), tf.convert_to_tensor(y), tf.convert_to_tensor(w)
     
    def get_pollens(self,pollen_dict,number_examples = 10):
        
        
        keys = list(pollen_dict.keys())
        ret_particles = []
        
        while len(ret_particles) < number_examples:
            key = np.random.choice(keys,)
            ret_particles.append([key,random.choice(pollen_dict[key])])
            
        # for i in range( np.random.randint(0,5)):
        #     ret_particles.append(["alternaria",random.choice(pollen_dict['alternaria'])])# Force to have at least one alternaria particle
       
        return ret_particles
    

    def add_pollen(self,current_image,current_mask,particles,value_dict):

        for idx,particle in enumerate(particles):
            key, path = particle
            
            y_min = random.randint(0, 1280)
            y_max = y_min + 360
            x_min = random.randint(0, 960)
            x_max = x_min + 360
            img = load_img(path, 
                        target_size=(360,360),
                        color_mode="grayscale")
            img = np.array(img)
            if not self.validation: 
                if self.augment:
                    flipud = np.random.random(1) > 0.5
                    fliplr = np.random.random(1) > 0.5
                    if np.random.random(1) > 0.75:
                        rotate_angle = np.random.randint(0,360,1)[0]
                    else:
                        rotate_angle = 0
                img = self.augment(img,flipud,fliplr,rotate_angle)
            mask = ( img > 0 )
            reverse_mask = np.logical_not(mask)
            value_mask = mask * value_dict[key]
            current_image[x_min:x_max,y_min:y_max] = current_image[x_min:x_max,y_min:y_max] * reverse_mask + img
            current_mask[x_min:x_max,y_min:y_max] = current_mask[x_min:x_max,y_min:y_max] * reverse_mask + value_mask

        return current_image, current_mask
    
    def augment(self,img,flipud,fliplr,rotate_angle):
        temp_x = np.array(img)
        if rotate_angle:
                temp_x = np.around(rotate(temp_x,rotate_angle,reshape=False))
                temp_x[np.where(temp_x<0)] = 0
    
        if flipud:
            temp_x = np.flip(temp_x,axis=0)
        if fliplr:
            temp_x = np.flip(temp_x,axis=1)
            
        return temp_x

class Pollen_synthetic_inst(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self,
                 batch_size,
                 step_per_epoch,
                 img_size,
                 input_img_paths,
                 value_dict,
                 validation=False):
        self.batch_size = batch_size
        self.step_per_epoch = step_per_epoch
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.value_dict = value_dict
        self.validation = validation

    def __len__(self):
        return self.step_per_epoch

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""

        # Generate data
        if self.validation:
            random.seed(idx)
        else:
            random.seed(np.random.randint(0,913829128))
            
        x, y, w = self.__data_generation(idx)
        
        return x, y, w
    
    def __data_generation(self, idx):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        i = 0
        x = np.zeros((self.batch_size,self.img_size[0],self.img_size[1],1))
        y_class = np.zeros((self.batch_size,self.img_size[0],self.img_size[1],1))
        y_inst = np.zeros((self.batch_size,self.img_size[0],self.img_size[1],1))
        w = np.zeros((self.batch_size,self.img_size[0],self.img_size[1],1))
        
        while i < self.batch_size:
            
            part = random.randint(48,64)
            
            image = np.zeros((1320,1640))    
            mask_class =  np.zeros((1320,1640)) 
            mask_instance =  np.zeros((1320,1640)) 
            
            selection = self.get_pollens(self.input_img_paths,number_examples = part)
            
            image,mask_class,mask_instance = self.add_pollen(image,mask_class,mask_instance,selection,self.value_dict)
                
                
            x[i,:,:,0] = image[180:1320-180,180:1640-180]
            y_class[i,:,:,0] = mask_class[180:1320-180,180:1640-180]
            y_inst[i,:,:,0] = mask_instance[180:1320-180,180:1640-180]
            
            i+=1
        w += 0.5
        w[np.where(y_class>0)]=1
        # w[np.where(y>1)]=2
        return tf.convert_to_tensor(x/255.),[ tf.convert_to_tensor(y_class),tf.convert_to_tensor(y_inst)], tf.convert_to_tensor(w)
     
    def get_pollens(self,pollen_dict,number_examples = 10):
        
        
        keys = list(pollen_dict.keys())
        ret_particles = []
        
        while len(ret_particles) < number_examples:
            key = np.random.choice(keys,)
            ret_particles.append([key,random.choice(pollen_dict[key])])
            
        # for i in range( np.random.randint(0,5)):
        #     ret_particles.append(["alternaria",random.choice(pollen_dict['alternaria'])])# Force to have at least one alternaria particle
       
        return ret_particles
    

    def add_pollen(self,current_image,current_mask,mask_instance,particles,value_dict):

        for idx,particle in enumerate(particles):
            key, path = particle
            
            y_min = random.randint(0, 1280)
            y_max = y_min + 360
            x_min = random.randint(0, 960)
            x_max = x_min + 360
            img = load_img(path, 
                        target_size=(360,360),
                        color_mode="grayscale")
            img = np.array(img)
            if not self.validation: 
                if self.augment:
                    flipud = np.random.random(1) > 0.5
                    fliplr = np.random.random(1) > 0.5
                    if np.random.random(1) > 0.75:
                        rotate_angle = np.random.randint(0,360,1)[0]
                    else:
                        rotate_angle = 0
                img = self.augment(img,flipud,fliplr,rotate_angle)
            mask = ( img > 0 )
            reverse_mask = np.logical_not(mask)
            value_mask = mask * value_dict[key]
            current_image[x_min:x_max,y_min:y_max] = current_image[x_min:x_max,y_min:y_max] * reverse_mask + img
            current_mask[x_min:x_max,y_min:y_max] = current_mask[x_min:x_max,y_min:y_max] * reverse_mask + value_mask
            mask_erroded = erosion(mask,selem=disk(5))
            mask_instance[x_min:x_max,y_min:y_max] = mask_instance[x_min:x_max,y_min:y_max] * reverse_mask + mask_erroded

        return current_image, current_mask,mask_instance
    
    def augment(self,img,flipud,fliplr,rotate_angle):
        temp_x = np.array(img)
        if rotate_angle:
                temp_x = np.around(rotate(temp_x,rotate_angle,reshape=False))
                temp_x[np.where(temp_x<0)] = 0
    
        if flipud:
            temp_x = np.flip(temp_x,axis=0)
        if fliplr:
            temp_x = np.flip(temp_x,axis=1)
            
        return temp_x 
class Pollen_synthetic_v2(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self,
                 batch_size,
                 step_per_epoch,
                 img_size,
                 input_img_paths,
                 value_dict,
                 background_path=None,
                 validation=False):
        self.batch_size = batch_size
        self.step_per_epoch = step_per_epoch
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.value_dict = value_dict
        self.background_path = background_path
        self.validation = validation

    def __len__(self):
        return self.step_per_epoch

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""

        # Generate data
        if self.validation:
            idx_seed = idx
        else:
            idx_seed = np.random.randint(0,913829128)
            
        random.seed(idx_seed)  
        # np.random.seed(idx)
        x, y, w = self.__data_generation(idx_seed)
        
        return x, y, w
    
    def __data_generation(self, idx):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        i = 0
        x = np.zeros((self.batch_size,self.img_size[0],self.img_size[1],1))
        y = np.zeros((self.batch_size,self.img_size[0],self.img_size[1],1))
        w = np.zeros((self.batch_size,self.img_size[0],self.img_size[1],1))
        random.seed(idx)
        # np.random.seed(idx)
        if self.background_path:
            paths = [os.path.join(self.background_path,file_name) for file_name in os.listdir(self.background_path)]
        while i < self.batch_size:
            
            part = random.randint(20,48)
            
            image = np.zeros((1320,1640))
            mask =  np.zeros((1320,1640))

            if self.background_path and random.random() > 0.9 and not self.validation:
                
                back_path = random.choice(paths)
                background = load_img(back_path, target_size=(960,1280),color_mode="grayscale")
                background = np.array(background)
                flipud = random.random() > 0.5
                fliplr = random.random() > 0.5
                if random.random() > 0.75:
                    rotate_angle = random.randint(0,360)
                else:
                    rotate_angle = 0
                background = self.augment(background,flipud,fliplr,rotate_angle)
                background_mask = background > 0
                background_mask = background_mask * self.value_dict["junk"]
                image[180:1320-180,180:1640-180] += background
                mask[180:1320-180,180:1640-180] += background_mask
                part = random.randint(8,20)


            selection = self.get_pollens(self.input_img_paths,number_examples=part,seed=idx)
            
            image,mask = self.add_pollen(image,mask,selection,self.value_dict,seed=idx)
                
                
            x[i,:,:,0] = image[180:1320-180,180:1640-180]
            y[i,:,:,0] = mask[180:1320-180,180:1640-180]
            
            i+=1
        w += 0.5
        w[np.where(y>0)]=1
        w[np.where(y==11)]=0.5

        return tf.convert_to_tensor(x/255.), tf.convert_to_tensor(y), tf.convert_to_tensor(w)
     
    def get_pollens(self,pollen_dict,number_examples = 10,seed=10):
        random.seed(seed)
        # np.random.seed(seed)
        keys = list(pollen_dict.keys())
        ret_particles = []
        
        while len(ret_particles) < number_examples:
            key = random.choice(keys,)
            ret_particles.append([key,random.choice(pollen_dict[key])])
            
        # for i in range(np.random.randint(0,5)):
        #     ret_particles.append(["alternaria",random.choice(pollen_dict['alternaria'])])# Force to have at least one alternaria particle
       
        return ret_particles
    

    def add_pollen(self,current_image,current_mask,particles,value_dict,seed=10):
        random.seed(seed)
        # np.random.seed(seed)
        for idx,particle in enumerate(particles):
            key, path = particle
            
            y_min = random.randint(0, 1280)
            y_max = y_min + 360
            x_min = random.randint(0, 960)
            x_max = x_min + 360
            img = load_img(path, 
                        target_size=(360,360),
                        color_mode="grayscale")
            img = np.array(img)
            if not self.validation: 
                flipud = random.random() > 0.5
                fliplr = random.random() > 0.5
                if random.random() > 0.75:
                    rotate_angle = random.randint(0,360)
                else:
                    rotate_angle = 0
                img = self.augment(img,flipud,fliplr,rotate_angle)
            mask = ( img > 0 )
            reverse_mask = np.logical_not(mask)
            value_mask = mask * value_dict[key]
            current_image[x_min:x_max,y_min:y_max] = current_image[x_min:x_max,y_min:y_max] * reverse_mask + img
            current_mask[x_min:x_max,y_min:y_max] = current_mask[x_min:x_max,y_min:y_max] * reverse_mask + value_mask

        return current_image, current_mask
    
    def augment(self,img,flipud,fliplr,rotate_angle):
        temp_x = np.array(img)
        if rotate_angle:
                temp_x = np.around(rotate(temp_x,rotate_angle,reshape=False))
                temp_x[np.where(temp_x<0)] = 0
    
        if flipud:
            temp_x = np.flip(temp_x,axis=0)
        if fliplr:
            temp_x = np.flip(temp_x,axis=1)
            
        return temp_x