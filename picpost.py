# coding: utf-8
import json as js
import os

import os
import sys
import random
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import post

# Set some parameters
im_width = 2592
im_height = 2048
im_chan = 1
path_train = '../testlv/'
path_test = '../testlv/'
scare_factor = 1 

train_ids = next(os.walk(path_train + 'images/'))[2][:3]

def binary_crossentropy_with_factor(y_true,y_pred):
    y_true_factor = tf.to_float(y_true > 0.5) * 1000 + 1
    loss_binary_crossentropy = K.binary_crossentropy(y_true, y_pred)
    return K.mean(np.multiply(y_true_factor,loss_binary_crossentropy),axis=-1)
# Define IoU metric
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)


# Get and resize train images and masks
X_train = np.zeros((len(train_ids), int(im_height/scare_factor), int(im_width/scare_factor), im_chan), dtype=np.uint8)
Y_train = np.zeros((len(train_ids), int(im_height/scare_factor), int(im_width/scare_factor), 1), dtype=np.uint8)
print('Getting and resizing train images and masks ... ')
sys.stdout.flush()
for n, id_ in enumerate(train_ids):
    print('train_id  ' + str(n))
    path = path_train
    img = load_img(path + 'images/' + id_)
    x = img_to_array(img)[:,:,1].astype('uint8')
    x = resize(x, (im_height/scare_factor, im_width/scare_factor, 1), mode='constant', preserve_range=True)
    X_train[n] = x
    mask = img_to_array(load_img('../testlv/masks/' + id_[:-4] + '_1.png'))[:,:,1].astype('bool')
    Y_train[n] = resize(mask, (im_height/scare_factor, im_width/scare_factor, 1), mode='constant', preserve_range=True) 

print('Done!')


# Predict on train, val and test
print('loading model...')
model = load_model('model_factor_conver.h5', custom_objects={'mean_iou': mean_iou})
model.summary()
true_count = 0
total_count = 0
zhaohui_num = 0
for index,item in enumerate(X_train):
    
    print(index,'pic is loading...')
    preds_train = model.predict([[item]], verbose=1)
    # Threshold predictions
    preds_train_t = (preds_train > 0.5).astype(np.uint8) * 255
    for i in preds_train_t:
        #print(np.squeeze(Y_train[index] * 255).shape)
        ture_rate,count,zhaohui = post.get_iou_by_box(np.squeeze(Y_train[index] * 255),np.squeeze(i))
        true_count += count * ture_rate
        total_count += count
        zhaohui_num += zhaohui
        #print(ture_rate)
    print('plot pic...')
    fig1 = plt.figure(figsize=(133,25.6))
    plt.subplot(1,4,1)
    plt.imshow(np.dstack((X_train[index],X_train[index],X_train[index])))
    #print(X_train[index].shape)
    tmp = np.squeeze(Y_train[index]).astype(np.float32)
    plt.subplot(1,4,2)
    plt.imshow(np.dstack((tmp,tmp,tmp)))
    
    x_train_ori = X_train[index]

    tmp = np.squeeze(preds_train_t[0])
    tmp,X_train[index] = post.lvbo(tmp,X_train[index])
    plt.subplot(1,4,3)
    plt.imshow(np.dstack((tmp,tmp,tmp)))

    plt.subplot(1,4,4)
    plt.imshow(np.dstack((X_train[index],X_train[index],X_train[index])))
    
    fig1.savefig('../lv/'+str(index)+'.png')
    plt.close()
print('准确率:',true_count * 1.0 / total_count)
print('召回率:',true_count * 1.0 / zhaohui_num)
print('true_count:',true_count)
print('total_count:',total_count)
print('zhaohui_num:',zhaohui_num)