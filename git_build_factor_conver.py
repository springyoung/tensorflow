# coding: utf-8

import os
import sys
import random
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

import model_factor_conver

#gpu
#import keras.backend.tensorflow_backend as KTF
#KTF.set_session(tf.Session(config=tf.ConfigProto(device_count={'gpu':0})))
# config = tf.ConfigProto()  
# config.gpu_options.allow_growth=True   
# session = tf.Session(config=config)
# KTF.set_session(session)

#比例gpu
# import tensorflow as tf
# import keras.backend.tensorflow_backend as KTF

# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.333
# session = tf.Session(config=config)
# KTF.set_session(session)

#no gpu
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# #


# Set some parameters
im_width = 2592
im_height = 2048
im_chan = 1
path_train = '../train/'
path_test = '../test/'
scare_factor = 1
init_con_num = 8
con_kernel = 3
maxpool_kernel = 2

train_ids1 = next(os.walk(path_train+"images"))[2]
train_ids = ['../train/images/' + i for i in train_ids1]
mask_ids = ['../train/masks/' + i[:-4] + '_1.png' for i in train_ids1]


# #check
# check_img = load_img(train_ids[1])
# #tmp = np.squeeze(check_img).astype(np.uint8)
# tmp = img_to_array(check_img)[:,:,1].astype('float32') / 255
# #x = resize(tmp, (im_height/scare_factor, im_width/scare_factor, 1), mode='constant', preserve_range=True)
# #plt.figure()
# #tmp = np.squeeze(x).astype(np.uint8)

# plt.imshow(np.dstack((tmp,tmp,tmp)))
# plt.show()

# # get picArray
def get_pic(paths,cat, im_high = 2048, im_wide = 2592, im_chan=1,scare_factor =1):
    if cat == 'image':
        imgs = np.zeros((len(paths), int(im_height/scare_factor), int(im_width/scare_factor), im_chan), dtype=np.uint8)
        #type_img = 'uint8'
        for n, id_ in enumerate(paths):
            img = load_img(id_)
            x = img_to_array(img)[:,:,1].astype(np.uint8)
            x = resize(x, (im_height/scare_factor, im_width/scare_factor, 1), mode='constant', preserve_range=True)
            imgs[n] = x
            # print('image show...')
            # plt.imshow(np.dstack((imgs[n],imgs[n],imgs[n])))
            # plt.show()
            return imgs
    if cat == 'mask':
        imgs_bool = np.zeros((len(paths), int(im_height/scare_factor), int(im_width/scare_factor), 1), dtype=np.bool)
        imgs_uint8 =np.zeros((len(paths), int(im_height/scare_factor), int(im_width/scare_factor), 1), dtype=np.uint8)
        #type_img = 'bool'
        for n, id_ in enumerate(paths):
            img = load_img(id_)
            x = img_to_array(img)[:,:,1].astype(np.bool)
            x = resize(x, (im_height/scare_factor, im_width/scare_factor, 1), mode='constant', preserve_range=True)
            imgs_uint8[n] = x
            # print('mask showing...')
            # plt.imshow(np.dstack((imgs_uint8[n],imgs_uint8[n],imgs_uint8[n])) * 255 ) 
            # plt.show()
            return imgs_uint8

def get_train_batch(X_train, y_train, batch_size, img_h=2048, img_w=2592, color_type=1,is_argumentation=False):
    while 1:
        for i in range(0, len(X_train), batch_size):
            x = get_pic(X_train[i:i+batch_size], cat='image', im_high = img_h, im_wide = img_w, im_chan = color_type,scare_factor=1)
            y = get_pic(y_train[i:i+batch_size], cat ='mask', im_high = img_h, im_wide = img_w, im_chan = color_type,scare_factor=1)
#            if is_argumentation:
                # 数据增强
#                x, y = img_augmentation(x, y)
            # 最重要的就是这个yield，它代表返回，返回以后循环还是会继续，然后再返回。就比如有一个机器一直在作累加运算，但是会把每次累加中间结果告诉你一样，直到把所有数加完
            yield({'input_1': x}, {'conv2d_19': y})

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


# x = get_pic(train_ids[1:2],'image')
# plt.imshow(np.dstack((x,x,x)))
# plt.show()
def run():
    model = model_factor_conver.unet(input_size=(im_height,im_width,1))
    model.load_weights('./mymodel_weight.h5')
    model.summary()
    print('starting training moedl...')

    earlystopper = EarlyStopping(patience=5, verbose=1)
    checkpointer = ModelCheckpoint('model_factor_conver.h5', verbose=1, monitor='loss', save_best_only=True)
    results = model.fit_generator(generator=get_train_batch(train_ids, mask_ids,batch_size=1, img_h=2048, img_w=2592, color_type=1), steps_per_epoch=1595,epochs=15, \
                    callbacks=[earlystopper, checkpointer])
    #save model 
    model.save('./mymodel_factor_conver.h5')
    model.save_weights('./mymodel_weight.h5')

    return True
run()