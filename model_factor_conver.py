import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np

#from sklearn.model_selection import train_test_split
from skimage.transform import resize
from keras import backend as K

import tensorflow as tf

from keras.preprocessing.image import load_img
from keras import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, Dropout

def binary_crossentropy_with_factor(y_true,y_pred,factor = 110):
    factor_mul = 1.0 / (factor + 1) * factor
    y_true_factor1 = tf.to_float(y_true > 0.5)
    y_true_factor = tf.clip_by_value(y_true_factor1,1-factor_mul,factor_mul)
    loss_binary_crossentropy = K.binary_crossentropy(y_true, y_pred)
    return K.mean(np.multiply(y_true_factor,loss_binary_crossentropy),axis=-1)


def binary_crossentropy_with_factor_conv(y_true,y_pred,factor = 110):
    factor_mul = 1.0 / (factor + 1) * factor
    filter = tf.constant([[1,1,1],  [1,1,1], [1,1,1]],shape=[3,3,1,1],dtype=tf.float32)
    op2 = tf.nn.conv2d(y_true,filter,strides = [1,1,1,1],padding = 'SAME')
    y_true_factor = tf.clip_by_value(op2,(1-factor_mul),factor_mul)
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


def unet(pretrained_weights = None,input_size = (256,256,1),start_neurons = 5,conv_num = 5):
    input_layer = Input(input_size)
    # 128 -> 64
    conv1 = Conv2D(start_neurons * 1, (conv_num,conv_num), activation="relu", padding="same")(input_layer)
    conv1 = Conv2D(start_neurons * 1, (conv_num,conv_num), activation="relu", padding="same")(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    pool1 = Dropout(0.25)(pool1)

    # 64 -> 32
    conv2 = Conv2D(start_neurons * 2, (conv_num,conv_num), activation="relu", padding="same")(pool1)
    conv2 = Conv2D(start_neurons * 2, (conv_num,conv_num), activation="relu", padding="same")(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    pool2 = Dropout(0.5)(pool2)

    # 32 -> 16
    conv3 = Conv2D(start_neurons * 4, (conv_num,conv_num), activation="relu", padding="same")(pool2)
    conv3 = Conv2D(start_neurons * 4, (conv_num,conv_num), activation="relu", padding="same")(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)
    pool3 = Dropout(0.5)(pool3)

    # 16 -> 8
    conv4 = Conv2D(start_neurons * 8, (conv_num,conv_num), activation="relu", padding="same")(pool3)
    conv4 = Conv2D(start_neurons * 8, (conv_num,conv_num), activation="relu", padding="same")(conv4)
    pool4 = MaxPooling2D((2, 2))(conv4)
    pool4 = Dropout(0.5)(pool4)

    # Middle
    convm = Conv2D(start_neurons * 16, (conv_num,conv_num), activation="relu", padding="same")(pool4)
    convm = Conv2D(start_neurons * 16, (conv_num,conv_num), activation="relu", padding="same")(convm)

    # 8 -> 16
    deconv4 = Conv2DTranspose(start_neurons * 8, (conv_num,conv_num), strides=(2, 2), padding="same")(convm)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Dropout(0.5)(uconv4)
    uconv4 = Conv2D(start_neurons * 8, (conv_num,conv_num), activation="relu", padding="same")(uconv4)
    uconv4 = Conv2D(start_neurons * 8, (conv_num,conv_num), activation="relu", padding="same")(uconv4)

    # 16 -> 32
    deconv3 = Conv2DTranspose(start_neurons * 4, (conv_num,conv_num), strides=(2, 2), padding="same")(uconv4)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Dropout(0.5)(uconv3)
    uconv3 = Conv2D(start_neurons * 4, (conv_num,conv_num), activation="relu", padding="same")(uconv3)
    uconv3 = Conv2D(start_neurons * 4, (conv_num,conv_num), activation="relu", padding="same")(uconv3)

    # 32 -> 64
    deconv2 = Conv2DTranspose(start_neurons * 2, (conv_num,conv_num), strides=(2, 2), padding="same")(uconv3)
    uconv2 = concatenate([deconv2, conv2])
    uconv2 = Dropout(0.5)(uconv2)
    uconv2 = Conv2D(start_neurons * 2, (conv_num,conv_num), activation="relu", padding="same")(uconv2)
    uconv2 = Conv2D(start_neurons * 2, (conv_num,conv_num), activation="relu", padding="same")(uconv2)

    # 64 -> 128
    deconv1 = Conv2DTranspose(start_neurons * 1, (conv_num,conv_num), strides=(2, 2), padding="same")(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    uconv1 = Dropout(0.5)(uconv1)
    uconv1 = Conv2D(start_neurons * 1, (conv_num,conv_num), activation="relu", padding="same")(uconv1)
    uconv1 = Conv2D(start_neurons * 1, (conv_num,conv_num), activation="relu", padding="same")(uconv1)

    #uconv1 = Dropout(0.5)(uconv1)
    output_layer = Conv2D(1, (1,1), padding="same", activation="sigmoid")(uconv1)
    #output = Output
    
    model = Model(input = input_layer, output = output_layer)

    model.compile(optimizer = Adam(lr = 1e-5), loss = binary_crossentropy_with_factor_conv, metrics = [mean_iou])
    
    #model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model


