from numpy.random import seed
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
import keras
from keras.preprocessing import sequence
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout,Input
from keras.layers import Conv1D, BatchNormalization
from tensorflow.keras.utils import to_categorical
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from __future__ import division
from __future__ import print_function
import math
import sys
import time
import argparse
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
import tensorflow as tf
import keras
import keras.backend as K
from keras.models import Sequential,load_model
from keras.layers import Dense,LSTM,Dropout,Conv1D, BatchNormalization,Activation
from tensorflow.keras.utils import to_categorical
from keras.preprocessing import sequence
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint,TensorBoard,LearningRateScheduler
from pyts.datasets import uea_dataset_list,fetch_uea_dataset,fetch_ucr_dataset,uea_dataset_info
from pyts.metrics import *
from sklearn.neighbors import KernelDensity
import argparse
from collections import defaultdict, OrderedDict
import json
import numpy as np
import scipy.sparse as sp
import sklearn
import sklearn.metrics
import torch
import pandas as pd
import random
from pyts.datasets import ucr_dataset_list
from pyts.datasets import uea_dataset_list,fetch_uea_dataset,uea_dataset_info
from pyts.metrics import boss
import tensorflow_addons as tfa

def build_mlp(input_shape,nb_classes):
    x = keras.layers.Input(shape=(input_shape))
    #a Layer instance is callable on a tensor , and returns a tensor
    x_nb = Dense(64 , activation='relu')(x)
    x_nb = keras.layers.Dropout(0.2)(x_nb)       
    out = Dense(nb_classes,activation='softmax')(x_nb)
    return x, out

def build_lstm(input_shape,nb_classes):
    x = keras.layers.Input(shape=(input_shape))
    x_nb = LSTM(64)(x)
    x_nb = keras.layers.Dropout(0.2)(x_nb)       
    out = Dense(nb_classes,activation='softmax')(x_nb)
    return x, out

def build_fcn(input_shape,nb_classes):
    x = keras.layers.Input(shape=(input_shape))
    conv_x = keras.layers.BatchNormalization()(x)
    conv_x = keras.layers.Conv1D(128, kernel_size=8, padding='same')(conv_x)
    # conv_x = keras.layers.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation('relu')(conv_x)
    conv_x = keras.layers.Dropout(0.2)(conv_x)       
    full = keras.layers.GlobalAveragePooling1D()(conv_x)
    out = keras.layers.Dense(nb_classes, activation='softmax')(full)
    return x, out
def build_resnet(input_shape, n_feature_maps, nb_classes):
    #print('build conv_x')
    x = keras.layers.Input(shape=(input_shape))
    conv_x = keras.layers.BatchNormalization()(x)
    conv_x = keras.layers.Conv1D(n_feature_maps, kernel_size=8, padding='same')(conv_x)
    conv_x = keras.layers.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation('relu')(conv_x)

    #print('build conv_y')
    conv_y = keras.layers.Conv1D(n_feature_maps, kernel_size=5, padding='same')(conv_x)
    conv_y = keras.layers.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation('relu')(conv_y)

    #print('build conv_z')
    conv_z = keras.layers.Conv1D(n_feature_maps, kernel_size=3, padding='same')(conv_y)
    conv_z = keras.layers.BatchNormalization()(conv_z)

    is_expand_channels = not (input_shape[-1] == n_feature_maps)
    if is_expand_channels:
        shortcut_y = keras.layers.Conv1D(n_feature_maps, kernel_size=1, padding='same')(x)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)
    else:
        shortcut_y = keras.layers.BatchNormalization()(x)
    #print('Merging skip connection')
    y = keras.layers.Add()([shortcut_y, conv_z])
    y = keras.layers.Activation('relu')(y)

    #print('build conv_x')
    x1 = y
    conv_x = keras.layers.Conv1D(n_feature_maps * 2, kernel_size=8, padding='same')(x1)
    conv_x = keras.layers.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation('relu')(conv_x)

    #print('build conv_y')
    conv_y = keras.layers.Conv1D(n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
    conv_y = keras.layers.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation('relu')(conv_y)

    #print('build conv_z')
    conv_z = keras.layers.Conv1D(n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
    conv_z = keras.layers.BatchNormalization()(conv_z)

    is_expand_channels = not (input_shape[-1] == n_feature_maps * 2)
    if is_expand_channels:
        shortcut_y = keras.layers.Conv1D(n_feature_maps * 2, kernel_size=1, padding='same')(x1)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)
    else:
        shortcut_y = keras.layers.BatchNormalization()(x1)
    #print('Merging skip connection')
    y = keras.layers.Add()([shortcut_y, conv_z])
    y = keras.layers.Activation('relu')(y)

    #print('build conv_x')
    x1 = y
    conv_x = keras.layers.Conv1D(n_feature_maps * 2, kernel_size=8, padding='same')(x1)
    conv_x = keras.layers.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation('relu')(conv_x)

    #print('build conv_y')
    conv_y = keras.layers.Conv1D(n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
    conv_y = keras.layers.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation('relu')(conv_y)

    #print('build conv_z')
    conv_z = keras.layers.Conv1D(n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
    conv_z = keras.layers.BatchNormalization()(conv_z)

    is_expand_channels = not (input_shape[-1] == n_feature_maps * 2)
    if is_expand_channels:
        shortcut_y = keras.layers.Conv1D(n_feature_maps * 2, kernel_size=1, padding='same')(x1)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)
    else:
        shortcut_y = keras.layers.BatchNormalization()(x1)
    #print('Merging skip connection')
    y = keras.layers.Add()([shortcut_y, conv_z])
    y = keras.layers.Activation('relu')(y)

    full = keras.layers.GlobalAveragePooling1D()(y)
    out = keras.layers.Dense(nb_classes, activation='softmax')(full)
    #print('        -- model was built.')
    return x, out
def build_encoder(input_shape, nb_classes):
    x = keras.layers.Input(input_shape)

    # conv block -1
    conv1 = keras.layers.Conv1D(filters=128,kernel_size=5,strides=1,padding='same')(x)
    conv1 = tfa.layers.InstanceNormalization()(conv1)
    conv1 = keras.layers.PReLU(shared_axes=[1])(conv1)
    conv1 = keras.layers.Dropout(rate=0.2)(conv1)
    conv1 = keras.layers.MaxPooling1D(pool_size=2)(conv1)
    # conv block -2
    conv2 = keras.layers.Conv1D(filters=256,kernel_size=11,strides=1,padding='same')(conv1)
    conv2 = tfa.layers.InstanceNormalization()(conv2)
    conv2 = keras.layers.PReLU(shared_axes=[1])(conv2)
    conv2 = keras.layers.Dropout(rate=0.2)(conv2)
    conv2 = keras.layers.MaxPooling1D(pool_size=2)(conv2)
    # conv block -3
    conv3 = keras.layers.Conv1D(filters=512,kernel_size=21,strides=1,padding='same')(conv2)
    conv3 = tfa.layers.InstanceNormalization()(conv3)
    conv3 = keras.layers.PReLU(shared_axes=[1])(conv3)
    conv3 = keras.layers.Dropout(rate=0.2)(conv3)
    # split for attention
    attention_data = keras.layers.Lambda(lambda x: x[:,:,:256])(conv3)
    attention_softmax = keras.layers.Lambda(lambda x: x[:,:,256:])(conv3)
    # attention mechanism
    attention_softmax = keras.layers.Softmax()(attention_softmax)
    multiply_layer = keras.layers.Multiply()([attention_softmax,attention_data])
    # last layer
    dense_layer = keras.layers.Dense(units=256,activation='sigmoid')(multiply_layer)
    dense_layer = tfa.layers.InstanceNormalization()(dense_layer)
    # output layer
    flatten_layer = keras.layers.Flatten()(dense_layer)
    out = keras.layers.Dense(units=nb_classes,activation='softmax')(flatten_layer)

    return x,out
