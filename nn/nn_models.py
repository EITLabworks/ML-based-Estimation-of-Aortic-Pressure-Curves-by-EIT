from numpy.random import seed
seed(1)
from tensorflow.random import set_seed
set_seed(2)
from tensorflow.keras.layers import Input, Dense, Dropout, Normalization, Activation, Conv2D, MaxPooling2D, Flatten, BatchNormalization
import argparse
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import json
from sklearn.model_selection import train_test_split
from nn.util import load_preprocess_examples
from nn.util_paras import load_preprocess_paras, write_configs, reload_aorta_segs_from_piginfo,check_resampled_paras, normalize_aorta, reload_ventparas_from_piginfo
from tensorflow import distribute as dist
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
#from parametrization.reconstruction import reconstruct_lin_tensors_block, reconstruct_lin_uncorrected_tensors_block
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from nn.evaluation_fcts import *
import numpy as np
import os
from src.reconstruction import rescalingfactors_for_cl, rescale_paras, reorder_aorta_paras
from nn.eva_metrics import EvaMetrics
from nn.help_function_training import *
import warnings
import psutil
process=psutil.Process()
warnings.filterwarnings('ignore')
import gc
import math
set_global_determinism(26)



def model_original(input_shape=(64, 1024, 1), latent_dim=42, ConfigParas=None):

    pattern = [4,4,4,4]
    pattern_time= [2,2,1,1]
    mapper_input = Input(shape=input_shape)
    x = mapper_input

    # convolutional layers
    x = Conv2D(8, 5, strides=(pattern_time[0], pattern[0]), padding="same")(x)
    # x = MaxPooling2D(pool_size=(1,2))(x)
    if ConfigParas.bBatchNorm:
        x = BatchNormalization()(x)
    x = Activation("elu")(x)
    if ConfigParas.bDropout:
        x = Dropout(ConfigParas.fNumDrop)(x)

    x = Conv2D(8, 5, strides=(pattern_time[1], pattern[1]), padding="same")(x)
    #     x = MaxPooling2D(pool_size=(1,2))(x)
    if ConfigParas.bBatchNorm:
        x = BatchNormalization()(x)
    x = Activation("elu")(x)
    if ConfigParas.bDropout:
        x = Dropout(ConfigParas.fNumDrop)(x)


    x = Conv2D(6, 5, strides=(pattern_time[2], pattern[2]), padding="same")(x)
    x = MaxPooling2D(pool_size=(pattern_time[3], pattern[3]))(x)
    #  x = BatchNormalization()(x)
    if ConfigParas.bBatchNorm:
        x = BatchNormalization()(x)
    x = Activation(ConfigParas.actiConv)(x)
    if ConfigParas.bDropout:
        x = Dropout(ConfigParas.fNumDrop)(x)

    x = Flatten()(x)        #-> 384


 #   if ConfigParas.bMoreDense:
     #       x = Dense(400, activation=ConfigParas.actiDense)(x)  # elu
    #        x = Dense(300, activation=ConfigParas.actiDense)(x)  # elu #hinzugefügt
   #         x = Dense(200, activation=ConfigParas.actiDense)(x)  # elu #hinzugefügt
    x = Dense(126, activation="elu")(x)# elu #hinzugefügt
    x = Dense(84, activation="elu")(x)   # elu
    mapper_output = Dense(latent_dim, activation="relu")(x)  ##linear
    return Model(mapper_input, mapper_output)



def model_no_padding(input_shape=(64, 1024, 1), latent_dim=42, ConfigParas=None):
    pattern = [4,4,4,4]
    pattern_time= [2,2,1,1]
    mapper_input = Input(shape=input_shape)
    x = mapper_input
    # convolutional layers
    x = Conv2D(8, 5, strides=(pattern_time[0], pattern[0]), padding="valid")(x)
    x = Activation("elu")(x)
    if ConfigParas.bDropout:
        x = Dropout(ConfigParas.fNumDrop)(x)
    x = Conv2D(8, 5, strides=(pattern_time[1], pattern[1]), padding="valid")(x)
    x = Activation("elu")(x)
    if ConfigParas.bDropout:
        x = Dropout(ConfigParas.fNumDrop)(x)
    x = Conv2D(6, 5, strides=(pattern_time[2], pattern[2]), padding="valid")(x)
    x = MaxPooling2D(pool_size=(pattern_time[3], pattern[3]))(x)
    x = Activation(ConfigParas.actiConv)(x)
    if ConfigParas.bDropout:
        x = Dropout(ConfigParas.fNumDrop)(x)
    x = Flatten()(x)        #-> 384
    x = Dense(126, activation="elu")(x)# elu #hinzugefügt
    x = Dense(84, activation="elu")(x)   # elu
    mapper_output = Dense(latent_dim, activation="relu")(x)  ##linear
    return Model(mapper_input, mapper_output)


def model_maxpool_instead_stride(input_shape=(64, 1024, 1), latent_dim=42, ConfigParas=None):

    pattern = [4,4,4,4]
    pattern_time= [2,2,1,1]
    mapper_input = Input(shape=input_shape)
    x = mapper_input

    # convolutional layers
    x = Conv2D(8, 5, strides=(1,1), padding="same")(x)
    if ConfigParas.bBatchNorm:
        x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2,4))(x)
    x = Activation("elu")(x)
    if ConfigParas.bDropout:
        x = Dropout(ConfigParas.fNumDrop)(x)

    x = Conv2D(8, 5, strides=(1,1), padding="same")(x)
    if ConfigParas.bBatchNorm:
        x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2,4))(x)
    x = Activation("elu")(x)
    if ConfigParas.bDropout:
        x = Dropout(ConfigParas.fNumDrop)(x)


    x = Conv2D(6, 5, strides=(1,4), padding="same")(x)
    if ConfigParas.bBatchNorm:
        x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(1,4))(x)
    x = Activation(ConfigParas.actiConv)(x)
    if ConfigParas.bDropout:
        x = Dropout(ConfigParas.fNumDrop)(x)

    x = Flatten()(x)        #-> 384
    x = Dense(126, activation="elu")(x)# elu #hinzugefügt
    x = Dense(84, activation="elu")(x)   # elu
    mapper_output = Dense(latent_dim, activation="relu")(x)  ##linear
    return Model(mapper_input, mapper_output)



def model_more_conv(input_shape=(64, 1024, 1), latent_dim=42, ConfigParas=None):

    pattern = [4,4,4,4]
    pattern_time= [2,2,1,1]
    mapper_input = Input(shape=input_shape)
    x = mapper_input

    # convolutional layers
    x = Conv2D(8, 5, strides=(pattern_time[0], pattern[0]), padding="same")(x)
    # x = MaxPooling2D(pool_size=(1,2))(x)
    if ConfigParas.bBatchNorm:
        x = BatchNormalization()(x)
    x = Activation("elu")(x)
    if ConfigParas.bDropout:
        x = Dropout(ConfigParas.fNumDrop)(x)

    x = Conv2D(8, 5, strides=(pattern_time[1], pattern[1]), padding="same")(x)
    #     x = MaxPooling2D(pool_size=(1,2))(x)
    if ConfigParas.bBatchNorm:
        x = BatchNormalization()(x)
    x = Activation("elu")(x)
    if ConfigParas.bDropout:
        x = Dropout(ConfigParas.fNumDrop)(x)


    x = Conv2D(6, 5, strides=(1, 2), padding="same")(x)
    x = Activation(ConfigParas.actiConv)(x)
    if ConfigParas.bDropout:
        x = Dropout(ConfigParas.fNumDrop)(x)
    x = Conv2D(6, 5, strides=(1, 2), padding="same")(x)
    x = Activation(ConfigParas.actiConv)(x)
    if ConfigParas.bDropout:
        x = Dropout(ConfigParas.fNumDrop)(x)
    x = MaxPooling2D(pool_size=(pattern_time[3], pattern[3]))(x)
    x = Flatten()(x)        #-> 384

    x = Dense(126, activation="elu")(x)# elu #hinzugefügt
    x = Dense(84, activation="elu")(x)   # elu
    mapper_output = Dense(latent_dim, activation="relu")(x)  ##linear
    return Model(mapper_input, mapper_output)


def model_diffpos_maxpool(input_shape=(64, 1024, 1), latent_dim=42, ConfigParas=None):

    pattern = [4,4,4,4]
    pattern_time= [2,2,1,1]
    mapper_input = Input(shape=input_shape)
    x = mapper_input

    # convolutional layers
    x = Conv2D(8, 5, strides=(pattern_time[0], pattern[0]), padding="same")(x)
    # x = MaxPooling2D(pool_size=(1,2))(x)
    if ConfigParas.bBatchNorm:
        x = BatchNormalization()(x)
    x = Activation("elu")(x)
    if ConfigParas.bDropout:
        x = Dropout(ConfigParas.fNumDrop)(x)

    x = Conv2D(8, 5, strides=(pattern_time[1], pattern[1]), padding="same")(x)
    #     x = MaxPooling2D(pool_size=(1,2))(x)
    if ConfigParas.bBatchNorm:
        x = BatchNormalization()(x)
    x = Activation("elu")(x)
    if ConfigParas.bDropout:
        x = Dropout(ConfigParas.fNumDrop)(x)


    x = Conv2D(6, 5, strides=(pattern_time[2], pattern[2]), padding="same")(x)
    #  x = BatchNormalization()(x)
    if ConfigParas.bBatchNorm:
        x = BatchNormalization()(x)
    x = Activation(ConfigParas.actiConv)(x)
    if ConfigParas.bDropout:
        x = Dropout(ConfigParas.fNumDrop)(x)
    x = MaxPooling2D(pool_size=(pattern_time[3], pattern[3]))(x)
    x = Flatten()(x)        #-> 384
 #   if ConfigParas.bMoreDense:
     #       x = Dense(400, activation=ConfigParas.actiDense)(x)  # elu
    #        x = Dense(300, activation=ConfigParas.actiDense)(x)  # elu #hinzugefügt
   #         x = Dense(200, activation=ConfigParas.actiDense)(x)  # elu #hinzugefügt
    x = Dense(126, activation="elu")(x)# elu #hinzugefügt
    x = Dense(84, activation="elu")(x)   # elu
    mapper_output = Dense(latent_dim, activation="relu")(x)  ##linear
    return Model(mapper_input, mapper_output)




def model_lessdense(input_shape=(64, 1024, 1), latent_dim=42, ConfigParas=None):
    pattern = [4,4,4,4]
    pattern_time= [2,2,1,1]
    mapper_input = Input(shape=input_shape)
    # convolutional layers
    x = Conv2D(8, 5, strides=(pattern_time[0], pattern[0]), padding="same")(mapper_input)
    x = Activation("elu")(x)
    if ConfigParas.bDropout:
        x = Dropout(ConfigParas.fNumDrop)(x)

    x = Conv2D(8, 5, strides=(pattern_time[1], pattern[1]), padding="same")(x)
    x = Activation("elu")(x)
    if ConfigParas.bDropout:
        x = Dropout(ConfigParas.fNumDrop)(x)

    x = Conv2D(6, 5, strides=(pattern_time[2], pattern[2]), padding="same")(x)
    x = MaxPooling2D(pool_size=(pattern_time[3], pattern[3]))(x)
    x = Activation(ConfigParas.actiConv)(x)
    if ConfigParas.bDropout:
        x = Dropout(ConfigParas.fNumDrop)(x)
    x = Flatten()(x)        #-> 384
    x = Dense(126, activation="elu")(x)# elu #hinzugefügt
    mapper_output = Dense(latent_dim, activation="relu")(x)  ##linear
    return Model(mapper_input, mapper_output)


def model_original_slimmer1(input_shape=(64, 1024, 1), latent_dim=42, ConfigParas=None):

    pattern = [4,4,4,4]
    pattern_time= [2,2,1,1]
    mapper_input = Input(shape=input_shape)
    x = mapper_input

    # convolutional layers
    x = Conv2D(8, 5, strides=(pattern_time[0], pattern[0]), padding="same")(x)

    if ConfigParas.bBatchNorm:
        x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(1,2))(x)
    x = Activation("elu")(x)
    if ConfigParas.bDropout:
        x = Dropout(ConfigParas.fNumDrop)(x)

    x = Conv2D(8, 5, strides=(pattern_time[1], pattern[1]), padding="same")(x)
    #     x = MaxPooling2D(pool_size=(1,2))(x)
    if ConfigParas.bBatchNorm:
        x = BatchNormalization()(x)
    x = Activation("elu")(x)
    if ConfigParas.bDropout:
        x = Dropout(ConfigParas.fNumDrop)(x)
    x = Conv2D(6, 5, strides=(pattern_time[2], pattern[2]), padding="same")(x)
    x = MaxPooling2D(pool_size=(pattern_time[3], pattern[3]))(x)
    #  x = BatchNormalization()(x)
    if ConfigParas.bBatchNorm:
        x = BatchNormalization()(x)
    x = Activation(ConfigParas.actiConv)(x)
    if ConfigParas.bDropout:
        x = Dropout(ConfigParas.fNumDrop)(x)

    x = Flatten()(x)        #-> 384
    x = Dense(126, activation="elu")(x)# elu #hinzugefügt
    x = Dense(84, activation="elu")(x)   # elu
    mapper_output = Dense(latent_dim, activation="relu")(x)  ##linear
    return Model(mapper_input, mapper_output)

def model_original_slimmer2(input_shape=(64, 1024, 1), latent_dim=42, ConfigParas=None):

    pattern = [4,2,4,4]
    pattern_time= [2,2,1,1]
    mapper_input = Input(shape=input_shape)
    x = mapper_input

    # convolutional layers
    x = Conv2D(8, 5, strides=(pattern_time[0], pattern[0]), padding="same")(x)

    if ConfigParas.bBatchNorm:
        x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(1,2))(x)
    x = Activation("elu")(x)
    if ConfigParas.bDropout:
        x = Dropout(ConfigParas.fNumDrop)(x)

    x = Conv2D(6, 5, strides=(pattern_time[1], pattern[1]), padding="same")(x)
    x = MaxPooling2D(pool_size=(1,2))(x)
    if ConfigParas.bBatchNorm:
        x = BatchNormalization()(x)
    x = Activation("elu")(x)
    if ConfigParas.bDropout:
        x = Dropout(ConfigParas.fNumDrop)(x)
    x = Conv2D(4, 5, strides=(pattern_time[2], pattern[2]), padding="same")(x)
    x = MaxPooling2D(pool_size=(pattern_time[3], pattern[3]))(x)
    #  x = BatchNormalization()(x)
    if ConfigParas.bBatchNorm:
        x = BatchNormalization()(x)
    x = Activation(ConfigParas.actiConv)(x)
    if ConfigParas.bDropout:
        x = Dropout(ConfigParas.fNumDrop)(x)

    x = Flatten()(x)        #-> 384
    x = Dense(126, activation="elu")(x)# elu #hinzugefügt
    x = Dense(84, activation="elu")(x)   # elu
    mapper_output = Dense(latent_dim, activation="relu")(x)  ##linear
    return Model(mapper_input, mapper_output)



def model_maxpool_instead_stride_slimmer1(input_shape=(64, 1024, 1), latent_dim=42, ConfigParas=None):

    pattern = [4,4,4,4]
    pattern_time= [2,2,1,1]
    mapper_input = Input(shape=input_shape)
    x = mapper_input

    # convolutional layers
    x = Conv2D(8, 5, strides=(1,2), padding="same")(x)
    if ConfigParas.bBatchNorm:
        x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2,4))(x)
    x = Activation("elu")(x)
    if ConfigParas.bDropout:
        x = Dropout(ConfigParas.fNumDrop)(x)

    x = Conv2D(8, 5, strides=(1,1), padding="same")(x)
    if ConfigParas.bBatchNorm:
        x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2,4))(x)
    x = Activation("elu")(x)
    if ConfigParas.bDropout:
        x = Dropout(ConfigParas.fNumDrop)(x)


    x = Conv2D(6, 5, strides=(1,4), padding="same")(x)
    if ConfigParas.bBatchNorm:
        x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(1,4))(x)
    x = Activation(ConfigParas.actiConv)(x)
    if ConfigParas.bDropout:
        x = Dropout(ConfigParas.fNumDrop)(x)

    x = Flatten()(x)        #-> 384
    x = Dense(126, activation="elu")(x)# elu #hinzugefügt
    x = Dense(84, activation="elu")(x)   # elu
    mapper_output = Dense(latent_dim, activation="relu")(x)  ##linear
    return Model(mapper_input, mapper_output)


def model_maxpool_instead_stride_slimmer2(input_shape=(64, 1024, 1), latent_dim=42, ConfigParas=None):

    pattern = [4,4,4,4]
    pattern_time= [2,2,1,1]
    mapper_input = Input(shape=input_shape)
    x = mapper_input

    # convolutional layers
    x = Conv2D(8, 5, strides=(1,2), padding="same")(x)
    if ConfigParas.bBatchNorm:
        x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2,4))(x)
    x = Activation("elu")(x)
    if ConfigParas.bDropout:
        x = Dropout(ConfigParas.fNumDrop)(x)

    x = Conv2D(6, [5,9], strides=(1,1), padding="same")(x)
    if ConfigParas.bBatchNorm:
        x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2,4))(x)
    x = Activation("elu")(x)
    if ConfigParas.bDropout:
        x = Dropout(ConfigParas.fNumDrop)(x)


    x = Conv2D(6, 5, strides=(1,4), padding="same")(x)
    if ConfigParas.bBatchNorm:
        x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(1,4))(x)
    x = Activation(ConfigParas.actiConv)(x)
    if ConfigParas.bDropout:
        x = Dropout(ConfigParas.fNumDrop)(x)

    x = Flatten()(x)        #-> 384
    x = Dense(126, activation="elu")(x)# elu #hinzugefügt
    x = Dense(84, activation="elu")(x)   # elu
    mapper_output = Dense(latent_dim, activation="relu")(x)  ##linear
    return Model(mapper_input, mapper_output)

def model_kt_tuner1(input_shape=(64, 1024, 1), latent_dim=42, ConfigParas=None):

    pattern = [4,4,4,4]
    pattern_time= [2,2,1,1]
    mapper_input = Input(shape=input_shape)
    x = mapper_input

    # convolutional layers
    x = Conv2D(4, 5, strides=(3,3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)

    x = Conv2D(5, 5, strides=(2, 2), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)

    x = Conv2D(7, 5, strides=(3, 1), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)

    x = Conv2D(9, 5, strides=(2, 1), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)


    x = Flatten()(x)        #-> 384
    x = Dense(84, activation="elu")(x)   # elu
    mapper_output = Dense(latent_dim, activation="relu")(x)  ##linear
    return Model(mapper_input, mapper_output)


model_selection= {"nopadding":model_no_padding, "maxpool":model_maxpool_instead_stride,
                  "moreconv":model_more_conv,
                  "maxpooldiffpos": model_diffpos_maxpool,
                  "lessdense":model_lessdense,
                  "slimmer1":model_original_slimmer1,
                  "slimmer2":model_original_slimmer2,
                  "maxpoolslimmer1":model_maxpool_instead_stride_slimmer1,
                  "maxpoolslimmer2":model_maxpool_instead_stride_slimmer2,
                  "ktpart1":model_kt_tuner1,}