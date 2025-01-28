from tensorflow.keras.layers import Input, Dense, Dropout, Normalization, Activation, Conv2D, MaxPooling2D, Flatten
import argparse
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import json
from sklearn.model_selection import train_test_split
from nn.util import load_preprocess_examples
from nn.util_paras import load_preprocess_paras, write_configs, reload_aorta_segs_from_piginfo, check_resampled_paras
import tensorflow as tf
from src.reconstruction import reconstruct_lin_tensors_block, reconstruct_lin_uncorrected_tensors_block
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from nn.evaluation_fcts import *
import numpy as np
import warnings
import os

warnings.filterwarnings('ignore')


# ------------------------------------------------------------------------------------------------------------------- #
def set_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)


def set_global_determinism(seed):
    set_seeds(seed=seed)

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    tf.config.experimental.enable_op_determinism()
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

    # see also:
    # https://www.tensorflow.org/api_docs/python/tf/config/experimental/enable_op_determinism


def set_global_undeterminism():
   # set_seeds(seed=seed)

    os.environ['TF_DETERMINISTIC_OPS'] = '0'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '0'
 #   tf.config.experimental.enable_op_determinism()
    tf.config.threading.set_inter_op_parallelism_threads(0)
    tf.config.threading.set_intra_op_parallelism_threads(0)

    # see also:
    # https://www.tensorflow.org/api_docs/python/tf/config/experimental/enable_op_determinism


# ------------------------------------------------------------------------------------------------------------------- #
# Array / List modifications
# ------------------------------------------------------------------------------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------- #
def set_array_len(y_truelen, y_change, l:int):
    """
    Changes the length of each subarray to length l and transform another list into an array
    :param y_truelen: This will be transformed into an array [num segs x l]
    :param y_change: The subarrays will be shortened/padded to length l   [num segs x individual lengths]
    :param l: Desired length
    :return: np.array(y_truelen) [num segs x l], np.array(y_change) [num segs x l]
    """
    for k in range(len(y_change)):
        if len(y_change[k]) < l:
            y_change[k] = pad_array(y_change[k], l)
        if len(y_change[k]) > l:
            y_change[k] = y_change[k][:l]
    return np.array(y_truelen), np.array(y_change)


# ------------------------------------------------------------------------------------------------------------------- #
def set_array_len_uneven(y_truelen, y_change, maxlen:int):
    """
    Pad/truncate the segments within two array to the min(max len of y_true, maxlen)
    :param y_truelen: Array with maximal length with segments to be truncated/padded [num segs x individual lengths]
    :param y_change: Array with segments to be truncated/padded [num segs x individual lengths]
    :param maxlen: Maximal length of each segment
    :return: np.array(y_truelen) [num segs x l], np.array(y_change) [num segs x l] with l=min(max_len(y_true_segs), maxlen)
    """
    l = len(y_truelen[0])           # Max len of y_true segments
    for j in range(len(y_truelen)):
        if len(y_truelen[j]) > l:
            l = len(y_truelen[j])

    l = min(maxlen, l)
    for k in range(len(y_truelen)):
        if len(y_truelen[k]) < l:
            y_truelen[k] = pad_array(y_truelen[k], l)
        if len(y_truelen[k]) > l:
            y_truelen[k] = y_truelen[k][:l]

    for k in range(len(y_change)):
        if len(y_change[k]) < l:
            y_change[k] = pad_array(y_change[k], l)
        if len(y_change[k]) > l:
            y_change[k] = y_change[k][:l]

    return np.array(y_truelen), np.array(y_change)


# ------------------------------------------------------------------------------------------------------------------- #
def pad_or_truncate(some_list:list, target_len:int):
    """
    Pad or truncate the segments within a list to target len
    :param some_list: List to be truncated/padded [num segs x individual lengths]
    :param target_len: Target segment length
    :return: some_list [num segs x target_len]
    """
    return some_list[:target_len] + [1] * some_list[-1]*(target_len - len(some_list))


# ------------------------------------------------------------------------------------------------------------------- #
def pad_list(some_list:list, target_len:int):
    """
    Pad the segments within a list to target len
    :param some_list: List to be padded [num segs x individual lengths]
    :param target_len: Target segment length
    :return: some_list [num segs x target_len]
    """
    return some_list + [1] *some_list[-1] *(target_len - len(some_list))


# ------------------------------------------------------------------------------------------------------------------- #
def pad_array(some_array, target_len:int):
    """
    Pad an array to target len
    :param some_array: Array to be padded [individual length]
    :param target_len: Target  length
    :return: some_array [target_len]
    """
    return np.append(some_array, np.ones(target_len - len(some_array))*some_array[-1])




