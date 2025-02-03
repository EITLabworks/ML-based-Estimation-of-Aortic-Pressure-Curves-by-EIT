"""
Author: Patricia Fuchs
"""

from tensorflow.keras.layers import Input, Dense, Dropout, Normalization, Activation, Conv2D, MaxPooling2D, Flatten, \
    BatchNormalization, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import json
from sklearn.model_selection import train_test_split
from nn.util_paras import load_preprocess_paras, write_configs,normalize_aorta, reload_aorta_segs_from_piginfo, check_resampled_paras
from tensorflow import distribute as dist
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
# from parametrization.reconstruction import reconstruct_lin_tensors_block, reconstruct_lin_uncorrected_tensors_block
from tensorflow.keras import backend as K
from src.reconstruction import rescalingfactors_for_polylin, rescale_paras, reorder_aorta_paras
import matplotlib.pyplot as plt
from nn.evaluation_fcts import *
import numpy as np
from nn.aorta import AortaNormalizer
import gc
from nn.eva_metrics import EvaMetrics
from nn.help_function_training import *
import warnings
import gc
import os
import time
from contextlib import redirect_stdout
import pickle
warnings.filterwarnings('ignore')

bSaveModel = True


# Loading config ------------------------------------------------------------------------------------------ #
config_path = "C:/Users\pfuchs\Documents/uni-rostock/python_projects\EIT/nn/configs/"
data_path = "C:/Users\pfuchs\Documents/Data/EIT/PulHypStudie/DataOriginal/"
segmentfiles= "C:/Users\pfuchs\Documents/Data/Segmentierung_Heartbeats\PulHyp_Segs_neu3_withVv73\Segmentation_2024-06-04/"

mprefix = 'C:/Users/pfuchs\Documents/uni-rostock/python_projects\EIT/nn/models/'
data_prefix = "C:/Users\\pfuchs\\Documents/Data/Data_npz/PulHyp_k20_20_EITbased1128/Data_Linear/"
training_examples= [ "P01_PulHyp", "P02_PulHyp", "P03_PulHyp", "P04_PulHyp", "P05_PulHyp", "P06_PulHyp", "P07_PulHyp",
                     "P08_PulHyp", "P09_PulHyp", "P10_PulHyp"]

# Parsing of arguments ------------------------------------------------------------------------------------------ #

iEpochs=150
bResampleParas=False

bSaveModel = True
sNormAorta= "fixed"
iBatchsize = 32
sParaType = "PolyHierarchicalLin"
iNumParas = 43
venttype = "start"

iNpigs = len(training_examples)
print(f"{iNpigs=}")
iTestpig = 8

# for pop_pig in range(n_pigs):
train_pigs = list(np.arange(iNpigs))
train_pigs.pop(iTestpig)
train_pigs = [training_examples[sel] for sel in train_pigs]
test_pig = training_examples[iTestpig]
print(f"Exclude pig: {test_pig}\n\n")


print("Started loading data.")
X, y, vsig, clrs_pig = load_preprocess_paras(
    data_prefix,
    train_pigs,
    zero_padding=False,
    shuffle=True,
    eit_length=64,
    aorta_length=1400,
    para_len=iNumParas,
    norm_eit="block",
    norm_aorta=sNormAorta,
    resample_paras=bResampleParas,
    sUseIndex="none",
    loadVent=venttype
)
if sNormAorta == "fixed":
    AortaNorm= AortaNormalizer(paratype=sParaType, mode=sNormAorta)
    y = AortaNorm.normalize_forward(y)
print("Finished loading data.")


def split_y(Y, curvelen=7, bLeaveLen=True):
    # Y=[numsamples, lenparas]
    if bLeaveLen:
        Ycurve = Y[:, 1:curvelen]
    else:
        Ycurve = Y[:, :curvelen]
    Yerror = Y[:, curvelen:]
    return Ycurve, Yerror


def append_y(ycurve, yresidual, bAddLength=True, iLen=1024):
    y = np.append(ycurve, yresidual, axis=1)
    if bAddLength:
        L = np.ones((len(ycurve), 1)) * iLen
        y = np.append(L, y, axis=1)
    return y


print("Finished loading data.")
X_train, X_valid, y_train, y_valid, clrs_train, clrs_valid, vsig_train, vsig_valid = train_test_split(
    X, y, clrs_pig, vsig,  test_size=0.1, random_state=42, shuffle=False
)
del X, y, vsig
print(f"{X_train.shape=},{X_valid.shape=},{y_train.shape=},{y_valid.shape=}, {vsig_train.shape=}, {vsig_valid.shape=}")
y1_train, y2_train = split_y(y_train)
y1_valid, y2_valid = split_y(y_valid)

print(y1_train.shape)
print(y2_train.shape)

strategy = dist.MirroredStrategy(["GPU:0", "GPU:1"], cross_device_ops=tf.distribute.ReductionToOneDevice())
print("Number of devices: {}".format(strategy.num_replicas_in_sync))



# Model Initialization----------------------------------------------------------------------------- #
def model(input_shape=(64, 1024, 1), latent_dim=43):
    curve_dim=6
    latent_dim=36
    mapper_input = Input(shape=input_shape)
    x = mapper_input

    vent_input = Input(shape=(1,))
    # Convolutional layers
    x = Conv2D(4, kernel_size=[5,8], strides=(2,4), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(0.2)(x)

    x = Conv2D(5, kernel_size=[1,3], strides=(1,2), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("tanh")(x)
    x = Dropout(0.2)(x)

    x = Conv2D(7, kernel_size=[5,4], strides=(3, 2), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("tanh")(x)
    x = Dropout(0.0)(x)

    x = Conv2D(9, kernel_size=[5,6], strides=(3,2), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("tanh")(x)
    x = Dropout(0.3)(x)

    x = Flatten()(x)        #-> 384
    x = Concatenate()([vent_input, x])

    x = Dense(126, activation="elu")(x)   # elu
    x = Dense(84, activation="elu")(x)   # elu
    xoutput = Dense(curve_dim, activation="linear")(x)
    mapper_output = Dense(latent_dim, activation="linear")(x)
    xnew = Concatenate()([xoutput, x])
    xoutput2 = Dense(latent_dim, activation="linear")(xnew)

    return Model([mapper_input, vent_input], [xoutput, xoutput2])


# Model Training ----------------------------------------------------------------------------- #
opt = Adam(learning_rate=0.001)
eit_sample_len = 1024


model = model(input_shape=(64, eit_sample_len, 1),latent_dim=iNumParas)
model.compile(optimizer=opt, loss="mae", metrics=["accuracy", "mae"])
model.summary()

history = model.fit(
    [X_train, vsig_train],
    [y1_train, y2_train],
    validation_data=([X_valid,vsig_valid], [y1_valid, y2_valid]),
    epochs=iEpochs,
    batch_size=iBatchsize)
print("Training finished")

# delete training stacks
del X_train, y_train, vsig_train

# Loading test data ----------------------------------------------------------------------------- #
print(f"float Test pig: {test_pig}")
X_test, y_test,vsig_test, p_test = load_preprocess_paras(
    data_prefix,
    [test_pig],
    zero_padding=False,
    shuffle=False,
    eit_length=64,
    aorta_length=1400,
    para_len=iNumParas,
    norm_eit="block",
    norm_aorta=sNormAorta,
    resample_paras=bResampleParas,
    loadVent=venttype
)

if sNormAorta == "fixed":
    y_test = AortaNorm.normalize_forward(y_test)



# Testing and Validation ----------------------------------------------------------------------------- #
(y_test_preds1, y_test_preds2) = model(X_test)
y_test_preds = append_y(y_test_preds1, y_test_preds2, iLen=1)
del X_test, y_test_preds1, y_test_preds2
gc.collect()

(y_valid_preds1, y_valid_preds2) = model(X_valid)
y_valid_preds = append_y(y_valid_preds1, y_valid_preds2, iLen=1)
del X_valid, y_valid_preds1, y_valid_preds2
gc.collect()

# Save model ----------------------------------------------------------------------------- #
path = mprefix
if bSaveModel:
    timestr = time.strftime("%Y%m%d-%H%M%S")
    path = os.path.join(mprefix, timestr)
    os.mkdir(path)
    # save weights
    #   model.save_weights(f'{path}/weights', overwrite=True)

    # save architecture
    with open(f'{path}/model_config.json', "w") as text_file:
        text_file.write(model.to_json())

    with open(f'{path}/model_summary.txt', "w") as text_file:
        text_file.write("\n\n\n")
        with redirect_stdout(text_file):
            model.summary()

    with open(f'{path}/history.pickle', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    model.save(f'{path}/model.keras')


# Visualization ----------------------------------------------------------------------------- #
bPlotGraphics = True

plot_history(history, "loss", bSave=bPlotGraphics, fSavePath=path)


plot_parameters(y_test_preds, y_test, "Testing", bSave=bPlotGraphics, fSavePath=path)
plot_parameters(y_valid_preds, y_valid, "Validation", bSave=bPlotGraphics, fSavePath=path)
M = EvaMetrics(path)
M.bByParatype=False
M.calc_metrics(y_test, y_test_preds, sParaType, "TestParas", bParas=True)
M.calc_metrics(y_valid, y_valid_preds, sParaType, "ValiParas", bParas=True)



y_test_recon1 = np.array(recon_paras_block(y_test, sParaType+"Curve", bScale=False, Denorm=sNormAorta))
y_test_preds_recon1 = np.array(recon_paras_block(y_test_preds, sParaType+"Curve", bScale=False, Denorm=sNormAorta))


y_valid_recon2 = np.array(recon_paras_block(y_valid, sParaType+"Rest",1024,bScale=False, Denorm=sNormAorta,))
y_valid_preds_recon2 = np.array(recon_paras_block(y_valid_preds, sParaType+"Rest", 1024, False, Denorm=sNormAorta))


y_test_recon = recon_paras_block(y_test, sParaType, bScale=False, Denorm=sNormAorta)
y_test_preds_recon = recon_paras_block(y_test_preds, sParaType, bScale=False, Denorm=sNormAorta)

del y_test, y_test_preds
gc.collect()


y_valid_recon = recon_paras_block(y_valid, sParaType, bScale=False, Denorm=sNormAorta)
y_valid_preds_recon = recon_paras_block(y_valid_preds, sParaType, bScale=False, Denorm=sNormAorta)


y_valid_recon1 = np.array(recon_paras_block(y_valid, sParaType+"Curve", bScale=bScale, Denorm=sNormAorta,normcoeffs= [facNorm, deduction]))
y_valid_preds_recon1 = np.array(recon_paras_block(y_valid_preds, sParaType+"Curve", bScale=bScale, Denorm=sNormAorta,normcoeffs= [facNorm, deduction]))

del y_valid, y_valid_preds
gc.collect()




y_test_recon_a, y_test_preds_recon_a = set_array_len(y_test_recon, y_test_preds_recon, len(y_test_recon[0]))
y_valid_recon_a, y_valid_preds_recon_a = set_array_len(y_valid_recon, y_valid_preds_recon, len(y_valid_recon[0]))



M.calc_metrics(y_test_recon_a, y_test_preds_recon_a, sParaType, "TestCurve")
M.calc_metrics(y_valid_recon_a, y_valid_preds_recon_a, sParaType, "ValiCurve")
M.calc_metrics(y_valid_recon1, y_valid_preds_recon1, sParaType, "ValiCurve1", bParas=False)
M.calc_metrics(y_test_recon1, y_test_preds_recon1, sParaType, "TestCurve1", bParas=False)
M.save_metrics()

plot_appended_recon_curves(y_test_preds_recon[0:16], y_test_recon[:16], "Testing Reconstructed Signals", sParaType,
                           recon_given=True, bSave=bPlotGraphics, fSavePath=path)



ind = [5, 25, 40, 50]
pig_plot = []
for k in ind:
    pig_plot.append(clrs_valid[k])
aorta_seg_valid = reload_aorta_segs_from_piginfo(data_path, pig_plot, False,bResampled=bResampleParas, iResampleLen=1024)
pig_plot = []
for k in ind:
    pig_plot.append(p_test[k])
aorta_seg_test = reload_aorta_segs_from_piginfo(data_path, pig_plot, False,bResampled=bResampleParas, iResampleLen=1024)

plot_4_recon_curves_paper(y_test_preds_recon, y_test_recon, "Testing Reconstructed Signal", sParaType,
                    segment=aorta_seg_test, recon_given=True, ind=ind, bSave=bPlotGraphics, fSavePath=path)
plot_4_recon_curves_paper(y_valid_preds_recon, y_valid_recon, "Validation Reconstructed Signal", sParaType,
                    segment=aorta_seg_valid, recon_given=True, ind=ind, bSave=bPlotGraphics, fSavePath=path)




print("Testing finished.")

