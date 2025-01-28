from numpy.random import seed
#seed(26)
from tensorflow.random import set_seed
#set_seed(26)

from tensorflow.keras.layers import Input, Dense, Dropout, Normalization, Activation, Conv2D, MaxPooling2D, Flatten, \
    BatchNormalization, Concatenate
import argparse
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
import os
from nn.aorta import AortaNormalizer
import gc
from nn.eva_metrics import EvaMetrics
from nn.help_function_training import *
import warnings

warnings.filterwarnings('ignore')
set_global_determinism(9)


bSaveModel = True
bParseArgs = False

# Loading config ------------------------------------------------------------------------------------------ #
config_path = "C:/Users\pfuchs\Documents/uni-rostock/python_projects\EIT/nn/configs/"
data_path = "C:/Users\pfuchs\Documents/Data/EIT/PulHypStudie/DataOriginal/"

segmentfiles= "C:/Users\pfuchs\Documents/Data/Segmentierung_Heartbeats\PulHyp_Segs_neu3_withVv73\Segmentation_2024-06-04/"

mprefix = 'C:/Users/pfuchs\Documents/uni-rostock/python_projects\EIT/nn/models/'


# Parsing of arguments ------------------------------------------------------------------------------------------ #

config_file = "/config_modelTEST_hier.json"
with open(config_path + TP.sConfigFile, "r") as file:
    config = json.load(file)
iEpochs=150
bResampleParas=False

sNormAorta= "fixed"
iBatchsize=32
sParaType = "PolyHierarchicalLin"
iNumParas = 43



iNpigs = len(config["training_examples"])
print(f"{iNpigs=}")
iTestpig=8

# for pop_pig in range(n_pigs):
train_pigs = list(np.arange(iNpigs))
train_pigs.pop(iTestpig)
train_pigs = [config["training_examples"][sel] for sel in train_pigs]
test_pig = config["training_examples"][iTestpig]
print(f"Exclude pig: {test_pig}\n\n")


print("Started loading data.")
X, y, clrs_pig = load_preprocess_paras(
    config["data_prefix"],
    train_pigs,
    zero_padding=False,
    shuffle=True,
    eit_length=64,
    aorta_length=1400,
    para_len=iNumParas,
    norm_eit="block",
    norm_aorta=sNormAorta,
    resample_paras=bResampleParas,
    sUseIndex="none"
)

if sNormAorta != 'none':
    if sNormAorta == "fixed":
        AortaNorm= AortaNormalizer(paratype=sParaType, mode=sNormAorta)
        y = AortaNorm.normalize_forward(y)
    else:
        y, deduction, facNorm = normalize_aorta(y, sNormAorta)



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

X_train, X_valid, y_train, y_valid, clrs_train, clrs_valid = train_test_split(
    X, y, clrs_pig, test_size=0.1, random_state=42, shuffle=False
)
del X, y
print(f"{X_train.shape=},{X_valid.shape=},{y_train.shape=},{y_valid.shape=}")
y1_train, y2_train = split_y(y_train)
y1_valid, y2_valid = split_y(y_valid)



# Model Initialization----------------------------------------------------------------------------- #
def model(input_shape=(128, 1024, 1), latent_dim=3,ConfigParas=None):
    pattern, pattern_time = stride_pattern(input_shape[1], keepTimeDim=ConfigParas.iStrideTimeLength)  # usually 4,4,4,4
    print(pattern)
    curve_dim = 6
    latent_dim = 32

    mapper_input = Input(shape=input_shape)
    print("actiout" + str(ConfigParas.actiOutput))
    # normalize input data
    # x = mapper_input
    #  if bWeightingLayer:
    # x = CustomWeightingLayer(w)(mapper_input)
    # else:
    x = mapper_input

    # convolutional layers
    x = Conv2D(int(ConfigParas.iFilter1), ConfigParas.iKernel, strides=(pattern_time[0], pattern[0]), padding="same")(x)
    # x = MaxPooling2D(pool_size=(1,2))(x)
    if ConfigParas.bBatchNorm:
        x = BatchNormalization()(x)
    x = Activation(ConfigParas.actiConv)(x)
    if ConfigParas.bDropout:
        x = Dropout(ConfigParas.fNumDrop)(x)

    x = Conv2D(int(ConfigParas.iFilter2), ConfigParas.iKernel, strides=(pattern_time[1], pattern[1]), padding="same")(x)
    #     x = MaxPooling2D(pool_size=(1,2))(x)
    if ConfigParas.bBatchNorm:
        x = BatchNormalization()(x)
    x = Activation(ConfigParas.actiConv)(x)
    if ConfigParas.bDropout:
        x = Dropout(ConfigParas.fNumDrop)(x)
    ##
    #      x = Conv2D(16, kernel, strides=(1, 1), padding="same")(x)
    # x = MaxPooling2D(pool_size=(2,4))(x)
    #   x = BatchNormalization()(x)
    #       x = Activation(actiConv)(x)
    #   x = Dropout(0.5)(x)

    x = Conv2D(int(ConfigParas.iFilter3), ConfigParas.iKernel, strides=(pattern_time[2], pattern[2]), padding="same")(x)
    x = MaxPooling2D(pool_size=(pattern_time[3], pattern[3]))(x)
    #  x = BatchNormalization()(x)
    if ConfigParas.bBatchNorm:
        x = BatchNormalization()(x)
    x = Activation(ConfigParas.actiConv)(x)
    if ConfigParas.bDropout:
        x = Dropout(ConfigParas.fNumDrop)(x)

    x = Flatten()(x)

    if bMoreDense:
        x = Dense(int(ConfigParas.iFactorDimDense) * latent_dim, activation=ConfigParas.actiDense)(x)  # elu #hinzugefügt
    x = Dense(int(ConfigParas.iFactorDimDense) * latent_dim, activation=ConfigParas.actiDense)(x)  # elu #hinzugefügt  #todo
    x = Dense(latent_dim, activation=ConfigParas.actiDense)(x)  # elu #hinzugefügt
    xoutput = Dense(curve_dim, activation="linear")(x)  # elu

    xnew = Concatenate()([xoutput, x])
    xoutput2 = Dense(latent_dim, activation=ConfigParas.actiOutput)(xnew)  # elu

    return Model(mapper_input, outputs=[xoutput, xoutput2])


# Model Training ----------------------------------------------------------------------------- #
opt = Adam(learning_rate=0.001)
eit_sample_len = 1024


model = model(input_shape=(config["eit_length"], eit_sample_len, 1),latent_dim=iNumParas)
model.compile(optimizer=opt, loss="mae", metrics=["accuracy", "mse"])
model.summary()

history = model.fit(
    X_train,
    [y1_train, y2_train],
    validation_data=(X_valid, [y1_valid, y2_valid]),
    epochs=iEpochs,
    batch_size=iBatchsize)
print("Training finished")

# delete training stacks
del X_train, y_train

# load test data
print(f"float test pig: {test_pig}")

X_test, y_test, p_test = load_preprocess_paras(
    config["data_prefix"],
    [test_pig],
    zero_padding=False,
    shuffle=False,
    eit_length=64,
    aorta_length=1400,
    para_len=iNumParas,
    norm_eit="block",
    norm_aorta=sNormAorta,
    resample_paras=bResampleParas
)


if sNormAorta != 'none':
    if sNormAorta == "fixed":
        y_test = AortaNorm.normalize_forward(y_test)
    else:
        y_test, de, f = normalize_aorta(y_test, sNormAorta, deduction=deduction, facgiven=facNorm)


(y_test_preds1, y_test_preds2) = model(X_test)
y_test_preds = append_y(y_test_preds1, y_test_preds2, iLen=(1024-deduction)/facNorm)
del X_test, y_test_preds1, y_test_preds2
gc.collect()

(y_valid_preds1, y_valid_preds2) = model(X_valid)
y_valid_preds = append_y(y_valid_preds1, y_valid_preds2, iLen=(1024-deduction)/facNorm)
del X_valid, y_valid_preds1, y_valid_preds2
gc.collect()

# Save model ----------------------------------------------------------------------------- #
path = mprefix
if bSaveModel:
    import os
    import time
    from contextlib import redirect_stdout
    import pickle
    import shutil

    timestr = time.strftime("%Y%m%d-%H%M%S")
    path = os.path.join(mprefix, timestr)
    os.mkdir(path)

    # save weights
    #   model.save_weights(f'{path}/weights', overwrite=True)

    # save architecture
    with open(f'{path}/model_config.json', "w") as text_file:
        text_file.write(model.to_json())

    with open(f'{path}/model_summary.txt', "w") as text_file:
        # text_file.write("Test_pig: "+str(test_pig)+"\n\n\n")
        TP.write_configs(text_file)
        text_file.write("\n\n\n")
        with redirect_stdout(text_file):
            model.summary()
        #   model.summary(print_fn=lambda x: text_file.write(x + '\n'))
        # with redirect_stdout(text_file):
        #   model.summary()
    with open(f'{path}/test_results.txt', "w") as text_file:
        write_testdata(text_file, y_test, y_test_preds)

    #   from contextlib import redirect_stdout

    #  def myprint(s):
    #  with open("TEST.txt",'a') as f:
    #   print(s, file=f)
    #    model.summary(print_fn=myprint)
    # with open(f'model_summarytest.txt', "w") as text_file:
    #   text_file.write("Test_pig: "+str(test_pig)+"\n\n\n")
    # Pass the file handle in as a lambda function to make it callable
    # with redirect_stdout(text_file):
    #     model.summary()
    #    text_file.write(str(model.summary))
    # history
    with open(f'{path}/history.pickle', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    model.save(f'{path}/model.keras')
    path = path + "/"
    shutil.copyfile(config_path + config_file, path + config_file)

# Visualization ----------------------------------------------------------------------------- #
bPlotGraphics = True

plot_history(history, "loss", bSave=bPlotGraphics, fSavePath=path)


plot_parameters(y_test_preds, y_test, "Testing", bSave=bPlotGraphics, fSavePath=path)
plot_parameters(y_valid_preds, y_valid, "Validation", bSave=bPlotGraphics, fSavePath=path)
M = EvaMetrics(path)
M.bByParatype=False
M.calc_metrics(y_test, y_test_preds, TP.sParaType, "TestParas", bParas=True)
M.calc_metrics(y_valid, y_valid_preds, TP.sParaType, "ValiParas", bParas=True)

bScale = False



y_test_recon1 = np.array(recon_paras_block(y_test, sParaType+"Curve", bScale=bScale, Denorm=sNormAorta,normcoeffs= [facNorm, deduction]))
y_test_preds_recon1 = np.array(recon_paras_block(y_test_preds, sParaType+"Curve", bScale=bScale, Denorm=sNormAorta,normcoeffs= [facNorm, deduction]))


y_valid_recon2 = np.array(recon_paras_block(y_valid, sParaType+"Rest",1024,bScale=bScale, Denorm=sNormAorta,normcoeffs= [facNorm, deduction]))
y_valid_preds_recon2 = np.array(recon_paras_block(y_valid_preds, sParaType+"Rest", 1024, bScale, Denorm=sNormAorta,normcoeffs= [facNorm, deduction]))


y_test_recon = recon_paras_block(y_test, sParaType, bScale=bScale, Denorm=sNormAorta,normcoeffs= [facNorm, deduction])
y_test_preds_recon = recon_paras_block(y_test_preds, sParaType, bScale=bScale, Denorm=sNormAorta,normcoeffs= [facNorm, deduction])

del y_test, y_test_preds
gc.collect()


y_valid_recon = recon_paras_block(y_valid, sParaType, bScale=bScale, Denorm=sNormAorta,normcoeffs= [facNorm, deduction])
y_valid_preds_recon = recon_paras_block(y_valid_preds, sParaType, bScale=bScale, Denorm=sNormAorta,normcoeffs= [facNorm, deduction])


y_valid_recon1 = np.array(recon_paras_block(y_valid, sParaType+"Curve", bScale=bScale, Denorm=sNormAorta,normcoeffs= [facNorm, deduction]))
y_valid_preds_recon1 = np.array(recon_paras_block(y_valid_preds, sParaType+"Curve", bScale=bScale, Denorm=sNormAorta,normcoeffs= [facNorm, deduction]))

del y_valid, y_valid_preds
gc.collect()



if bResampleParas:
    y_test_recon_a, y_test_preds_recon_a = set_array_len(y_test_recon, y_test_preds_recon, len(y_test_recon[0]))
    y_valid_recon_a, y_valid_preds_recon_a = set_array_len(y_valid_recon, y_valid_preds_recon, len(y_valid_recon[0]))

else:
    y_test_recon_a, y_test_preds_recon_a = set_array_len_uneven(y_test_recon, y_test_preds_recon, 1200)
    y_valid_recon_a, y_valid_preds_recon_a = set_array_len_uneven(y_valid_recon, y_valid_preds_recon, 1200)

plt.plot(y_test_preds_recon_a[0])
plt.show()


M.calc_metrics(y_test_recon_a, y_test_preds_recon_a, sParaType, "TestCurve")
M.calc_metrics(y_valid_recon_a, y_valid_preds_recon_a, sParaType, "ValiCurve")
M.calc_metrics(y_valid_recon1, y_valid_preds_recon1, sParaType, "ValiCurve1", bParas=False)
M.calc_metrics(y_test_recon1, y_test_preds_recon1, sParaType, "TestCurve1", bParas=False)
M.save_metrics()

plot_appended_recon_curves(y_test_preds_recon[0:16], y_test_recon[:16], "Testing Reconstructed Signals", sParaType,
                           recon_given=True, bSave=bPlotGraphics, fSavePath=path)

plot_appended_recon_curves(y_test_preds_recon[100:116], y_test_recon[100:116], "Testing Reconstructed Signals Part2",
                           sParaType, recon_given=True, bSave=bPlotGraphics, fSavePath=path)

plot_appended_recon_curves(y_valid_preds_recon[0:16], y_valid_recon[:16], "Validation Reconstructed Signals", sParaType,
                           recon_given=True, bSave=bPlotGraphics, fSavePath=path)

plot_appended_recon_curves(y_valid_preds_recon[100:116], y_valid_recon[100:116],
                           "Validation Reconstructed Signals Part2", sParaType, recon_given=True, bSave=bPlotGraphics,
                           fSavePath=path)




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



ind = [100, 125, 140, 150]
pig_plot = []
for k in ind:
    pig_plot.append(clrs_valid[k])
aorta_seg_valid = reload_aorta_segs_from_piginfo(data_path, pig_plot, False,bResampled=bResampleParas, iResampleLen=1024)
pig_plot = []
for k in ind:
    pig_plot.append(p_test[k])
aorta_seg_test = reload_aorta_segs_from_piginfo(data_path, pig_plot, False,bResampled=bResampleParas, iResampleLen=1024)
plot_4_recon_curves_paper(y_test_preds_recon, y_test_recon, "Testing Reconstructed Signal Part2", sParaType,
                    segment=aorta_seg_test, recon_given=True, ind=ind, bSave=bPlotGraphics, fSavePath=path)
plot_4_recon_curves_paper(y_valid_preds_recon, y_valid_recon, "Validation Reconstructed Signal Part2", sParaType,
                    segment=aorta_seg_valid, recon_given=True, ind=ind, bSave=bPlotGraphics, fSavePath=path)




print("Testing finished.")

