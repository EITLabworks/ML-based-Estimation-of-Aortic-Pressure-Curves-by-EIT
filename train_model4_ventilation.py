from numpy.random import seed
seed(1)
from tensorflow.random import set_seed
set_seed(2)
from tensorflow.keras.layers import Input, Dense, Dropout, Normalization, Activation, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Concatenate
import argparse
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import json
from sklearn.model_selection import train_test_split
from nn.util import load_preprocess_examples
from nn.util_paras import load_preprocess_paras, write_configs, reload_aorta_segs_from_piginfo,check_resampled_paras, normalize_aorta
from tensorflow import distribute as dist
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
#from parametrization.reconstruction import reconstruct_lin_tensors_block, reconstruct_lin_uncorrected_tensors_block
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from nn.evaluation_fcts import *
import numpy as np
import os
from src.reconstruction import rescalingfactors_for_cl, rescale_paras
from nn.eva_metrics import EvaMetrics
from nn.help_function_training import *
import warnings
import psutil
process=psutil.Process()
warnings.filterwarnings('ignore')
import gc
set_global_determinism(9)

bSaveModel= True
bParseArgs = False


# Loading config ------------------------------------------------------------------------------------------ #
config_path= "C:/Users\pfuchs\Documents/uni-rostock/python_projects\EIT/nn/configs"
data_path = "C:/Users\pfuchs\Documents/Data/EIT/PulHypStudie/DataOriginal/"



bMoreDense=False
# Parsing of arguments ------------------------------------------------------------------------------------------ #
(pop_pig, kernel_num, filter1, filter2, filter3, actiConv, actiDense, actiOutput, lr, factor_dim_dense1,bDropout,
 numDrop, sUseIndex, batchsize, epochs, bZeropadding, norm, resample_paras, lossfct,
 bRunEagerly, config_file, bWeightingLayer, bBatchNorm, bHPEIT, bLPEIT, useReziprok, normAorta, bUseVentpara, reorder_mea, bMoreDense, bUseLength)=parse_arguments(bParseArgs, 300, 16)



normAorta = "bipolar1200"
#useReziprok="merge"
#sUseIndex="none"
bWeightingLayer=False
bRunEagerly=False

config_file = "/config_model4_startvent.json"
with open(config_path+config_file, "r") as file:
    config = json.load(file)
print("Starting with config:")
print(config)
epochs=10
resample_paras=True
bDropout=True
numDrop=0.2
filter3=8

bUseLength=True

venttype= config["venttype"]            # middle or start of segment
if venttype != "middle":
    venttype="start"

deduction=0
facNorm = 0

#resample_paras=True
para_type= config["para_type"]

if para_type=="CauchyLorentz":
    actiOutput="linear"
    rescalevec= rescalingfactors_for_cl(14)

if normAorta=="standard" or (normAorta=="bipolar" or normAorta=="bipolar1200"):
    actiOutput="linear"

mprefix = 'C:/Users/pfuchs\Documents/uni-rostock/python_projects\EIT/nn/models/'
n_pigs = len(config["training_examples"])
print(f"{n_pigs=}")



print(sUseIndex)
print(actiConv + "  " + str(batchsize))
# for pop_pig in range(n_pigs):
train_pig = list(np.arange(n_pigs))
train_pig.pop(pop_pig)
train_pig = [config["training_examples"][sel] for sel in train_pig]
test_pig = config["training_examples"][pop_pig]
print(f"Exclude pig: {test_pig}\n\n")
print(f"{train_pig=}, {test_pig=}")
s_path = f"/model/all_pigs_without_{test_pig}/"



print("Started loading data.")
X, y, vsig, clrs_pig = load_preprocess_paras(
    config["data_prefix"],
    train_pig,
    zero_padding=bZeropadding,
    shuffle=True,
    eit_length=config["eit_length"],
    aorta_length=config["aorta_max_length"],
    para_len=config["para_size"],
    norm_eit=norm,
    norm_aorta=normAorta,
    resample_paras=resample_paras,
    sUseIndex=sUseIndex,
    bWeighting=bWeightingLayer,
    bHP_EIT=bHPEIT,
    bLP_EIT=bLPEIT,
    useReziprok=useReziprok,
    reorder_mea=reorder_mea,
    loadVent=venttype,
    getLengthSig=bUseLength
)
print("Memory used " + str(process.memory_info().rss/1024))
if para_type=="CauchyLorentz":
    y = rescale_paras(y, rescalevec)


if normAorta != 'none':
    y,deduction, facNorm = normalize_aorta(y, normAorta)

print("Finished loading data.")

X_train, X_valid, y_train, y_valid, clrs_train, clrs_valid, vsig_train, vsig_valid = train_test_split(
    X, y, clrs_pig,vsig, test_size=0.1, random_state=42, shuffle=False)


del X, y, vsig
gc.collect()
print(f"{X_train.shape=},{X_valid.shape=},{y_train.shape=},{y_valid.shape=}, {vsig_train.shape=}, {vsig_valid.shape=}")

strategy = dist.MirroredStrategy(["GPU:0", "GPU:1"])
#print("Number of devices: {}".format(strategy.num_replicas_in_sync))



num_paras= config["para_size"]

#arguments
latent_dim=num_paras

if bWeightingLayer:
    w =create_weighting_vec(1)


# Start model initialization --------------------------------------------------------------------- #
#with strategy.scope():
#Xscaler = Normalization(axis=2)


# Model Initialization----------------------------------------------------------------------------- #
def model(input_shape=(128, 1024, 1), latent_dim=3, kernel=9, filter1=8, filter2=8, filter3=12, actiConv="elu",
          actiDense="elu", actiOutput="relu", factor=3, bDropout=False, numDrop=0.1, bWeightingLayer=False, bBatch=False ):


    pattern, pattern_time = stride_pattern(input_shape[1], keepTimeDim=16)  # usually 4,4,4,4
    print(pattern)

    mapper_input = Input(shape=input_shape)
    if bUseLength:
        vent_input = Input(shape=(2))
    else:
        vent_input = Input(shape=(1,))
  #  vent_input = Flatten()(vent_input)
    print("actiout" + str(actiOutput))
    # normalize input data
    #x = mapper_input
  #  if bWeightingLayer:
       # x = CustomWeightingLayer(w)(mapper_input)
   # else:
    x = mapper_input

    # convolutional layers
    x = Conv2D(int(filter1), kernel, strides=(2, pattern[0]), padding="same")(x)
    # x = MaxPooling2D(pool_size=(1,2))(x)
    if bBatch:
        x = BatchNormalization()(x)
    x = Activation(actiConv)(x)
    if bDropout:
        x = Dropout(numDrop)(x)

    x = Conv2D(int(filter2), kernel, strides=(2, pattern[1]), padding="same")(x)
    #     x = MaxPooling2D(pool_size=(1,2))(x)
    if bBatch:
        x = BatchNormalization()(x)
    x = Activation(actiConv)(x)
    if bDropout:
        x = Dropout(numDrop)(x)
    ##
    #      x = Conv2D(16, kernel, strides=(1, 1), padding="same")(x)
    # x = MaxPooling2D(pool_size=(2,4))(x)
    #   x = BatchNormalization()(x)
    #       x = Activation(actiConv)(x)
    #   x = Dropout(0.5)(x)

    x = Conv2D(int(filter3), kernel, strides=(1, pattern[2]), padding="same")(x)
    x = MaxPooling2D(pool_size=(1, pattern[3]))(x)
    #  x = BatchNormalization()(x)
    if bBatch:
        x = BatchNormalization()(x)
    x = Activation(actiConv)(x)
    if bDropout:
        x = Dropout(numDrop)(x)

    x = Flatten()(x)
    x = Concatenate()([vent_input, x])

    if bMoreDense:
            x = Dense(int(factor)*latent_dim, activation=actiDense)(x)# elu #hinzugefügt
    x = Dense(int(factor)*latent_dim, activation=actiDense)(x)# elu #hinzugefügt
    x = Dense(2*latent_dim, activation=actiDense)(x)# elu

    # x = Dense(latent_dim, activation="elu")(x)
    mapper_output = Dense(latent_dim, activation=actiOutput)(x)  ##linear

    return Model([mapper_input, vent_input], mapper_output)


# Model Training ----------------------------------------------------------------------------- #
opt = Adam(learning_rate=lr)
eit_sample_len = 1024
#if sUseIndex != "none":
eit_sample_len = len(X_valid[0][0])
print(eit_sample_len)
model = model(input_shape=(config["eit_length"], eit_sample_len, 1), latent_dim=num_paras, kernel=kernel_num,
              filter1=filter1, filter2=filter2, filter3=filter3, actiConv=actiConv, actiDense=actiDense, actiOutput=actiOutput,
              factor=factor_dim_dense1, bDropout=bDropout, numDrop=numDrop, bWeightingLayer=bWeightingLayer, bBatch=bBatchNorm)
model.compile(optimizer=opt, loss=lossfct, metrics= ["accuracy", "mse"], run_eagerly=bRunEagerly)
model.summary()
#import visualkeras
#visualkeras.layered_view(model, font="Calibri").show() # display using your system viewer
#visualkeras.layered_view(model, to_file='output.png') # write to disk
#visualkeras.layered_view(model, to_file='output.png').show() # write and show

#visualkeras.layered_view(model)


print("Memory used " + str(process.memory_info().rss/1024))
history = model.fit(
    [X_train, vsig_train],
    y_train,
    validation_data=([X_valid,vsig_valid], y_valid),
    epochs=epochs,
    batch_size=batchsize)
print("Training finished")

# delete training stacks
del X_train, y_train, vsig_train
# load test data
print(f"float test pig: {test_pig}")
print("Memory used " + str(process.memory_info().rss/1024))
X_test, y_test, vsig_test, p_test = load_preprocess_paras(
    config["data_prefix"],
    [test_pig],
    para_len=config["para_size"],
    zero_padding=bZeropadding,
    shuffle=True,
    eit_length=config["eit_length"],
    aorta_length=config["aorta_max_length"],
    norm_eit=norm,
    resample_paras=resample_paras,
    sUseIndex=sUseIndex,
    bWeighting=bWeightingLayer,
    bHP_EIT=bHPEIT,
    bLP_EIT=bLPEIT,
    useReziprok=useReziprok,
    loadVent=venttype,
    getLengthSig=bUseLength
)

if para_type=="CauchyLorentz":
    y_test = rescale_paras(y_test, rescalevec)

if normAorta != 'none':
    y_test,de, f =normalize_aorta(y_test, normAorta, deduction=deduction, facgiven=facNorm)

y_test_preds = model([X_test, vsig_test])
del X_test
gc.collect()
y_valid_preds = model([X_valid, vsig_valid])
del X_valid
gc.collect()
print("Memory used " + str(process.memory_info().rss/1024))

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
        #text_file.write("Test_pig: "+str(test_pig)+"\n\n\n")
        write_configs(text_file, test_pig, epochs, batchsize, lr, factor_dim_dense1, latent_dim, kernel_num, filter1,
                      filter2, filter3, actiConv, actiDense, actiOutput, bZeropadding, resample_paras, norm,lossfct,
                      sUseIndex, bDropout, numDrop,  bWeightingLayer, bBatchNorm, bHPEIT, bLPEIT, useReziprok=useReziprok, normAorta=normAorta, bUseLength=bUseLength)
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
print("Memory used " + str(process.memory_info().rss/1024))
bPlotGraphics = True
plot_history(history, "mse", bSave=bPlotGraphics,fSavePath=path)
plot_history(history, "loss", bSave=bPlotGraphics,fSavePath=path)


print("Memory used " + str(process.memory_info().rss/1024))

bScale = False
if para_type == "CauchyLorentz":
    bScale = True
y_test_recon = recon_paras_block(y_test, para_type, bScale=bScale, Denorm=normAorta,normcoeffs= [facNorm, deduction])
y_test_preds_recon = recon_paras_block(y_test_preds, para_type, bScale=bScale, Denorm=normAorta,normcoeffs= [facNorm, deduction])

y_valid_recon = recon_paras_block(y_valid, para_type, bScale=bScale, Denorm=normAorta,normcoeffs= [facNorm, deduction])
y_valid_preds_recon = recon_paras_block(y_valid_preds, para_type, bScale=bScale, Denorm=normAorta,normcoeffs= [facNorm, deduction])



plot_appended_recon_curves(y_test_preds_recon[0:16], y_test_recon[:16], "Testing Reconstructed Signals", para_type, recon_given=True, bSave=bPlotGraphics,fSavePath=path)

plot_appended_recon_curves(y_test_preds_recon[100:116], y_test_recon[100:116], "Testing Reconstructed Signals Part2", para_type, recon_given=True, bSave=bPlotGraphics,fSavePath=path)

plot_appended_recon_curves(y_valid_preds_recon[0:16], y_valid_recon[:16], "Validation Reconstructed Signals", para_type, recon_given=True, bSave=bPlotGraphics,fSavePath=path)

plot_appended_recon_curves(y_valid_preds_recon[100:116], y_valid_recon[100:116], "Validation Reconstructed Signals Part2", para_type, recon_given=True, bSave=bPlotGraphics,fSavePath=path)

error_curves(y_valid_recon, y_valid_preds_recon, para_type, recon_given=True,  title="Validation Error Distribution", bSave=bPlotGraphics,fSavePath=path)
error_curves(y_test_recon, y_test_preds_recon, para_type, recon_given=True, title="Testing Error Distribution", bSave=bPlotGraphics,fSavePath=path)

error_curves_mean_form(y_valid_recon, y_valid_preds_recon, para_type, recon_given=True,  title="Validation", bSave=bPlotGraphics,fSavePath=path)
error_curves_mean_form(y_test_recon, y_test_preds_recon, para_type, recon_given=True, title="Testing", bSave=bPlotGraphics,fSavePath=path)
print("Memory used " + str(process.memory_info().rss/1024))


plot_parameters(y_test_preds, y_test,"Testing", bSave=bPlotGraphics,fSavePath=path)
plot_parameters(y_valid_preds, y_valid, "Validation", bSave=bPlotGraphics,fSavePath=path)

#plot_recon_curve(y_test_preds_recon, y_test_recon, "Testing",para_type, recon_given=True, bSave=bPlotGraphics,fSavePath=path)
#plot_recon_curve(y_valid_preds_recon, y_valid_recon, "Validation ",para_type, recon_given=True, bSave=bPlotGraphics,fSavePath=path)

double_boxplot_error("Error Distribution for Aorta Curves", y_test_recon, y_test_preds_recon, y_valid_recon, y_valid_preds_recon,para_type, recon_given=True, bSave=bPlotGraphics,fSavePath=path)
double_boxplot_error_mean_form("Error Distribution for Aorta Curves Mean and Form", y_test_recon, y_test_preds_recon, y_valid_recon, y_valid_preds_recon,para_type, recon_given=True, bSave=bPlotGraphics,fSavePath=path)



bRemovedMin = config["Minremoved"]
if bRemovedMin=="False":
    bRemovedMin=False
else:
    bRemovedMin = True
print("Memory used " + str(process.memory_info().rss/1024))
print(bRemovedMin)


pig_plot = [p_test[3]]
aorta_seg = reload_aorta_segs_from_piginfo(data_path, pig_plot, bRemovedMin)
plot_single_recon_curve(y_test_preds_recon[3], y_test_recon[3], "Testing Data Reconstructed Signal", para_type,aorta_seg[0], recon_given=True,bSave=bPlotGraphics,fSavePath=path)


print("Memory used " + str(process.memory_info().rss/1024))
ind = [5,25,40,50]
pig_plot = []
for k in ind:
    pig_plot.append(clrs_valid[k])
aorta_seg_valid = reload_aorta_segs_from_piginfo(data_path, pig_plot, bRemovedMin)
pig_plot = []
for k in ind:
    pig_plot.append(p_test[k])
aorta_seg_test =reload_aorta_segs_from_piginfo(data_path, pig_plot, bRemovedMin)

plot_4_recon_curves(y_test_preds_recon, y_test_recon, "Testing Reconstructed Signal", para_type,
                    segment=aorta_seg_test, recon_given=True,ind=ind, bSave=bPlotGraphics,fSavePath=path)
plot_4_recon_curves(y_valid_preds_recon, y_valid_recon, "Validation Reconstructed Signal", para_type,
                    segment=aorta_seg_valid, recon_given=True,ind=ind, bSave=bPlotGraphics,fSavePath=path)


ind = [100,125,140,150]
pig_plot = []
for k in ind:
    pig_plot.append(clrs_valid[k])
aorta_seg_valid = reload_aorta_segs_from_piginfo(data_path, pig_plot, bRemovedMin)
pig_plot = []
for k in ind:
    pig_plot.append(p_test[k])
aorta_seg_test = reload_aorta_segs_from_piginfo(data_path, pig_plot, bRemovedMin)
plot_4_recon_curves(y_test_preds_recon, y_test_recon, "Testing Reconstructed Signal Part2", para_type,
                    segment=aorta_seg_test, recon_given=True,ind=ind, bSave=bPlotGraphics,fSavePath=path)
plot_4_recon_curves(y_valid_preds_recon, y_valid_recon, "Validation Reconstructed Signal Part2", para_type,
                    segment=aorta_seg_valid, recon_given=True,ind=ind, bSave=bPlotGraphics,fSavePath=path)
print("Memory used " + str(process.memory_info().rss/1024))
del aorta_seg_test, aorta_seg_valid
gc.collect()
print("Memory used " + str(process.memory_info().rss/1024))


M = EvaMetrics(path)
if para_type != "Linear":
    M.bByParatype=False
if resample_paras:
    y_test_recon_a, y_test_preds_recon_a = set_array_len(y_test_recon, y_test_preds_recon, len(y_test_recon[0]))
    y_valid_recon_a, y_valid_preds_recon_a = set_array_len(y_valid_recon, y_valid_preds_recon, len(y_valid_recon[0]))

else:
    y_test_recon_a, y_test_preds_recon_a = set_array_len_uneven(y_test_recon, y_test_preds_recon, 1200 )
    y_valid_recon_a, y_valid_preds_recon_a = set_array_len_uneven(y_valid_recon, y_valid_preds_recon, 1200)
del y_valid_preds_recon, y_valid_recon, y_test_recon, y_test_preds_recon
gc.collect()


y_test_real = reload_aorta_segs_from_piginfo(data_path, p_test, bRemovedMin,bResampled=resample_paras, iResampleLen=1024 )
y_valid_real = reload_aorta_segs_from_piginfo(data_path, clrs_valid, bRemovedMin,bResampled=resample_paras, iResampleLen=1024 )


M.calc_metrics(y_test_real, y_test_preds_recon_a, para_type, "TestCurve")
M.calc_metrics(y_valid_real, y_valid_preds_recon_a, para_type, "ValiCurve")


M.calc_metrics(y_test_recon_a, y_test_preds_recon_a, para_type, "TestRecon")
M.calc_metrics(y_valid_recon_a, y_valid_preds_recon_a, para_type, "ValiRecon")
M.calc_metrics(y_test, y_test_preds.numpy(), para_type, "TestParas", bParas=True)
M.calc_metrics(y_valid, y_valid_preds.numpy(), para_type, "ValiParas", bParas=True)
M.save_metrics()



#e_block, e_block_db = calc_error_block(y_test_preds, y_test,"NMSE")
#plot_histogram_error("Histogram NMSE [dB] Testing Data for Model 1", e_block_db)

print("Testing finished.")
print("Memory used " + str(process.memory_info().rss/1024))
