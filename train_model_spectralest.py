"""
Author: Patricia Fuchs
"""
from tensorflow.keras.layers import Input, Activation, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from nn.aorta import AortaNormalizer
from nn.eva_metrics import EvaMetrics
from nn.help_function_training import *
import warnings
import psutil
process = psutil.Process()
warnings.filterwarnings('ignore')
import gc
import os
import time
from contextlib import redirect_stdout
import pickle


# Loading config ------------------------------------------------------------------------------------------ #
data_path = "C:/Users\pfuchs\Documents/Data/EIT/PulHypStudie/DataOriginal/"


mprefix = 'C:/Users/pfuchs\Documents/uni-rostock/python_projects\EIT/nn/models/'
data_prefix = "/home/pfuchs/Data/Data_npz/PulHyp_k14_14Resampled/Data_CauchyLorentz/"

training_examples= [ "P01_PulHyp", "P02_PulHyp", "P03_PulHyp", "P04_PulHyp", "P05_PulHyp", "P06_PulHyp", "P07_PulHyp",
                     "P08_PulHyp", "P09_PulHyp", "P10_PulHyp"]


# Parsing of arguments ------------------------------------------------------------------------------------------ #
bSaveModel = True
iEpochs = 150
bResampleParas = False
sNormAorta = "fixed"
iTestpig = 8
iBatchsize = 32

sParaType = "CauchyLorentz"
iNumParas = 58
actiOutput = "linear"
venttype= "start"

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
    AortaNorm = AortaNormalizer(paratype=sParaType, mode=sNormAorta)
    y = AortaNorm.normalize_forward(y)
print("Finished loading data.")


# Split into train and validation set ------------------------------------------------------------- #
X_train, X_valid, y_train, y_valid, clrs_train, clrs_valid, vsig_train, vsig_valid = train_test_split(
    X, y, clrs_pig, vsig, test_size=0.1, random_state=42, shuffle=False)
del X, y, vsig
gc.collect()
print(f"{X_train.shape=},{X_valid.shape=},{y_train.shape=},{y_valid.shape=}, {vsig_train.shape=}, {vsig_valid.shape=}")


# Model Initialization----------------------------------------------------------------------------- #
def model(input_shape=(64, 1024, 1), latent_dim=42):
    mapper_input = Input(shape=input_shape)
    x = mapper_input

    vent_input = Input(shape=(1,))

    # Convolutional Layers
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

    # Dense Layers
    x = Flatten()(x)
    x = Concatenate()([vent_input, x])
    x = Dense(126, activation="elu")(x)
    x = Dense(84, activation="elu")(x)
    mapper_output = Dense(latent_dim, activation="linear")(x)
    return Model([mapper_input, vent_input], mapper_output)


# Model Training ----------------------------------------------------------------------------- #
opt = Adam(learning_rate=0.001)
eit_sample_len = 1024

model = model(input_shape=(64, eit_sample_len, 1), latent_dim=iNumParas)
model.compile(optimizer=opt, loss="mae", metrics=["accuracy", "mae"])
model.summary()

history = model.fit(
    [X_train, vsig_train],
    y_train,
    validation_data=([X_valid,vsig_valid], y_valid),
    epochs=iEpochs,
    batch_size=iBatchsize
)
print("Training finished")
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
y_test_preds = model(X_test)
y_valid_preds = model(X_valid)

del X_valid, X_test
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
bResampleParas = True

y_test_recon = recon_paras_block(y_test, sParaType, bScale=False, Denorm=sNormAorta)
y_test_preds_recon = recon_paras_block(y_test_preds, sParaType, bScale=False, Denorm=sNormAorta)

y_valid_recon = recon_paras_block(y_valid, sParaType, bScale=False, Denorm=sNormAorta)
y_valid_preds_recon = recon_paras_block(y_valid_preds, sParaType, bScale=False, Denorm=sNormAorta)



plot_appended_recon_curves(y_test_preds_recon[700:712], y_test_recon[700:712], "Testing Reconstructed Signals", sParaType,
                           recon_given=True, bSave=bPlotGraphics, fSavePath=path)



plot_parameters(y_test_preds, y_test, "Testing", bSave=bPlotGraphics, fSavePath=path)
plot_parameters(y_valid_preds, y_valid, "Validation", bSave=bPlotGraphics, fSavePath=path)

ind = [800, 5000, 7100, 7000]
pig_plot = []
for k in ind:
    pig_plot.append(clrs_valid[k])
aorta_seg_valid = reload_aorta_segs_from_piginfo(data_path, pig_plot, False, sNormAorta=sNormAorta,
                                                 bResampled=bResampleParas)
pig_plot = []
for k in ind:
    pig_plot.append(p_test[k])
aorta_seg_test = reload_aorta_segs_from_piginfo(data_path, pig_plot, False, sNormAorta=sNormAorta,
                                                bResampled=bResampleParas)

plot_4_recon_curves_paper(y_test_preds_recon, y_test_recon, "Testing Reconstructed Signal", sParaType,
                          segment=aorta_seg_test, recon_given=True, ind=ind, bSave=bPlotGraphics, fSavePath=path)


del aorta_seg_test, aorta_seg_valid
gc.collect()

M = EvaMetrics(path)
M.bByParatype = False

y_test_recon_a, y_test_preds_recon_a = set_array_len(y_test_recon, y_test_preds_recon, len(y_test_recon[0]))
y_valid_recon_a, y_valid_preds_recon_a = set_array_len(y_valid_recon, y_valid_preds_recon, len(y_valid_recon[0]))

del y_valid_preds_recon, y_valid_recon, y_test_recon, y_test_preds_recon
gc.collect()

y_test_real = np.array(
    reload_aorta_segs_from_piginfo(data_path, p_test, False, bResampled=bResampleParas, iResampleLen=1024,
                                   sNormAorta=sNormAorta))
y_valid_real = np.array( reload_aorta_segs_from_piginfo(data_path, clrs_valid, False, bResampled=bResampleParas, iResampleLen=1024,
                                   sNormAorta=sNormAorta))

M.calc_metrics(y_test_real, y_test_preds_recon_a, sParaType, "TestCurve")
M.calc_metrics(y_valid_real, y_valid_preds_recon_a, sParaType, "ValiCurve")

M.calc_metrics(y_test, y_test_preds.numpy(), sParaType, "TestParas", bParas=True)
M.calc_metrics(y_valid, y_valid_preds.numpy(), sParaType, "ValiParas", bParas=True)

M.save_metrics()

print("Testing finished.")
