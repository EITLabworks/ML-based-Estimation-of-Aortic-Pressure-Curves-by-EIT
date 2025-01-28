from numpy.random import seed

seed(1)
from tensorflow.random import set_seed

set_seed(2)
from tensorflow.keras.layers import Input, Activation, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from tensorflow.keras.models import Model
from nn.util_paras import normalize_aorta
from tensorflow.keras.optimizers import Adam
#from parametrization.reconstruction import reconstruct_lin_tensors_block, reconstruct_lin_uncorrected_tensors_block
from nn.aorta import AortaNormalizer
from src.reconstruction import reorder_aorta_paras
from nn.eva_metrics import EvaMetrics
from nn.help_function_training import *
import warnings
import psutil

process = psutil.Process()
warnings.filterwarnings('ignore')
import gc
import math

set_global_determinism(26)
bSaveModel = True
bParseArgs = False

# Loading config ------------------------------------------------------------------------------------------ #
config_path = "C:/Users\pfuchs\Documents/uni-rostock/python_projects\EIT/nn/configs"
data_path = "C:/Users\pfuchs\Documents/Data/EIT/PulHypStudie/DataOriginal/"

segmentfiles = "C:/Users\pfuchs\Documents/Data/Segmentierung_Heartbeats\PulHyp_Segs_neu3_withVv73\Segmentation_2024-06-04/"

mprefix = 'C:/Users/pfuchs\Documents/uni-rostock/python_projects\EIT/nn/models/'

data_path = "Cv---"

#bMoreDense=False
# Parsing of arguments ------------------------------------------------------------------------------------------ #
#(pop_pig, kernel_num, filter1, filter2, filter3, actiConv, actiDense, actiOutput, lr, factor_dim_dense1,bDropout,
#numDrop, sUseIndex, batchsize, epochs, bZeropadding, norm, resample_paras, lossfct,
#bRunEagerly, config_file, bWeightingLayer, bBatchNorm, bHPEIT, bLPEIT, useReziprok, normAorta, bUseVentpara, reorder_mea, bMoreDense, bUseLength)=parse_arguments(bParseArgs, 300, 16)

config_file = "/config_modelTEST.json"
with open(config_path + config_file, "r") as file:
    config = json.load(file)
print("Starting with config:\n" + str(config))

iEpochs = 150
bResampleParas = True
sNormAorta = "fixed"
iTestpig = 8
iBatchsize = 32

sParaType = "CauchyLorentz"
iNumParas = 58
actiOutput = "linear"

iNpigs = len(config["training_examples"])
print(f"{iNpigs=}")

# for pop_pig in range(n_pigs):
train_pigs = list(np.arange(iNpigs))
train_pigs.pop(iTestpig)
train_pigs = [config["training_examples"][sel] for sel in train_pigs]
test_pig = config["training_examples"][iTestpig]
print(f"Exclude pig: {test_pig}\n\n")

print("Memory used " + str(process.memory_info().rss / 1024))

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
        AortaNorm = AortaNormalizer(paratype=sParaType, mode=sNormAorta)
        y = AortaNorm.normalize_forward(y)
    else:
        y, deduction, facNorm = normalize_aorta(y, sNormAorta)

print("Finished loading data.")

X_train, X_valid, y_train, y_valid, clrs_train, clrs_valid = train_test_split(
    X, y, clrs_pig, test_size=0.1, random_state=42, shuffle=False
)
del X, y
gc.collect()
print(f"{X_train.shape=},{X_valid.shape=},{y_train.shape=},{y_valid.shape=}")




# Start model initialization --------------------------------------------------------------------- #
#strategy = dist.MirroredStrategy(["GPU:0", "GPU:1"])
#print("Number of devices: {}".format(strategy.num_replicas_in_sync))

#with strategy.scope():
#Xscaler = Normalization(axis=2)


# Model Initialization----------------------------------------------------------------------------- #
def model(input_shape=(64, 1024, 1), latent_dim=58, ConfigParas=None):
    mapper_input = Input(shape=input_shape)
    x = mapper_input

    # convolutional layers
    x = Conv2D(8, kernel_size=5, strides=(2, 4), padding="same")(x)
    if ConfigParas.bBatchNorm:
        x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = Dropout(0.3)(x)

    x = Conv2D(8, kernel_size=5, strides=(2, 4), padding="same")(x)
    if ConfigParas.bBatchNorm:
        x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = Dropout(0.3)(x)

    x = Conv2D(6, kernel_size=5, strides=(1, 4), padding="same")(x)
    x = MaxPooling2D(pool_size=(1, 4))(x)
    #  x = BatchNormalization()(x)
    if ConfigParas.bBatchNorm:
        x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = Dropout(0.3)(x)

    x = Flatten()(x)

    # Dense Layer
    x = Dense(3 * latent_dim, activation="elu")(x)
    x = Dense(2 * latent_dim, activation="elu")(x)
    mapper_output = Dense(latent_dim, activation="linear")(x)
    return Model(mapper_input, mapper_output)


# Model Training ----------------------------------------------------------------------------- #
opt = Adam(learning_rate=0.001)
eit_sample_len = 1024

model = model(input_shape=(config["eit_length"], eit_sample_len, 1), latent_dim=iNumParas)

model.compile(optimizer=opt, loss="mae", metrics=["accuracy", "mse"])
model.summary()

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_valid, y_valid),
    epochs=150,
    batch_size=32, steps_per_epoch=math.ceil(X_train.shape[0] / iBatchsize),
    validation_steps=math.ceil(X_valid.shape[0] / iBatchsize)

)
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


y_test_preds = model(X_test)
y_valid_preds = model(X_valid)

del X_valid, X_test
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
        #text_file.write("Test_pig: "+str(test_pig)+"\n\n\n")
        TP.write_configs(text_file)
        text_file.write("\n\n\n")
        with redirect_stdout(text_file):
            model.summary()

    with open(f'{path}/test_results.txt', "w") as text_file:
        write_testdata(text_file, y_test, y_test_preds)

    with open(f'{path}/history.pickle', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    model.save(f'{path}/model.keras')
    path = path + "/"
    shutil.copyfile(config_path + sConfigFile, path + sConfigFile)

# Visualization ----------------------------------------------------------------------------- #
bPlotGraphics = True
plot_history(history, "loss", bSave=bPlotGraphics, fSavePath=path)

bScale = False
#if  TP.sParaType == "CauchyLorentz":
#  bScale = True
y_test_recon = recon_paras_block(y_test, sParaType, bScale=bScale, Denorm=sNormAorta,
                                 normcoeffs=[facNorm, deduction], bReorderLinParas=False)
y_test_preds_recon = recon_paras_block(y_test_preds, sParaType, bScale=bScale, Denorm=sNormAorta,
                                       normcoeffs=[facNorm, deduction], bReorderLinParas=False)

y_valid_recon = recon_paras_block(y_valid, sParaType, bScale=bScale, Denorm=sNormAorta,
                                  normcoeffs=[facNorm, deduction], bReorderLinParas=False)
y_valid_preds_recon = recon_paras_block(y_valid_preds, sParaType, bScale=bScale, Denorm=sNormAorta,
                                        normcoeffs=[facNorm, deduction], bReorderLinParas=False)

plot_appended_recon_curves(y_test_preds_recon[0:16], y_test_recon[:16], "Testing Reconstructed Signals", sParaType,
                           recon_given=True, bSave=bPlotGraphics, fSavePath=path)
plot_appended_recon_curves(y_valid_preds_recon[100:116], y_valid_recon[100:116],
                           "Validation Reconstructed Signals Part2", sParaType, recon_given=True,
                           bSave=bPlotGraphics, fSavePath=path)

plot_appended_recon_curves(y_test_preds_recon[100:116], y_test_recon[100:116], "Testing Reconstructed Signals Part2",
                           sParaType, recon_given=True, bSave=bPlotGraphics, fSavePath=path)

plot_parameters(y_test_preds, y_test, "Testing", bSave=bPlotGraphics, fSavePath=path)
plot_parameters(y_valid_preds, y_valid, "Validation", bSave=bPlotGraphics, fSavePath=path)

ind = [5, 25, 40, 50]
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
plot_4_recon_curves_paper(y_test_preds_recon, y_test_recon, "Testing Reconstructed Signal", sParaType,
                    segment=aorta_seg_test, recon_given=True, ind=ind, bSave=bPlotGraphics, fSavePath=path)
plot_4_recon_curves_paper(y_valid_preds_recon, y_valid_recon, "Validation Reconstructed Signal", sParaType,
                    segment=aorta_seg_valid, recon_given=True, ind=ind, bSave=bPlotGraphics, fSavePath=path)

ind = [100, 125, 140, 150]
pig_plot = []
for k in ind:
    pig_plot.append(p_test[k])
aorta_seg_test = reload_aorta_segs_from_piginfo(data_path, pig_plot, False, bResampled=bResampleParas,
                                                iResampleLen=1024, sNormAorta=sNormAorta)
plot_4_recon_curves_paper(y_test_preds_recon, y_test_recon, "Testing Reconstructed Signal Part2", sParaType,
                    segment=aorta_seg_test, recon_given=True, ind=ind, bSave=bPlotGraphics, fSavePath=path)

del aorta_seg_test, aorta_seg_valid
gc.collect()

M = EvaMetrics(path)
M.bByParatype = False
if bResampleParas:
    y_test_recon_a, y_test_preds_recon_a = set_array_len(y_test_recon, y_test_preds_recon, len(y_test_recon[0]))
    y_valid_recon_a, y_valid_preds_recon_a = set_array_len(y_valid_recon, y_valid_preds_recon, len(y_valid_recon[0]))

else:
    y_test_recon_a, y_test_preds_recon_a = set_array_len_uneven(y_test_recon, y_test_preds_recon, 1200)
    y_valid_recon_a, y_valid_preds_recon_a = set_array_len_uneven(y_valid_recon, y_valid_preds_recon, 1200)

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
