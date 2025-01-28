from tensorflow.keras.layers import Input, Dense, Dropout, Normalization, Activation, Conv2D, MaxPooling2D, Flatten
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
from src.reconstruction import rescalingfactors_for_cl, rescale_paras, reorder_aorta_paras
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from nn.evaluation_fcts import *
import numpy as np
from nn.eva_metrics import EvaMetrics
import os
from os.path import join
from glob import glob
from nn.help_function_training import *
import warnings
from nn.util_paras import load_vent_signal
warnings.filterwarnings('ignore')


# Loading config ------------------------------------------------------------------------------------------ #
config_path= "C:/Users\pfuchs\Documents/uni-rostock/python_projects\EIT/nn"
data_path = "C:/Users\pfuchs\Documents/Data/EIT/PulHypStudie/DataOriginal/"
mprefix = 'C:/Users/pfuchs/Documents/uni-rostock/python_projects/EIT/nn/models/'
modelpath= "20240301-204901/"
modelpath="20250109-160205/"
path= mprefix+modelpath
files = glob(join(path, "*.txt"), recursive=True)
files = list(sorted(files))


TP = parse_arguments_new(False, 300,16)        # TrainingParameters

TP.sConfigFile = "/configs/config_model1.json"
with open(config_path+TP.sConfigFile, "r") as file:
    config = json.load(file)
print("Starting with config:\n"+str(config))

TP.bResampleParas=True
deduction=0
facNorm = 0
TP.iFilter3 = 6
TP.sLossfct ="mae"
TP.sParaType = config["para_type"]
if TP.sParaType=="CauchyLorentz":
    TP.actiOutput="linear"
    rescalevec= rescalingfactors_for_cl(14)
TP.check_normAorta()



# Test pig
TP.iTestpig= 8
TP.iNpigs = len(config["training_examples"])
test_pig = config["training_examples"][TP.iTestpig]
bEvalValidation=False
#todo load vali....
#resample_paras=True
TP.sParaType= config["para_type"]


# Load model
model = tf.keras.models.load_model(mprefix+modelpath+"model.keras")
model.summary()
#bWeightingLayer=True

# for pop_pig in range(n_pigs):
test_pig = [config["training_examples"][TP.iTestpig]]


## Load data ##################################################################
# load test data
print(f"float test pig: {test_pig}")
X_test, y_test, p_test = load_preprocess_paras(
    config["data_prefix"],
    test_pig,
    zero_padding=TP.bZeropadding,
    shuffle=True,
    eit_length=config["eit_length"],
    aorta_length=config["aorta_max_length"],
    para_len=config["para_size"],
    norm_eit=TP.sNormEIT,
    norm_aorta=TP.sNormAorta,
    resample_paras=TP.bResampleParas,
    sUseIndex=TP.sUseIndex,
    bWeighting=TP.bWeighting,
    bHP_EIT=TP.bHP_EIT,
    bLP_EIT=TP.bLP_EIT,
    useReziprok=TP.bUseReziprok,
    reorder_mea=TP.bReorderMea
)
if TP.sParaType=="CauchyLorentz":
    y_test = rescale_paras(y_test, rescalevec)
if TP.sNormAorta != 'none':
    y_test, de, f = normalize_aorta(y_test, TP.sNormAorta, deduction=deduction, facgiven=facNorm)
if TP.bReorderParas:
    y_test = reorder_aorta_paras(y_test, len(y_test[0]))


vmid= load_vent_signal([], join(config["data_prefix"], test_pig[0]),"middle")
vstart= load_vent_signal([], join(config["data_prefix"], test_pig[0]),"start")

from typing import List


def split_indices_by_value_range(values: List[float], x: int) -> List[List[int]]:
    """
    Split the value range [0, 1] into x parts and return a list of index arrays,
    where each array contains the indices of the input values that fall within the corresponding range.

    Args:
    - values (List[float]): List of values ranging from 0 to 1.
    - x (int): Number of parts to split the value range into.

    Returns:
    - List[List[int]]: A list of index lists, where each sublist contains the indices of values that fall in the corresponding range.
    """

    if x <= 0:
        raise ValueError("Number of parts 'x' must be a positive integer.")
    # Create an empty list to hold the indices for each range
    indices_per_range = [[] for _ in range(x)]
    # Calculate the size of each range segment
    range_size = 1 / x
    # Assign each value to the corresponding range
    for idx, value in enumerate(values):
        # Determine which range this value belongs to (clamp it to x-1 to handle value=1)
        range_index = min(int(value // range_size), x - 1)
        indices_per_range[range_index].append(idx)
    return indices_per_range


def error_over_venttype(mae_sig, vsig, xparts=20):
    indices_per_range = split_indices_by_value_range(vsig, xparts)
    esigs = []
    for j in range(xparts):
        vals= mae_sig[np.array(indices_per_range[j])]
        esigs.append(vals)
        print(len(vals))
    fig, ax = plt.subplots(figsize=(12, 12))
    bp1 = ax.boxplot(esigs, sym='', widths=0.3, meanline=False,
                     showmeans=True, patch_artist=True,
                     boxprops=dict(facecolor='lightblue', color='black', linewidth=3),
                     medianprops=dict(color='maroon', linewidth=3))
    ax.set_title("Average MAE depending on the ventilation state")
    ax.grid()
    ax.set_xticks(np.arange(0,1,20))
    ax.set_xlabel("Ventilation State", loc="right")
    ax.set_ylabel("Average MAE over an AP Curve", loc="top")
    plt.show()






y_test_preds = model(X_test)

# Visualization ----------------------------------------------------------------------------- #
# Visualization ----------------------------------------------------------------------------- #
bPlotGraphics = False
bShowGraphics= True
bRemovedMin = bool(config["Minremoved"])
bRemovedMin = False
print(bRemovedMin)


y_test_recon = recon_paras_block(y_test, TP.sParaType)
y_test_preds_recon = recon_paras_block(y_test_preds, TP.sParaType)
print("recon complete")
if TP.bResampleParas:
    y_test_recon_a, y_test_preds_recon_a = set_array_len(y_test_recon, y_test_preds_recon, len(y_test_recon[0]))
print("Resmapled")
#y_test_real = np.array(reload_aorta_segs_from_piginfo(data_path, p_test, bRemovedMin,bResampled=TP.bResampleParas, iResampleLen=1024, sNormAorta=TP.sNormAorta))
y_test_real = y_test_recon_a

print("Recon finished")

rel_err_config = {
    "std" : True,
    "var" : True,
    "mean" : True,
    "s_name" : None,
}
mae, mse, pn = plot_relative_error_aorta(y_test_real, y_test_preds_recon_a, **rel_err_config)



#plot_appended_recon_curves(y_test_preds_recon[0:16], y_test_recon[:16], "Testing Reconstructed Signals", TP.sParaType, recon_given=True, bSave=bPlotGraphics,fSavePath=path, bShow=bShowGraphics)

#plot_appended_recon_curves(y_test_preds_recon[100:116], y_test_recon[100:116], "Testing Reconstructed Signals Part2", TP.sParaType, recon_given=True, bSave=bPlotGraphics,fSavePath=path, bShow=bShowGraphics)
#error_curves(y_test_recon, y_test_preds_recon, TP.sParaType, recon_given=True, title="Testing Error Distribution", bSave=bPlotGraphics,fSavePath=path, bShow=bShowGraphics)
#error_curves_mean_form(y_test_recon, y_test_preds_recon, TP.sParaType, recon_given=True, title="Testing", bSave=bPlotGraphics,fSavePath=path, bShow=bShowGraphics)
#plot_parameters(y_test_preds, y_test,"Testing", bSave=bPlotGraphics,fSavePath=path, bShow=bShowGraphics)

ind = [5,25,40,50]
pig_plot = []
for k in ind:
    pig_plot.append(p_test[k])
aorta_seg_test = reload_aorta_segs_from_piginfo(data_path, pig_plot, bRemovedMin, sNormAorta=TP.sNormAorta, bResampled=TP.bResampleParas, iResampleLen=1024, )
plot_4_recon_curves_paper(y_test_preds_recon, y_test_recon, "Testing Reconstructed Signal", TP.sParaType,
                    segment=aorta_seg_test, recon_given=True,ind=ind, bSave=bPlotGraphics,fSavePath=path, bShow=True)


ind = [100,125,140,150]
pig_plot = []
for k in ind:
    pig_plot.append(p_test[k])
aorta_seg_test = reload_aorta_segs_from_piginfo(data_path, pig_plot, bRemovedMin, bResampled=TP.bResampleParas, iResampleLen=1024, sNormAorta=TP.sNormAorta)
plot_4_recon_curves_paper(y_test_preds_recon, y_test_recon, "Testing Reconstructed Signal Part2", TP.sParaType,
                    segment=aorta_seg_test, recon_given=True,ind=ind, bSave=bPlotGraphics,fSavePath=path, bShow=True)





M = EvaMetrics(path)
M.calc_metrics(y_test_real, y_test_preds_recon_a, TP.sParaType, "TestCurve", bSave=bPlotGraphics)
#M.calc_metrics(y_test_recon_a, y_test_preds_recon_a, TP.sParaType, "TestRecon", bSave=bPlotGraphics)
M.calc_metrics(y_test, y_test_preds.numpy(), TP.sParaType, "TestParas", bParas=True, bSave=bPlotGraphics)


M.gather_info()
print(M.metrics["Compact"])
print("MAE: ", M.metrics["Compact"]["Testing"]["Curve"]["MAE"])
print("MSE: ", M.metrics["Compact"]["Testing"]["Curve"]["MSE"])
print("Pearson: ", M.metrics["Compact"]["Testing"]["Curve"]["PearsonR"])
print("MAEform: ", M.metrics["Compact"]["Testing"]["Curve"]["MAEform"])
print("Testing finished.")

DAP = np.min(y_test_real, axis=1)
print(DAP.shape)
DAP = DAP-np.min(y_test_preds_recon_a, axis=1)
SAP =np.max(y_test_real, axis=1) -np.max(y_test_preds_recon_a, axis=1)
MAP = np.mean(y_test_real, axis=1) -np.mean(y_test_preds_recon_a, axis=1)
import pandas as pd
import seaborn as sns
dd= {"DAP": DAP, "SAP":SAP, "MAP":MAP}
DF = pd.DataFrame(dd)
sns.histplot(DF)
plt.show()
sns.boxplot(DF)
plt.grid()
plt.show()

y_test_preds_recon_a = y_test_preds_recon_a + 10
mae, mse, pn = plot_relative_error_aorta(y_test_real, y_test_preds_recon_a, **rel_err_config)
M.gather_info()
print(M.metrics["Compact"])
print("MAE: ", M.metrics["Compact"]["Testing"]["Curve"]["MAE"])
print("MSE: ", M.metrics["Compact"]["Testing"]["Curve"]["MSE"])
print("Pearson: ", M.metrics["Compact"]["Testing"]["Curve"]["PearsonR"])
print("MAEform: ", M.metrics["Compact"]["Testing"]["Curve"]["MAEform"])
print("Testing finished.")
