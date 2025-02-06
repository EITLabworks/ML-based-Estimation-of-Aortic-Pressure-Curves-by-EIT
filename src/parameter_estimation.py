"""
Author:  Patricia Fuchs
"""
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
from src.linear_regression import linear_parametrization_block
from src.cl_estimation import cl_estimation_block
from src.save_paras import store_block, store_pig_mat
import json
import warnings

warnings.filterwarnings('ignore')

# ----------------------------------------------------------------------------------------------------------------- #
# Estimation functions -------------------------------------------------------------------------------------------- #
def estimate_linear_total(Nseg, segs, segPos, Kinit, Kmax, nmsedBLimit, seg_len, bPlot=False,
                          bSaveToMat=False, bPlotOutliers=False, bResampled=False):
    print("---------------------------------------------------------------------")
    print("Starting PIECEWISE LINEAR REGRESSION")
    model, K, nmse_vec, paras, xp, yp = linear_parametrization_block(Nseg, segs, Kinit, Kmax, nmsedBLimit)
    print("Linear estimation finished")
    print("The measurement block was divided into " + str(Nseg) + " segments.")
    print("The average amount of linear parts is " + str(np.mean(K)))
    print("The max amount of pulses is " + str(np.max(K)))
    print("The average NMSEdb is " + str(np.mean(nmse_vec)))
    if bSaveToMat:
        blockMat = store_block("Linear", Nseg, segs, segPos, K, nmse_vec, paras, xp, yp, [[]], [[]],
                               seg_len, bResampled=bResampled)
    else:
        blockMat = None

    if bPlotOutliers:
        plot_outliers(model, segs, nmse_vec, nmsedBLimit, "Linear")

    if bPlot:
        plot_estimations(model, "Linear")

    return blockMat

# ----------------------------------------------------------------------------------------------------------------- #
def estimate_cauchylorentz_total(Nseg, segs, segPos, Kinit, Kmax, nmsedBLimit, seg_len, bPlot=False,
                                 bSaveToMat=False, bPlotOutliers=False, Rval=float(1 / 200), bResampled=False):
    print("---------------------------------------------------------------------")
    print("Starting Cauchy-Lorentz estimation")
    model, K, nmse_vec, paras, p, s, a, d = cl_estimation_block(Nseg, segs, Kinit, Kmax, nmsedBLimit, R=Rval)
    print("Starting CAUCHY-LORENTZ ESTIMATION")
    print("The measurement block was divided into " + str(Nseg) + " segments.")
    print("The average amount of pulses is " + str(np.mean(K)))
    print("The max amount of pulses is " + str(np.max(K)))
    print("The average NMSEdb is " + str(np.mean(nmse_vec)))

    if bSaveToMat:
        blockMat = store_block("CauchyLorentz", Nseg, segs, segPos, K, nmse_vec, paras, p, a, s, d, seg_len,
                               bResampled=bResampled)
    else:
        blockMat = None

    if bPlotOutliers:
        plot_outliers(model, segs, nmse_vec, nmsedBLimit, "Spectral")

    if bPlot:
        plot_estimations(model, "Spectral")

    return blockMat


# ----------------------------------------------------------------------------------------------------------------- #
def estimate_hierarchical_total_linear(h, Nseg, segs, segPos, Kinit, Kmax, nmsedBLimit, seg_len, bPlot=False,
                                       bSaveToMat=False, bPlotOutliers=False, Rval=float(1 / 200),
                                       bResampled=False):
    print("---------------------------------------------------------------------")
    print("Starting HIERARCHICAL ESTIMATION")
    model, K, nmse_vec, paras, curve, p, a = h.hier_block_linear(Nseg, segs, Kinit, Kmax, nmsedBLimit)
    print("The measurement block was divided into " + str(Nseg) + " segments.")
    print("The average amount for K is " + str(np.mean(K)))
    print("The max amount of pulses is " + str(np.max(K)))
    print("The average NMSEdb is " + str(np.mean(nmse_vec)))
    if bSaveToMat:
        blockMat = store_block( "PolyHierarchicalLin", Nseg, segs, segPos, K, nmse_vec, paras, p, a, [[]],
                               [[]], seg_len, curve, bResampled=bResampled)
    else:
        blockMat = None

    if bPlotOutliers:
        plot_outliers(model, segs, nmse_vec, nmsedBLimit, "Hierarchical")

    if bPlot:
        plot_estimations(model, "Hierarchical")
    return blockMat



# ----------------------------------------------------------------------------------------------------------------- #
# Help functions -------------------------------------------------------------------------------------------------- #
class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """

    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


# ----------------------------------------------------------------------------------------------------------------- #
def plot_estimations(model, paratype):
    fig, ax = plt.subplots(1, figsize=(15, 9))
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(2)
    for i in range(len(model)):
        ax.plot(model[i], 'steelblue', linewidth=0.1)
    ax.grid(linewidth=0.3)
    ax.set_ylabel("Pressure [mmHg]", fontsize=20, loc="top")
    ax.set_xlabel("Samples", fontsize=20, loc="right")
    ax.set_title(paratype+ " Estimations", fontsize=24)
    ax.yaxis.set_tick_params(labelsize=18)
    ax.xaxis.set_tick_params(labelsize=18)
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------------------------------------------------- #
def plot_outliers(model, original, nmse_vec, dbLim, paratype):
    # plot error loaded modeled aorta segments
    fig, ax = plt.subplots(figsize=(18, 6), nrows=1, ncols=2)
    for axi in ax.flat:
        axi.yaxis.set_tick_params(labelsize=18)
        axi.xaxis.set_tick_params(labelsize=18)
        axi.set_ylabel("Pressure [mmHg]", fontsize=20, loc="top")
        axi.set_xlabel("Samples", fontsize=20, loc="right")
        for axis in ["top", "bottom", "left", "right"]:
            axi.spines[axis].set_linewidth(2)
    total = 0
    for i in np.arange(len(model)):
        if nmse_vec[i] > dbLim:
            ax[0].plot(model[i], 'maroon', linewidth=0.5)
            ax[1].plot(original[i], "steelblue", linewidth=0.5)
            total += 1

    print(str(total) + " of " + str(len(model)) + " segments are outliers.")
    ax[0].set_title(paratype+ " Estimations with NMSE > " + str(dbLim), fontsize=24)
    ax[1].set_title("Original CAP",fontsize=24)
    ax[0].grid(linewidth=0.3)
    ax[1].grid(linewidth=0.3)
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------------------------------------------------- #
def finish_pig_calculation(matpath, matpathdict, pigname, bSavemat, matdict):
    if bSavemat:
        for val, data in matdict.items():
            pigMat = {pigname: matdict[val]}
            store_pig_mat(matpath+matpathdict[val]+pigname+".mat", val, pigMat)


# ----------------------------------------------------------------------------------------------------------------- #
def resample_aorta(y: list, aorta_length: int):
    len_vec = np.zeros(len(y))
    for k in range(len(y)):
        len_vec[k] = len(y[k])
    y = [signal.resample(sample, aorta_length) for sample in y]
    return y, len_vec
