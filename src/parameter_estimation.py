"""
Author:  Patricia Fuchs
"""
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
from src.linear_regression import linear_parametrization_block
from parametrization.cl_estimation import cl_estimation_block_test
from parametrization.save_paras import store_block, store_pig_mat
import json
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

block_dict = {"Kinit": 0, "Kmax": 0, "NMSElimit": 0, "Method": "", "Nsegments": 0, "Npulsesmean": 0, "Npulsesmax": 0,
              "Npulsessigma": 0, "NMSEmean": 0, "NMSEsigma": 0}

pig_dict = {"Kpulses": np.array([]), "NMSE": np.array([]), "N": 0}

pig_options = ["P01", "P02", "P03", "P04", "P05", "P06", "P07", "P08", "P09"]


# ----------------------------------------------------------------------------------------------------------------- #
# Estimation functions -------------------------------------------------------------------------------------------- #
def estimate_linear_total(Nseg, segs, segPos, Kinit, Kmax, nmsedBLimit, seg_len, bPlot=False,
                          bSaveToMat=False, bPlotOutliers=False, bResampled=False):
    print("---------------------------------------------------------------------")
    print("Start PIECEWISE LINEAR REGRESSION")
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
        plot_outliers(model, segs, nmse_vec, nmsedBLimit)

    if bPlot:
        fig, ax = plt.subplots(2, figsize=(15, 9))
        for i in range(len(model)):
            if nmse_vec[i] > -30:
                ax[0].plot(model[i], 'b-', linewidth=0.1)
        for i in np.arange(Nseg):
            if nmse_vec[i] > -30:
                ax[1].plot(segs[i], 'r-', linewidth=0.1)
        ax[0].grid()
        ax[1].grid()
        ax[0].set_title("Piecewise Linear Regression")
        ax[1].set_title("Original segments")
        plt.show()

    return blockMat

# ----------------------------------------------------------------------------------------------------------------- #
def estimate_cauchylorentz_total(Nseg, segs, segPos, Kinit, Kmax, nmsedBLimit, seg_len, bPlot=False,
                                 bSaveToMat=False, bPlotOutliers=False, Rval=float(1 / 200), bResampled=False):
    print("---------------------------------------------------------------------")
    print("Starting Cauchy-Lorentz estimation")
    model, K, nmse_vec, paras, p, s, a, d = cl_estimation_block_test(Nseg, segs, Kinit, Kmax, nmsedBLimit, R=Rval)
    print("CAUCHY-LORENTZ ESTIMATION")
    print("The measurement block was divided into " + str(Nseg) + " segments.")
    print("The average amount of linear parts is " + str(np.mean(K)))
    print("The max amount of pulses is " + str(np.max(K)))
    print("The average NMSEdb is " + str(np.mean(nmse_vec)))

    if bSaveToMat:
        blockMat = store_block("CauchyLorentz", Nseg, segs, segPos, K, nmse_vec, paras, p, a, s, d, seg_len,
                               bResampled=bResampled)
    else:
        blockMat = None


    if bPlotOutliers:
        plot_outliers(model, segs, nmse_vec, nmsedBLimit)

    out30 = 0
    for i in range(len(model)):
        if nmse_vec[i] > -30:
            out30 += 1

    if bPlot:
        fig, ax = plt.subplots(2, figsize=(15, 9))
        for i in range(len(model)):
            if nmse_vec[i] > -30:
                ax[0].plot(model[i], 'b-', linewidth=0.1)
        # plt.plot(x_axis, y_1, "r",  label="Originalsegment")

        for i in np.arange(Nseg):
            if nmse_vec[i] > -30:
                ax[1].plot(segs[i], 'r-', linewidth=0.1)
        ax[0].grid()
        ax[1].grid()
        ax[0].set_title("Spectral Estimations")
        ax[1].set_title("Original segments")
        plt.show()

    return blockMat


# ----------------------------------------------------------------------------------------------------------------- #
def estimate_hierarchical_total_linear(h_class, Nseg, segs, segPos, Kinit, Kmax, nmsedBLimit, seg_len, bPlot=False,
                                       bSaveToMat=False, bPlotOutliers=False, Rval=float(1 / 200),
                                       bResampled=False):
    print("---------------------------------------------------------------------")
    print("Starting Hierarchical estimation")
    blockmats = {}
    model, K, nmse_vec, paras, curve, p, a = h.hier_block_linear(Nseg, segs, Kinit, Kmax, hconfig[method][1])
    # model, K, nmse_vec, paras, p, s, a, d = cl_estimation_block_test(Nseg, segs, Kinit, Kmax, nmsedBLimit, R=Rval)
    print("Starting HIERARCHICAL ESTIMATION")
    print("The measurement block was divided into " + str(Nseg) + " segments.")
    print("The average amount for K is " + str(np.mean(K)))
    print("The max amount of pulses is " + str(np.max(K)))
    print("The average NMSEdb is " + str(np.mean(nmse_vec)))
    if bSaveToMat:
        blockMat = store_block( "PolyHierarchicalLin", Nseg, segs, segPos, K, nmse_vec, paras, p, a, [[]],
                               [[]], seg_len, curve, bResampled=bResampled)
        blockmats.update({"PolyHierarchical": blockMat})
    else:
        blockMat = None


    if bPlotOutliers:
        plot_outliers(model, segs, nmse_vec, hconfig[method][1])

    if bPlot:
        fig, ax = plt.subplots(2, figsize=(15, 9))
        for i in range(len(model)):
            if nmse_vec[i] > -30:
                ax[0].plot(model[i], 'b-', linewidth=0.1)
        # plt.plot(x_axis, y_1, "r",  label="Originalsegment")

        for i in np.arange(Nseg):
            if nmse_vec[i] > -30:
                ax[1].plot(segs[i], 'r-', linewidth=0.1)
        ax[0].grid()
        ax[1].grid()
        ax[0].set_title("Hierarchical estimations")
        ax[1].set_title("Original segments")
        plt.show()

    return blockmats







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
def print_fct(method, measurementtime, Nseg, K, nmsevec):
    print("---------------------------------------------------------------------")
    print(method + " for " + str(measurementtime))
    print("The measurement block was divided into " + str(Nseg) + " segments.")
    print("The average amount of parts/pulses is " + str(np.mean(K)))
    print("The max amount of pulses is " + str(np.max(K)))
    print("The average NMSEdb is " + str(np.mean(nmsevec)))





# ----------------------------------------------------------------------------------------------------------------- #
def save_total(d, method, path):
    title = "SigEstimation_" + method + datetime.today().strftime('_%H-%M-%S_%d-%m-%Y') + ".json"
    dumped = json.dumps(d, cls=NumpyEncoder)
    with open(path + title, "w") as outfile:
        outfile.write(dumped)


# ----------------------------------------------------------------------------------------------------------------- #
def save_total_short(d, method, path):
    for p in pig_options:
        if p in d:
            if "Kpulses" in d[p]:
                del d[p]["Kpulses"]
                del d[p]["NMSE"]
    del d["Kpulses"]
    del d["NMSE"]
    title = "SigEstimationShort_" + method + datetime.today().strftime('_%H-%M-%S_%d-%m-%Y') + ".json"
    dumped = json.dumps(d, cls=NumpyEncoder)
    with open(path + title, "w") as outfile:
        outfile.write(dumped)


# ----------------------------------------------------------------------------------------------------------------- #
def load_total(title, path, method, bShort=False):
    if bShort:
        t = "SigEstimationShort_" + method + "_" + title + ".json"
    else:
        t = "SigEstimation_" + method + "_" + title + ".json"
    with open(path + t, 'r') as openfile:
        # Reading from json file
        d = json.load(openfile)
    return d


# ----------------------------------------------------------------------------------------------------------------- #
def plot_outliers(model, original, nmse_vec, dbLim):
    # plot error loaded modeled aorta segments
    fig, ax = plt.subplots(figsize=(18, 6), nrows=1, ncols=2)
    total = 0
    for i in np.arange(len(model)):
        if nmse_vec[i] > dbLim:
            ax[0].plot(model[i], 'g-', linewidth=0.1)
            ax[1].plot(original[i], "b", linewidth=0.1)
            total += 1

    print(str(total) + " of " + str(len(model)) + " segments are outliers.")
    ax[0].set_title("Reconstructed aorta segments with NMSE > " + str(dbLim))
    ax[1].set_title("Original aorta")
    ax[0].grid()
    ax[1].grid()
    plt.show()




# ----------------------------------------------------------------------------------------------------------------- #
def finish_pig_calculation(pigdict, total, pigname, path, bSavemat, matdict, matpath):
    for val, pdPartial in pigdict.items():
        if bSavemat:
            pigMat = {pigname: matdict[val]}
            store_pig_mat(matpath, val, pigMat)


# ----------------------------------------------------------------------------------------------------------------- #
def update_setting(config):
    print("Config with following parameters started:")
    for k, v in config.items():
        print(k + ":  " + str(v))
    return (
    config["bGaussian"], config["bCauchyLorentz"], config["bLinear"], config["bHierarchical"], config["bSaveToMat"],
    config["nmsedBLimit"],
    config["Kmax"], config["Kinit"], config["Rval"])


# ----------------------------------------------------------------------------------------------------------------- #
def resample_aorta(y: list, aorta_length: int):
    len_vec = np.zeros(len(y))
    for k in range(len(y)):
        len_vec[k] = len(y[k])
    y = [signal.resample(sample, aorta_length) for sample in y]
    return y, len_vec
