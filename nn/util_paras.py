import numpy as np
from os.path import join
from glob import glob
from scipy import signal
from joblib import Parallel, delayed
from sklearn.preprocessing import LabelEncoder
import h5py
import json
np.random.seed(1)


# ------------------------------------------------------------------------------------------------------------------- #
# Util.py
# ------------------------------------------------------------------------------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------- #
def quality_checks(X: list, y: list, eit_length=128, aorta_length=1024):
    """
    Performs quality check on X and y by checking their corresponding lengths
    :return: Segment-indices not fullfilling qulatiy checks
    """
    idx = list()
    eit_min_length = 15

    # quality checks for EIT data
    for n, eit in enumerate(X):
        if eit.shape[0] > eit_length or eit.shape[0] <= eit_min_length:
            # print(f"dataset excluded: {eit.shape[0]=}>{eit_length}.")
            idx.append(n)

    # quality checks for aorta data
    for n, aorta in enumerate(y):
        if len(aorta) > aorta_length:
            # print(f"dataset excluded: {aorta.shape[0]=}>{aorta_length}.")
            idx.append(n)

    return idx





# ------------------------------------------------------------------------------------------------------------------- #
def normalize_eit(X: np.ndarray, pigs: np.ndarray, norm_eit: str):
    """Normalize EIT signals (z-score normalization)

    norm_eit : str with 'global' or per 'block' or 'block2' with not channel independent standard deviation

    """
    # TODO Unterscheidung verschiedene Studien
    if norm_eit == 'global':
        mx = np.mean(X, axis=(0, 1))
        sx = np.std(X, axis=(0, 1))
        X = (X - mx) / sx

    elif norm_eit == 'pig':
        le_p = LabelEncoder()
        le_p.fit(pigs[:, 0])

        for p in le_p.classes_:
            idx = np.where(pigs[:, 0] == p)
            mx = np.mean(X[idx, :, :], axis=(0, 1))
            sx = np.std(X[idx, :, :], axis=(0, 1))
            X[idx, :, :] = (X[idx, :, :] - mx) / sx


    elif norm_eit == 'block':
        le_p = LabelEncoder()
        le_b = LabelEncoder()
        le_p.fit(pigs[:, 0])

        for p in le_p.classes_:
            idx_p = np.where(pigs[:, 0] == p)
            le_b.fit(np.squeeze(pigs[idx_p, 1]))
            for b in le_b.classes_:
                idx = np.where((pigs[:, 0] == p) & (pigs[:, 1] == b))
                mx = np.mean(X[idx, :, :], axis=(0, 1))
                sx = np.std(X[idx, :, :], axis=(0, 1))
                X[idx, :, :] = (X[idx, :, :] - mx) / sx

    return X


# ------------------------------------------------------------------------------------------------------------------- #
def load_paras(X: list, y: list, pigs: list, path: str, para_len: int):
    """
    Load aorta pressure (and eit) signals from npz files in a given path

    X, y:      list to append the loaded data to
    path:      path to directory with npz files
    """
    print(f"Loading data from {path}")
    files = glob(join(path, "*.npz"), recursive=True)
    files = list(sorted(files))
    if len(files) == 0:
        raise Exception("No npz files found in directory")
    files= files[:1500]

    for filepath in files:
        tmp = np.load(filepath)
        X.append(tmp["eit_v"])
        y.append(tmp["aorta_para"])
        pigs.append(tmp["data_info"])
        if len(y[-1]) != para_len:
            if len(y[-1]) < para_len:
                p = np.append(y[-1], np.zeros(para_len - len(y[-1])))
                y[-1] = p
            else:
                y[-1] = y[-1][:para_len]
        paraType = tmp["para_type"]
    return X, y, pigs, paraType




# ------------------------------------------------------------------------------------------------------------------- #
def load_vent_signal(vsig,  path: str,venttype):
    """
    Load aorta pressure (and eit) signals from npz files in a given path

    X, y:      list to append the loaded data to
    path:      path to directory with npz files
    """
    print(f"Loading Ventilation Signal from {path}")
    files = glob(join(path, "*.npz"), recursive=True)
    files = list(sorted(files))
    if len(files) == 0:
        raise Exception("No npz files found in directory")
    files= files[:1500]

    if venttype=="middle":
        for filepath in files:
            tmp = np.load(filepath)
            vsig.append(tmp["vent_midseg"])
    else:
        for filepath in files:
            tmp = np.load(filepath)
            vsig.append(tmp["vent_startseg"])
    return vsig


# ------------------------------------------------------------------------------------------------------------------- #
def get_segment_length(X, fs = 47.6837):
    """
    Determine time length of segments
    X:      EIT data
    fs:     Sampling frequency
    """
    L= []
    for i in range(len(X)):
        L.append(len(X[i]/fs))
    return L


# ------------------------------------------------------------------------------------------------------------------- #
def get_pig_info(pigs: list, path: str):
    """


    Get info on segment origin [pig, study, index, block...
    """
    files = glob(join(path, "*.npz"), recursive=True)
    for filepath in files:
        tmp = np.load(filepath)
        pigs.append(tmp["data_info"])
    return pigs


# ------------------------------------------------------------------------------------------------------------------- #
def resample_eit(X: list, eit_length: int):
    """
    Resample EIT data to eit_length
    """
    num_cores = 8
    eit_frame_length = X[0].shape[1]

    def worker(eit_arr, length, eit_frame_length):
        return np.array(
            [signal.resample(eit_arr[:, j], length) for j in range(eit_frame_length)]
        ).T

    X = Parallel(n_jobs=num_cores)(
        delayed(worker)(eit_block, eit_length, eit_frame_length) for eit_block in X
    )
    return X


# ------------------------------------------------------------------------------------------------------------------- #
def resample_aorta(y: list, aorta_length: int):
    """
    Resample aorta segments to aorta_length
    """
    y = [signal.resample(sample, aorta_length) for sample in y]
    return y


# ------------------------------------------------------------------------------------------------------------------- #
def load_preprocess_paras(
        data_prefix: str,
        examples: list,
        para_len=60,
        raw=False,
        zero_padding=False,
        shuffle=True,
        eit_length=128,
        aorta_length=1024,
        norm_aorta="none",
        norm_eit="none",
        resample_paras=False,
        sUseIndex="none",
        useReziprok="none",
        loadVent="none",
        reorder_mea= False,
        getLengthSig=False,
        iLeaveOut=1
):
    """
    Loads the preprocessed NPZ files with data
    :return: X = EIT data [num segments x eit_length=64 x 1024(or number indices) x 1], y=aorta data parameters [numsegments x para_len]
             pig = Pig info [numsegments x 6] with [Pig, block, index, len, study..]
    """
    X = list()
    y = list()
    pigs = list()

        # load raw data from npz files
    for example in examples:
        X, y, pigs, para_type = load_paras(X, y, pigs, join(data_prefix, example), para_len)

    #  pigs = get_pig_info(pigs, join(data_prefix, example))

    if iLeaveOut!=1:
        X = X[::iLeaveOut]
        y = y[::iLeaveOut]
        pigs= pigs[::iLeaveOut]


    # if requested return raw data
    if raw:
        return X, y, pigs

    # perform quality checks and clean dataset
    rm_idx = quality_checks(X, y, eit_length, aorta_length)

    for idx in sorted(rm_idx, reverse=True):
        del X[idx]
        del y[idx]
        del pigs[idx]

    if loadVent !="none":
        vsig= list()
        for example in examples:
            vsig = load_vent_signal(vsig, join(data_prefix, example), loadVent)
        for idx in sorted(rm_idx, reverse=True):
            del vsig[idx]
        vsig = np.array(vsig)
        vsig = vsig[:,  np.newaxis]



    # create index for shuffling
    N = len(y)

    X = resample_eit(X, eit_length)

    X = np.array(X)
    if len(X)%8 != 0:
        n = len(X)//8
        X = X[:int(8*n)]
        pigs = pigs[:int(8*n)]
        y = y[:int(8*n)]
        if loadVent!="none":
            vsig=vsig[:int(8*n)]

    # create index for shuffling
    N = len(y)
    if shuffle:
        shuffle= np.arange(N)
        np.random.shuffle(shuffle)
    else:
        shuffle = range(N)

    # if requested normalize EIT signals
    if norm_eit != 'none':
        X = normalize_eit(X, np.array(pigs), norm_eit)

    X = X[:, :, :, np.newaxis]

    if resample_paras:
        y = resample_paras_aorta(y, 1024)

    # pre-process aorta signals not possible due to parameters
    y = np.array(y)
    pigs = np.array(pigs)
    if loadVent != "none":
        return X[shuffle, ...], y[shuffle, ...],vsig[shuffle, ...], pigs[shuffle, ...]

    # return EIT and aorta signals
    return X[shuffle, ...], y[shuffle, ...], pigs[shuffle, ...]


# ------------------------------------------------------------------------------------------------------------------- #
# Parameter specific functions
# ------------------------------------------------------------------------------------------------------------------- #




# ------------------------------------------------------------------------------------------------------------------- #
def check_resampled_paras(data_prefix: str, examples: list):
    """
    Check if resampled parameter data is at data_prefix
    """
    path = join(data_prefix, examples[0])
    print(f"Loading data from {path}")
    files = glob(join(path, "*.npz"), recursive=True)
    if len(files) == 0:
        raise Exception("No npz files found in directory")
    tmp = np.load(files[0])
    bResampledParas = tmp["bResampleAorta"]
    return bResampledParas



# ------------------------------------------------------------------------------------------------------------------- #
def reload_aorta_segs_from_piginfo(path_data, piglist, bRemovedMin=False,bResampled=False, iResampleLen=0, sNormAorta="none"):
    """
    Reloads original aorta segments from data
    :param path_data: Path for data
    :param piglist: list containing with explicit segments should be loaded [[Pig, Block,Study,  Startindex, Segmentlength], ...]
    :param bRemovedMin:
    :param bResampled: If segements should be resampled
    :param iResampleLen: If Resmpling, the desired length
    :return: Reloaded segments
    """
    segs = []
    for c in piglist:
        p = c[0]
        b = c[1]
        fpath = path_data + p + "_kalibrierteRuhephasen.mat"
        f = h5py.File(fpath, 'r')
        data = f.get('PigData')
        try:
            aorta = np.array(data[p][b]["Aorta"]).flatten()
            segs.append(aorta[int(c[3]):int(c[3]) + int(float(c[4]))])
        except:
            print("Seg not found")
            print(c)
           # segs.append(np.array([None]))
    if bRemovedMin:
        segs = remove_min_curve(segs)

    if bResampled:
        segs= resample_aorta(segs, iResampleLen)

    if sNormAorta =="formnorm":
        for k in range(len(segs)):
            segs[k] = (segs[k]-np.min(segs[k])) / (np.max(segs[k]) - np.min(segs[k]))

    return segs



# ------------------------------------------------------------------------------------------------------------------- #
def reload_ventparas_from_piginfo(path_data,adding, ydata, piglist, rangeAorta=3):
    """
    Reloads original aorta segments from data
    :param path_data: Path for data
    :param piglist: list containing with explicit segments should be loaded [[Pig, Block,Study,  Startindex, Segmentlength], ...]
    :param bRemovedMin:
    :param bResampled: If segements should be resampled
    :param iResampleLen: If Resmpling, the desired length
    :return: Reloaded segments
    """
    vs = []
    for c in piglist:
        p = c[0]
        b = c[1]
        aorta_index  = int(c[3])+1
        fpath = path_data + p + "_Ruhephasen_HB_Segments.mat"
        f = h5py.File(fpath, 'r')
        data = f.get('PigData')
        try:
            aortasegs = np.array(data[p][b]["Aorta_Segs"]).flatten()

            index= np.argwhere(aortasegs==aorta_index)[0][0]
            Vsig = np.array(data[p][b]["Ventilation_MidSeg"]).flatten()
            vs.append(Vsig[index])
        except:
            print("Seg not found")
            print(c)
           # segs.append(np.array([None]))
  #  rangeAorta= 3
    pressures = []
    for k in range(len(ydata)):
        pressure = rangeAorta*vs[k]

        if adding =="mean":
            pressure-= rangeAorta/2
        elif adding=="maximum":
            pressure-= rangeAorta
        ydata[k]= ydata[k]+ pressure
        pressures.append(pressure)

    return ydata, vs, pressures







# ------------------------------------------------------------------------------------------------------------------- #
def resample_paras_aorta(y: list, aorta_length: int):
    """
    Resample aorta curves represented by linear regression parameters
    :param y: list of parameter arrays
    :param aorta_length: Target length
    :return: y_resampled [num segs x aorta_length]
    """
    num_cores = 8
    para_length = y[0].shape[0]
    resample_index = np.arange(0, para_length, 2)

    def worker(para_array, aorta_frame_length):
        l = para_array[-2]
        checklen = 2
        while l == 0:
            l = para_array[-2 * checklen]
            checklen += 1
        para_array[resample_index] = np.array(para_array[resample_index] * (aorta_frame_length - 1) / l, dtype=int)
        #   para_array[resample_index] = int(para_array[resample_index] )
        return para_array

    y = Parallel(n_jobs=num_cores)(
        delayed(worker)(para_vec, aorta_length) for para_vec in y
    )

    ###
    # resample/interpolate signals to equal length aorta_length
    # y = [signal.resample(sample, aorta_length) for sample in y]
    return y


# ------------------------------------------------------------------------------------------------------------------- #
def write_configs(f, testpig, epochs, batch, lr, fac, latent_dim, kernel, f1, f2, f3, actiConv, actiDense, actiOutput,
                  bZero, bResamParas, norm, loss, sUseIndex, bDropout, numDrop, bWeight, bBatchNorm, bHPEIT, bLPEIT=False, useReziprok="none", normAorta="none", venttype="none", bUseVentPara=False, reordermea=False, bMoreDense=False, bUseLength=False  ):
    """
    Writes used parameter configuration to file f
    """
    f.write("Test_pig: " + str(testpig) + "\n")
    f.write("Epochs: " + str(epochs) + "\n")
    f.write("Batchsize: " + str(batch) + "\n")
    f.write("Learningrate: " + str(lr) + "\n")
    f.write("FactorDense: " + str(fac) + "\n")
    f.write("Latent_dim = Num Paras: " + str(latent_dim) + "\n")
    f.write("Kernel: " + str(kernel) + "\n")
    f.write("Filter number Conv layer 1: " + str(f1) + "\n")
    f.write("Filter number Conv layer 2: " + str(f2) + "\n")
    f.write("Filter number Conv layer 3: " + str(f3) + "\n")
    f.write("Activation function conv layer: " + str(actiConv) + "\n")
    f.write("Activation function dense layer: " + str(actiDense) + "\n")
    f.write("Activation function output layer: " + str(actiOutput) + "\n")
    f.write("bZeroPadding EIT signal: " + str(bZero) + "\n")
    f.write("Resampling Parameters : " + str(bResamParas) + "\n")
    f.write("Normalization strategy: " + str(norm) + "\n")
    f.write("Lossfunction: " + str(loss) + "\n")
    f.write("sUseIndex: " + str(sUseIndex) + "\n")
    f.write("bWeightingLayer: " + str(bWeight) + "\n")
    f.write("bDropout layer: " + str(bDropout) + "\n")
    f.write("Dropout factor: " + str(numDrop) + "\n")
    f.write("Batchnorm: " + str(bBatchNorm) + "\n")
    f.write("HighpassEIT: " + str(bHPEIT) + "\n")
    f.write("LowpassEIT: " + str(bLPEIT) + "\n")
    f.write("useReziprok: " + str(useReziprok) + "\n")
    f.write("normAorta: " + str(normAorta) + "\n")
    if venttype != "none":
        f.write("Ventilation signal type: " + str(venttype) + "\n")
    if bUseVentPara !=False:
        f.write("Use vent para: " + str(bUseVentPara) + "\n")

    if bUseLength !=False:
        f.write("Use length para: " + str(bUseLength) + "\n")
    f.write("Reorder Mea: " + str(reordermea) + "\n")
    f.write("bMoreDense: " + str(bMoreDense) + "\n")

