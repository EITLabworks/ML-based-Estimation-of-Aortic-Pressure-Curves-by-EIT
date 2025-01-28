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
def normalize_aorta(y: np.ndarray, norm_aorta:str ,  invert=False, facgiven=0, deduction=0):
    """New normalization of aortic paras

    invert:  Denormalize if True"""
    if facgiven ==0 or (facgiven==1 and deduction==0):

        if norm_aorta == "positive1024":
            deduction = 0
            factor = 1024
        elif norm_aorta == "bipolar":
            deduction = 0
            factor = np.max(np.abs(y), axis=None)
        elif norm_aorta == "bipolar1200":
            deduction = 0
            factor = np.max(np.abs(y), axis=None)
            if factor > 1200:
                factor = 1200

        elif norm_aorta == "standard":
            deduction = np.mean(y,axis=None)
            factor = np.std(y,axis=None)

        elif norm_aorta =="formnorm":
            if not invert:
                y = form_norm_aorta_paras(y)
            else:
                y = make_pos_to1(y,bInvert=invert)
            deduction = 0
            factor=1

        elif norm_aorta == "normtoone":
            y =make_pos_to1(y, bInvert=invert)
            y,factor,deduction = norm_amplitudes(y, bInvert=invert, factor=facgiven, deduction=deduction)

        elif norm_aorta == "normPos":
            y = make_pos_to1(y, bInvert=invert)
            factor= 1
            deduction=0

        else:    # ===positive
            deduction = np.min(y,axis=None)
            factor = np.max(y,axis=None) - deduction
    elif norm_aorta == "normtoone":
        y = make_pos_to1(y, bInvert=invert)
        y, factor, deduction = norm_amplitudes(y, bInvert=invert, factor=facgiven, deduction=deduction)
    elif norm_aorta == "normPos":
        y = make_pos_to1(y, bInvert=invert)
        factor = 1
        deduction = 0
    else:
        factor = facgiven
    if norm_aorta != "normtoone" and norm_aorta != "normPos":
        if invert:
            y = y * factor + deduction
        else:
            y = (y - deduction) / factor
    return y, deduction, factor



#todo check if correct
def form_norm_aorta_paras(y):
    pos_index = np.arange(0, y.shape[1], 2)
    form_index = pos_index+1
    L = y[0][-2]
    y[:, pos_index] = y[:, pos_index]/L
    MaxVal = np.max(y[:, form_index], axis=1)
    MinVal = np.min(y[:, form_index], axis=1)
    Factor = np.max(y[:, form_index], axis=1) -MinVal
    y[:, form_index] = (y[:, form_index] -MinVal[:, np.newaxis ])/ Factor[:, np.newaxis]
    return y



def make_pos_to1(y, bInvert=False):
    pos_index = np.arange(0, y.shape[1], 2)

    if not bInvert:
        L = y[0][-2]
        y[:, pos_index] = y[:, pos_index]/L
    else:
        L = 1023
        y[:, pos_index] = y[:, pos_index] * L
    return y



def norm_amplitudes(y, bInvert=False, factor=1, deduction=0):
    form_index= np.arange(0, y.shape[1], 2)+1
    if not bInvert:
        if deduction == 0 or factor==0:
            amount= y[:, form_index]
            amount= amount [amount!= 0]
            deduction = np.min(amount)
            factor = np.max(y[:, form_index]) -deduction
        y[:, form_index] = (y[:, form_index]-deduction) / factor
    else:
        y[:, form_index] = y[:, form_index] *factor+deduction
    return y, factor, deduction


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

    elif norm_eit == 'block1024':
        le_p = LabelEncoder()
        le_b = LabelEncoder()
        le_p.fit(pigs[:, 0])

        for p in le_p.classes_:
            idx_p = np.where(pigs[:, 0] == p)
            le_b.fit(np.squeeze(pigs[idx_p, 1]))
            for b in le_b.classes_:
                idx = np.where((pigs[:, 0] == p) & (pigs[:, 1] == b))
                mx = np.mean(X[idx, :, :], axis=(0, 1,2))
                sx = np.std(X[idx, :, :], axis=(0, 1,2))
                X[idx, :, :] = (X[idx, :, :] - mx) / sx

    elif norm_eit == 'minmaxblock1024':
        le_p = LabelEncoder()
        le_b = LabelEncoder()
        le_p.fit(pigs[:, 0])

        for p in le_p.classes_:
            idx_p = np.where(pigs[:, 0] == p)
            le_b.fit(np.squeeze(pigs[idx_p, 1]))
            for b in le_b.classes_:
                idx = np.where((pigs[:, 0] == p) & (pigs[:, 1] == b))
                mx = np.min(X[idx, :, :], axis=(0, 1,2))
                maxi = np.max(X[idx, :, :], axis=(0, 1,2))
                X[idx, :, :] = (X[idx, :, :] - mx) / (maxi-mx)



    elif norm_eit == 'block1024M':
        s = 1000000     # Scale factor
        le_p = LabelEncoder()
        le_b = LabelEncoder()
        le_p.fit(pigs[:, 0])

        for p in le_p.classes_:
            idx_p = np.where(pigs[:, 0] == p)
            le_b.fit(np.squeeze(pigs[idx_p, 1]))
            for b in le_b.classes_:
                idx = np.where((pigs[:, 0] == p) & (pigs[:, 1] == b))
                mx = np.mean(X[idx, :, :], axis=(0, 1,2))
                X[idx, :, :] = (X[idx, :, :] - mx)/s


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

    # todo RAUS
 #   files = files[:1500]

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

    # todo RAUS
    files = files[0:1500]
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
        bWeighting=False,
        bHP_EIT=False,
        bLP_EIT=False,
        useReziprok="none",
        loadVent="none",
        reorder_mea= False,
        newShape=False,
        getLengthSig=False,
        iLeaveOut=1
):
    """
    Loads the preprocessed NPZ files with data
    :return: X = EIT data [num segments x eit_length=64 x 1024(or number indices) x 1], y=aorta data parameters [numsegments x para_len]
             pig = Pig info [numsegments x 6] with [Pig, block, index, len, study..]
    """
    #todo HP eit
    # initialize data lists
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

    print(len(X))
    print(X[0].shape)
    max_len_eit = 0
    min_len_eit = 1000
    for j in range(len(X)):
        w = len(X[j])
        if w > max_len_eit:
            max_len_eit = w
        if w < min_len_eit:
            min_len_eit = w
    print("Min length of EIT segments " + str(min_len_eit))
    print("Max length of EIT segments " + str(max_len_eit))

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

        #only in combi with vsig
        if getLengthSig:
            lsig = np.array(get_segment_length(X))
            lsig = lsig[:, np.newaxis]
            vsig= np.append(vsig, lsig, axis=1)
            print(vsig.shape)

    # create index for shuffling
    N = len(y)

    # Use the reziprok channels by
    #  useReziprok= "none"
    if useReziprok == "average":
        X = average_reziprok_channels(X)
        sUseIndex="NonReziprok"
    elif useReziprok=="merge":
        # FOR MERGING THE EIT_LENGTH HAS TO BE DOUBLED FROM THE CONFIG FILE; BECAUSE THE SEGMENT LENGTH IS DOUBLED
        X = merge_reziprok_chas(X)
        sUseIndex="none"




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
       # shuffle = np.random.randint(N, size=N)
    else:
        shuffle = range(N)

    if sUseIndex != "none":
        X = use_indices(sUseIndex, X)

    if reorder_mea:
        X = reorder_meaindex(X)

    # if requested normalize EIT signals
    if norm_eit != 'none':
        X = normalize_eit(X, np.array(pigs), norm_eit)



    X = X[:, :, :, np.newaxis]

    if resample_paras:
        # only for lin paras
        y = resample_paras_aorta(y, aorta_length)


    # pre-process aorta signals not possible due to parameters
    y = np.array(y)

   # if norm_aorta != 'none':
      #  normalize_aorta(y, norm_aorta)
    pigs = np.array(pigs)
    # if requested normalize aorta signals
    #  if norm_aorta:
    #     y = normalize_aorta(y)
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





# ------------------------------------------------------------------------------------------------------------------- #
# Index functions functions
# ------------------------------------------------------------------------------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------- #
def use_indices(type, X):
    """
    Use only certain of the 1024 EIT channels
    :param type: Describes the type, after which is decided which indices to use
    :param X: EIT input data [segments x time per segment x 1024]
    :return: X = EIT with reduced indices [segments x time per segment x numIndices <1024]
    """
    print(type)
    ind = get_indices(type)
    if ind.any() != None:
        X = X[:, :, ind]
    return X


# ------------------------------------------------------------------------------------------------------------------- #
def get_indices(type):
    """
    Gets index numbers to use for EIT data
    :param type: Which selection of indices should happen
    :return: The indices to use (python counting style beginning with zero)
    """
    if type == "FirstHalf":
        ind = np.arange(0, 512)
    elif type == "NonReziprok":
        ind = calc_non_reziprok_ind(32)
    elif type == "AortaIndex":
        ind = load_index("nn/indices/aorta_index.json", "aorta_index")
    elif type == "AortaIndexGuard":
        ind = load_index("nn/indices/aorta_index.json", "aorta_index_guard")
    elif type == "CrossIndex":
        ind = load_index("nn/indices/crosscorr_index.json", "above70")
    elif type == "CrossIndexGuard":
        ind = load_index("nn/indices/crosscorr_index.json", "above60")
    elif type == "FreqIndex":
        ind = load_index("nn/indices/freq_index.json", "FreqIndex")
    elif type == "AutocorrIndex":
        ind = load_index("nn/indices/autocorr_index.json", "under600")
    elif type == "AutocorrIndexGuard":
        ind = load_index("nn/indices/autocorr_index.json", "under800")
    elif type == "NonInjection":
        ind2, ind = calc_non_injection_ind()
    elif type == "NonInjectionGuard":
        ind, ind2 = calc_non_injection_ind()
    elif type == "NonInjectionMid":
        ind = calc_non_injection_ind_single("mid")
    elif type == "NonInjectionLeft":
        ind = calc_non_injection_ind_single("left")
    elif type == "NonInjectionRight":
        ind = calc_non_injection_ind_single("right")
    elif type == "NonInjectionLeftRight":
        ind = calc_non_injection_ind_single("right")
        ind2 = calc_non_injection_ind_single("left")
        ind = np.array(list(set(ind) & set(ind2)))
    elif type == "NonInjectionMidRight":
        ind = calc_non_injection_ind_single("right")
        ind2 = calc_non_injection_ind_single("mid")
        ind = np.array(list(set(ind) & set(ind2)))
    elif type == "NonInjectionMidLeft":
        ind = calc_non_injection_ind_single("mid")
        ind2 = calc_non_injection_ind_single("left")
        ind = np.array(list(set(ind) & set(ind2)))

    elif type == "VisualIndex":
        ind = load_index("nn/indices/visual_index.json", "EITindex")
    elif type == "VisualIndexGuard":
        ind = load_index("nn/indices/visual_index.json", "EITindexguard")
    elif type == "NNbasedIndex":
        ind = load_index("nn/indices/cha_selection.json", "NNbasedIndex")
    elif type == "SingleBolusIndex":
        ind = load_index("nn/indices/cha_selection.json", "BolusP07Index")
    elif type == "BolusIndex":
        ind = load_index("nn/indices/cha_selection.json", "BolusCombiIndex")
    elif type == "NNbasedIndex03":
        ind = load_index("nn/indices/cha_selection03.json", "NNbasedIndex")
    elif type == "SingleBolusIndex03":
        ind = load_index("nn/indices/cha_selection03.json", "BolusP07Index")
    elif type == "BolusIndex03":
        ind = load_index("nn/indices/cha_selection03.json", "BolusCombiIndex")
    elif type == "NNbasedIndexFreq":
        ind = load_index("nn/indices/cha_selection_revised.json", "NNbasedIndexFreq")
    elif type == "NNbasedIndexCross":
        ind = load_index("nn/indices/cha_selection_revised.json", "NNbasedIndexCross")
    elif type == "NNbasedIndexVisu":
        ind = load_index("nn/indices/cha_selection_revised.json", "NNbasedIndexVisu")
    else:
        ind = np.array([None])
    return ind


# ------------------------------------------------------------------------------------------------------------------- #
def calc_non_reziprok_ind(n, bKeepInj=True):
    """
    Calc indices for only non-reziprok channels
    :param n: number of electrodes
    :return: array with indices
    """
    l = []
    if bKeepInj:
        start = 0
    else:
        start = 1
    for i in range(32):
        for k in range(start, n):
            number = int(i * 32 + k)
            l.append(number)
        start += 1
    l = np.array(l)
    return l


# ------------------------------------------------------------------------------------------------------------------- #
def calc_non_injection_ind():
    """
    Calc indices for only measurement=non-injeciton channels
    :return: array with indices, and array with guard indices
    """
    injection_ind = []
    injection_ind_guard = []
    for i in range(32):
        offset = i * 32
        for k in [i, i + 5, i - 5]:
            if k < 0:
                k += 32
            elif k > 31:
                k -= 32
            injection_ind.append(offset + k)
            injection_ind_guard.append((offset + k))
        for k in [i - 1, i + 1, i + 6, i + 4, i - 6, i - 4]:
            if k < 0:
                k += 32
            elif k > 31:
                k -= 32
            injection_ind_guard.append((offset + k))

    injection_ind = sorted(injection_ind, reverse=True)
    injection_ind_guard = sorted(injection_ind_guard, reverse=True)
    inj_ind = list(np.arange(0, 1024))
    inj_ind_guard = list(np.arange(0, 1024))
    for j in injection_ind:
        del inj_ind[j]
    for j in injection_ind_guard:
        del inj_ind_guard[j]

    return np.array(inj_ind), np.array(inj_ind_guard)
# ------------------------------------------------------------------------------------------------------------------- #
def calc_non_injection_ind_single(type="mid"):
    """
    Calc indices for only measurement=non-injeciton channels
    :return: array with indices, and array with guard indices
    """
    def get_i(i, type):
        if type=="mid":
            return i
        elif type=="left":
            return i-5
        else:
            return i+5
    injection_ind = []
    for i in range(32):
        offset = i * 32
        k = get_i(i, type)
        if k < 0:
            k += 32
        elif k > 31:
            k -= 32
        injection_ind.append(offset + k)
    injection_ind = sorted(injection_ind, reverse=True)
    inj_ind = list(np.arange(0, 1024))
    for j in injection_ind:
        del inj_ind[j]

    return np.array(inj_ind)

# ------------------------------------------------------------------------------------------------------------------- #
def load_index(path, name):
    """
    Load a file with stored EIT indices
    :param path: path to loaf
    :param name: FIle name
    :return: Array of indices
    """
    try:
        with open(path, "r") as file:
            config = json.load(file)
        return np.array(config[name])
    except:
        return np.array([None])



# ------------------------------------------------------------------------------------------------------------------- #
def reorder_meaindex(X):
    """
    Perform weighting to the input matrix
    :param X: Input data EIT matrix [num time x 1024]
    :param weight_vec: Weight vector [1024]
    :return: Weighted matrix X
    """
    new_Idx = np.zeros(1024, dtype=int)
    idx = 0
    for j in range(32):
        for i in range(32):
            new_Idx[idx] = i*32+j
            idx+=1
    X = X[:, :, new_Idx]
    return X
