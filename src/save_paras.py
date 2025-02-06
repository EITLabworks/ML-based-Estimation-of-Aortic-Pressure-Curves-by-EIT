import numpy as np
import h5py
from scipy.io import savemat

READ_ME_TEXT = "Stored estimated parameters for Cauchy-Lorentz impulses and linear approximation\n\n "+ \
"Nun mit äquidistanten Nullstellen \n Base Files 'P0X_kalibrierteRuhephasen.mat' \n Used Signal 'Aorta' \n \n File structure:\n" +\
"PigData : struct \n     -P0X: struct\n			      - Block01: struct\n						- NSegments : int     # Anzahl der Segmente, in die das Aorta Signal unterteilt wurde\n"+\
"						- ParaType : string   # Art des Schätzungsverfahrens: Linear, CauchyLorentz oder Gaussian\n"\
"						- AortaSeg : array NSegments x SegmentLengths[i]  # Die Original-Aortasegmente, die ausgeschnnitten wurden\n"\
"						- AortaSegPos : array dim= Nsegments: Indizes (Python style; beginned mit 0), die den jeweiligen Anfang der ausgeschnittenen Segmente aus der Originalarray kennzeichen; wichtig zur Synchro von Segmente zu u.A. EIT-Daten\n"\
"						- AortaMean : array dim= Nsegments: Mittelwerte der Segmente\n"\
"						- AortaParameters: array dim = NSegments x Länge der jeweiligen Parametervektoren: hier Aufbau unterschiedlich nach ParaType: siehe unten\n"\
"						- SegmentLengths : array dim= NSegments; Länge der einzelnen Segmente\n"\
"						- NPulses: array dim = NSegments: Anzahl der Pulse je Segmente Anzahl der insgesamt Parameter Linear = 2*Kpulses+2; CL = 4*Kpulses+2; Gaussian 2*Kpulses+3\n"\
"						- NMSE: array dim = NSegments: NMSE der Parameterrekonstruktion zum Originalsegment\n"\
"						- Pulses: array dim = NSegments x NPulses[i]: Linear= XPositionen; CL/Gaussian=Puls \n"\
"						- PulsAmp: array dim = NSegments x NPulses[i]: Linear= YPunkte;\n\n"\
"                       - CL: ak, Gaussian: amplitude\n"\
"						- Sigma: Lin=[], CL= array dim = NSegments x NPulses[i]: sigma; Gaussian = array dim=NSegments: fixed sigma für alle Pulse eines Segments\n"\
"						- Dk: Lin/Gaussian = []; CL = dk = Amplituden zweiter Teilimpuls \n"\
"                       - CurveParas: Hierarchical first curve paras 3-5 paras\n"\
"                       - bResampled: If Aorta Segment has been resampled to a fixed length before parameter estimation\n"\
"                       - iLenAorta: If bResampled,then this is the original segment length \n"\
"			      - Block02: struct\n"\
"			      - Block03: struct\n"\
"			      - ....\n"


# ------------------------------------------------------------------------------------------------------------------- #
def store_block(paratype,Nsegments:int, segments, segPos, K, nmse, data_vec, pulses, amp, sigma, dk,pSegLen,
                curveParas=[[]], bResampled=False):
    """
    Stores all paramters for one block in a dictionary (for .mat storage)
    :param paratype: Type of parametrization
    :param Nsegments: Number of segments within block
    :param segments: Original data segments of aorta pressure
    :param segPos: Positions of the segments (index beginning with =0) in original-aorta array
    :param K: Vec of number of pulses/pieces per segment [Nseg]
    :param nmse: Reached nmse in DB [Nseg]
    :param data_vec: AortaParas [Nseg, len indiviual paras]
    :param pulses: Pulse pos for Pulses and xpoints for lin
    :param amp: Amp for pulses and ypoints for lin
    :param sigma: = 0 for lin, 1D for Gaussian and 2D for CL
    :param dk: only for CL
    :param pSegLen: Length of the segments (index beginning with =0) in original-aorta array
    :param curveParas: only for hierarchical
    :param bResampled: If the aorta segments had been resampled to a fixed length
    :return: Dictionary containing all the info
    """
    segMeans, segLens = calc_segmeans_and_lengths(segments)

    curBlockDict = {'NSegments': Nsegments}
    curBlockDict["ParaType"] = paratype
    curBlockDict['AortaSeg'] = make_append_zero_array(segments)
    curBlockDict['AortaSegPos'] = segPos
    curBlockDict['AortaMean'] = segMeans  # aorta_mean
    curBlockDict["AortaParameters"] = make_append_zero_array(data_vec)
    curBlockDict["SegmentLengths"] = pSegLen

    curBlockDict['NPulses'] = K
    curBlockDict['NMSE'] = nmse
    curBlockDict['Pulses'] = make_append_zero_array(pulses)  # Xpoints oder tk = Positions

    curBlockDict['PulsAmp'] = make_append_zero_array(amp)
    curBlockDict['Sigma'] = make_append_zero_array(sigma)
    curBlockDict['Dk'] = make_append_zero_array(dk)
    curBlockDict["CurveParas"] = curveParas
    curBlockDict["bResampled"] = bResampled
    curBlockDict["iLenAorta"] = segLens[0]
    return curBlockDict


# ------------------------------------------------------------- #
def store_pig_mat(path, type, pigdict):
    """
    Stores the blockdicts into a single PigData Matlab file
    """
    s = "_"+type + "model.mat"
    outfilepath = path.replace('.mat', s)
    g = {"PigData": pigdict}
    # hdf5storage.savemat(path+outfilename, g)
    #  hdf5storage.write(data, path,outfilename, matlab_compatible=True)
    savemat(outfilepath, g,long_field_names=True)
    print("Parameters successfully stored at "+ outfilepath)


# ------------------------------------------------------------- #
def create_blockdict_paras(h5block):
    """
    For reloading the stored data from h5py file -> h5py block to dictionary
    """
    b = dict()
    b["NSegments"] = h5block["NSegments"][0][0]
    b["ParaType"] = h5block["ParaType"][0].tobytes()[::2].decode()
    b["SegmentLengths"] = np.array(h5block["SegmentLengths"]).flatten()

    b['AortaSegPos'] = np.array(h5block["AortaSegPos"]).flatten()
    b['AortaMean'] = np.array(h5block["AortaMean"]).flatten() # aorta_mean
    b["AortaParameters"] = np.transpose(h5block["AortaParameters"])

    b['NPulses'] = np.array(h5block["NPulses"]).flatten()
    b['NMSE'] = np.array(h5block["NMSE"]).flatten()
    b['Pulses'] = np.transpose(h5block["Pulses"])  # # Xpoints oder tk = Positions

    b['PulsAmp'] = np.transpose(h5block["PulsAmp"])
    b['Sigma'] = np.transpose(h5block["Sigma"])
    b['Dk'] = np.transpose(h5block["Dk"])

    keys =h5block.keys()
    if "CurveParas" in keys:
        b["CurveParas"] = np.transpose(h5block["CurveParas"])
    else:
        b["CurveParas"] = [None]

    if "bResampled" in keys:
        b["bResampled"] = bool(h5block["bResampled"])
    else:
        b["bResampled"] = False

    if "iLenAorta" in keys:
        b["iLenAorta"] = int(h5block["iLenAorta"][0][0])
    else:
        b["iLenAorta"] = 0
    aortaseg = np.transpose(h5block["AortaSeg"])
    seglist= []
    if b["bResampled"] == False:
        for k in range(len(aortaseg)):
            seglist.append(aortaseg[k][:int(b["SegmentLengths"][k])])
    else:
        for k in range(len(aortaseg)):
            seglist.append(aortaseg[k][:b["iLenAorta"]])
    b['AortaSeg'] = seglist      # todo shorten to correct len
    return b


# ------------------------------------------------------------- #
def load_stored_pigmat(path, type):
    """
    Loads stored pigmat paramter file
    """
    f = h5py.File(path, "r")
    print(path)
    pdict ={"PigData":{}}
    x = f.get("PigData")

    # iterate over pigs
    for key in x.keys():
        pdict["PigData"].update({key:{}})
        for k in x[key].keys():
            bdict = create_blockdict_paras(x[key][k])
            pdict["PigData"][key].update({k:bdict})
    return pdict


# ------------------------------------------------------------------------------------------------------------------- #
# Help functions ---------------------------------------------------------------------------------------------------- #
def calc_segmeans_and_lengths(segs):
    means = []
    lengths = []
    for i in range(len(segs)):
        s = segs[i]
        means.append(np.mean(s.flatten()))
        lengths.append(len(s.flatten()))
    return means, lengths


# ------------------------------------------------------------- #
def make_append_zero_array(l):
    max_len = len(l[0])
    for i in range(1, len(l)):
        if len(l[i]) > max_len:
            max_len = len(l[i])
    new = []
    for i in range(len(l)):
        q = np.array(l[i])
        app = np.zeros(max_len - len(q))
        q = np.append(q, app)
        new.append(q)
    return np.array(new)
