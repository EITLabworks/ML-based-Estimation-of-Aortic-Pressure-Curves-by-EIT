#  Author Patricia Fuchs
import numpy as np
import matplotlib.pyplot as plt
import h5py
from nn.util_paras import resample_eit, resample_aorta, highpass_eit, lowpass_eit, normalize_eit
from channelchart.data_preproc import create_timestamps
from parametrization.save_paras import load_stored_pigmat
from sklearn.preprocessing import LabelEncoder


# ------------------------------------------------------------------------------------------------------------------- #
# Loading of data --------------------------------------------------------------------------------------------------- #
def load_data(file, pnum, block, name):
    data = file.get('PigData')
    fs = np.array(file.get('PigData/f_ADInstruments'))
    if fs == None:
        fs = 1000.0
    #  d = data["P01"]
    try:
        data = data[pnum]
        data = data[block]
        loaded_data = data[name]
        loaded_data = np.array(loaded_data)
    except:
        loaded_data = np.array([None])

    return loaded_data, fs


def segment_block(fseg, signal,name, pnum, block):
    # non eit
    signal = signal.flatten()
    if signal.any() == None:
        print("\n" + pnum + str(" fail ") + block)
        return np.array([None]), None, np.array([None]), None
    else:
        print("\nStart loading "+name + " "+pnum + " - " + block)

    #todo
    aorta_index, aorta_len, eit_index, eit_len, aorta_rejected_index, aorta_rejected_len = load_segmentation_index(
        fseg, pnum, block, True, 0)
    Nsegments = 0
    segments = []
    index_used = []
    len_used = []

    if aorta_index.any() == None:
        return np.array(aorta_index), segments, aorta_len, Nsegments
    for i in range(len(aorta_index)):
        if aorta_len[i] > 150:
            s = signal[aorta_index[i]:aorta_index[i] + aorta_len[i]]
            segments.append(s)
            Nsegments += 1
            index_used.append(aorta_index[i])
            len_used.append(aorta_len[i])

    return np.array(index_used), segments, np.array(len_used), Nsegments
    # data sampled with fsEIT


def load_pigs_segments(pathData, pathSeg, piglist,blocklist, nameLoading, iLeaveOut=1,  bResampleData=False, iScaleFactor=1):
    seglist = []
    starts = []
    for pig in piglist:
        fname = pig + "_kalibrierteRuhephasen.mat"
        fpath = pathData + fname
        print("Loading data from {0}".format(fpath))
        f = h5py.File(fpath, 'r')

        fSegname = pathSeg + pig + "_Ruhephasen_HB_Segments.mat"
        fSeg = h5py.File(fSegname, "r")

        starts.append(len(seglist))
        for block in blocklist:
            #load data
            signal, fs = load_data(f, pig, "Block"+ block, nameLoading)
            if signal.any()== None:
                return None
            #segments data
            signal_index, segs, signal_len, Nseg = segment_block(fSeg, signal, nameLoading, pig, "Block"+block)

            if signal_index.any() == None:
                continue
            seglist = seglist + segs
            plot_segments(segs)


    if iScaleFactor != 1:
        seglist = [item / iScaleFactor for item in seglist]

    #  if bHPEIT:
    #  eit_seglist = highpass_eit(eit_seglist)
    #  if bLPEIT:
    #  eit_seglist = lowpass_eit(eit_seglist)

    starts.append(len(seglist))
    starts = np.array(starts)
    if iLeaveOut > 1:
        seglist = seglist[::iLeaveOut]
        starts = starts//iLeaveOut

    if bResampleData:
        seglist = resample_aorta(seglist, 1024)


    for j in range(len(seglist)):
        seglist[j] = seglist[j].astype(np.float32)

    return seglist, starts




# ------------------------------------------------------------- #
def load_segmentation_index(file, pnum, block, bReduceIndexMatlab=False, iSyncToUncut=0):
    """
    Returns the pre-segmented indices and segment lengths for the EIT and Aorta data
    """
    data = file.get('PigData')
    try:
        data = data[pnum]
        data = data[block]
        aorta_index = np.array(data["Aorta_Segs"], dtype=int).flatten()
        aorta_len = np.array(data["Aorta_Seglen"], dtype=int).flatten()
        eit_index = np.array(data["Heartbeat_Segs"], dtype=int).flatten()
        eit_len = np.array(data["Heartbeat_Seglen"], dtype=int).flatten()
        aorta_rejected_index = np.array(data["Aorta_Segs_Rejected"], dtype=int).flatten()
        aorta_rejected_len = np.array(data["Aorta_Seglen_Rejected"], dtype=int).flatten()

        if iSyncToUncut != 0:
            eit_index = np.add(eit_index, -(iSyncToUncut - 1))

        if bReduceIndexMatlab:
            aorta_index = np.add(aorta_index, -1)
            eit_index = np.add(eit_index, -1)
            aorta_rejected_index = np.add(aorta_rejected_index, -1)
    except:
        aorta_index = np.array([None])
        aorta_len = np.array([None])
        eit_index = np.array([None])
        eit_len = np.array([None])
        aorta_rejected_index = np.array([None])
        aorta_rejected_len = np.array([None])

    return aorta_index, aorta_len, eit_index, eit_len, aorta_rejected_index, aorta_rejected_len


# ------------------------------------------------------------- #
def load_segmentation_block(file, pnum, block):
    """
    Returns the pre-segmented indices and segment lengths for the EIT and Aorta data
    """
    data = file.get('PigData')
    d = dict()
    try:
        data = data[pnum]
        data = data[block]
        d["Heartbeat_Segs"] = np.array(data["Heartbeat_Segs"], dtype=int).flatten()
        d["Heartbeat_Seglen"] = np.array(data["Heartbeat_Seglen"], dtype=int).flatten()
        d["Heartbeat_Comment"] =data["Heartbeat_Comment"]
        d["Aorta_Segs_Rejected"] = np.array(data["Aorta_Segs_Rejected"], dtype=int).flatten()
        d["Aorta_Seglen_Rejected"] = np.array(data["Aorta_Seglen_Rejected"], dtype=int).flatten()
        d["Aorta_Segs_Rejected_Comment"] = data["Aorta_Segs_Rejected_Comment"]
        d["Aorta_Segs"] = np.array(data["Aorta_Segs"], dtype=int).flatten()
        d["Aorta_Seglen"] = np.array(data["Aorta_Seglen"], dtype=int).flatten()
        d["Aorta_Comment"] = data["Aorta_Comment"]
        return d
    except:
        return dict()

def use_eit_offsets(fSeg, pnum, b, eit_seg, eit_index):
    data = fSeg.get('PigData')
    try:
        data = data[pnum][b]
        bOffsetEIT = int(data["bEITOffset"][0])
        bDeleteBlock = int(data["bDeleteBlock"][0])
    except:
        return False, eit_seg
    if bDeleteBlock == 1:
        return True, eit_seg
    elif bOffsetEIT == 0 and bDeleteBlock == 0:
        return False, eit_seg
    try:
        eit_start_index= int(data["EIT_StartIndex"][0])
        eit_end_index = int(data["EIT_EndIndex"][0])
        offset_chas = np.array(data["OffsetChannels"]).flatten()
        m_offset = np.array(data["Channel_MeanOffset"]).flatten()
    except:
        return False, eit_seg
    print("Changing EIT offset")
    for i in range(len(eit_seg)):
        if eit_start_index <= eit_index[i] < eit_end_index:
            eit_seg[i] = eit_seg[i]+ m_offset

    return False, eit_seg

def load_block(p, block, f, fseg, bLP= False, bHP=False, nameLoading="Aorta"):
    aorta, fs_ecg = load_data(f, p, block, nameLoading)
    aorta = aorta.flatten()
    eit_data, fs_eit = load_data(f, p, block, "EIT_Voltages")
    if bHP:
        eit_data= highpass_eit([eit_data], f)
        eit_data = eit_data[0]
    if bLP:
        eit_data = lowpass_eit([eit_data], f)
        eit_data = eit_data[0]
    print(len(eit_data))

    if aorta.any() == None:
        print("\n" + p + str(" fail ") + block)
        return np.array([None]), None, np.array([None]), None, None, None, None, None, None, None
    else:
        print("\nStarting " + p + " - " + block)

    if p == "P09" and block == "09":
        return np.array([None]), None, np.array([None]), None, None, None, None, None, None, None
        # Block segmentation: Load segmentation indices or own segmentation

    # Load segmentation for Aorta
    aorta_index, aorta_segs, aorta_len, Nseg_aorta = segment_with_index(f, fseg, aorta, p, block, True, True)
    if aorta_index.any() == None:
        return np.array([None]), None, np.array([None]), None, None, None, None, None, None, None
    # Load segmentation for eit
    eit_index, eit_seg, eit_len, Nseg_eit, aorta_index_eit = segment_with_index_eit(fseg, eit_data, aorta, p, block)
    if eit_index.any() == None:
        return np.array([None]), None, np.array([None]), None, None, None, None, None, None, None

    print("With eit offset")
    bdelete, eit_seg = use_eit_offsets(fseg, p, block, eit_seg, eit_index)


    keep_eit, keep_aorta = compare_sort(aorta_index_eit, aorta_index)
    eit_index, eit_seg, eit_len, Nseg_eit, aorta_index, aorta_segs, aorta_len, Nseg_aorta = update_eit_and_aorta(
        keep_eit, keep_aorta, eit_index, eit_seg, eit_len, Nseg_eit, aorta_index, aorta_segs, aorta_len, Nseg_aorta)
    return eit_data, aorta, eit_index, eit_seg, eit_len, Nseg_eit, aorta_index, aorta_segs, aorta_len, Nseg_aorta



def load_pigs(pathData, pathSeg, piglist,iLeaveOut=1,  bResampleData=False, iScaleFactor=1, bCreateTimeStamps=False, normData="none", bHPEIT=False, bLPEIT=False):
    eit_seglist = []
    aorta_seglist = []
    blocks = ["01", "02", "03", "04", "05", "06", "07", "08", "09"]
    blocks = ["06"]         # zweiter Versuch was 02; erster 06, norm 07,09
    starts = []
    timestamps = []
    tcurrent = 0
    metadata = []
    for pig in piglist:
        fname = pig + "_kalibrierteRuhephasen.mat"
        fpath = pathData + fname
        print("Loading data from {0}".format(fpath))
        f = h5py.File(fpath, 'r')

        fSegname = pathSeg + pig + "_Ruhephasen_HB_Segments.mat"
        fSeg = h5py.File(fSegname, "r")
        starts.append(len(eit_seglist))

        for block in blocks:
            eit_data,aorta, eit_index, eit_seg, eit_len, Nseg_eit, aorta_index, aorta_segs, aorta_len, Nseg_aorta=load_block(pig, "Block"+block, f,fSeg, bLP=bLPEIT, bHP=bHPEIT)
          #  eit_data, aorta, eit_index, eit_seg, eit_len, Nseg_eit, aorta_index, aorta_segs, aorta_len, Nseg_aorta = load_block(
               # pig, "Block" + block, f, fSeg)

            if eit_index.any()==None:
                continue
            eit_seglist = eit_seglist+eit_seg
            metadata = metadata + Nseg_eit*[pig+block]
            aorta_seglist = aorta_seglist + aorta_segs
            if bCreateTimeStamps:
                ts = create_timestamps(aorta_index, 1000, tcurrent)
                timestamps = timestamps + ts   # in seconds
                tcurrent = timestamps[-1]+1


    if iScaleFactor!=1:
        eit_seglist = [item/iScaleFactor for item in eit_seglist]

  #  if bHPEIT:
      #  eit_seglist = highpass_eit(eit_seglist)
  #  if bLPEIT:
      #  eit_seglist = lowpass_eit(eit_seglist)

    starts.append(len(aorta_seglist))
    starts = np.array(starts)
    if iLeaveOut>1:
        eit_seglist = eit_seglist[::iLeaveOut]
        aorta_seglist = aorta_seglist[::iLeaveOut]
        metadata = metadata [::iLeaveOut]
        if bCreateTimeStamps:
            timestamps = timestamps[::iLeaveOut]
        starts = starts//iLeaveOut

    if bResampleData:
        eit_seglist = resample_eit(eit_seglist, 64)
        aorta_seglist = resample_aorta(aorta_seglist, 1024)

  #  plt.plot(eit_seglist[0][:, 9])
 #   plt.grid()
   # plt.show()


    if normData!= "none":
        eit_seglist = np.array(eit_seglist)

        if normData == 'global':
            mx = np.mean(eit_seglist, axis=(0, 1))
            sx = np.std(eit_seglist, axis=(0, 1))
            eit_seglist = (eit_seglist - mx) / sx

        le_p = LabelEncoder()
        le_p.fit(metadata)

        for p in le_p.classes_:
            idx = np.where(np.array(metadata )== p)
            if normData=="block":
                mx = np.mean(eit_seglist[idx,:,:], axis=(0, 1))
                sx = np.std(eit_seglist[idx, :, :], axis=(0, 1))
                eit_seglist[idx,:,:] = (eit_seglist[idx,:,:] - mx)/sx

            elif normData == "blockM":
                    mx = np.mean(eit_seglist[idx, :, :], axis=(0, 1))
                    eit_seglist[idx, :, :] = (eit_seglist[idx, :, :] - mx)

            elif normData=="block1024":
                mx = np.mean(eit_seglist[idx,:,:], axis=(0, 1,2))
                sx = np.std(eit_seglist[idx, :, :], axis=(0, 1,2))
                eit_seglist[idx,:,:] = (eit_seglist[idx,:,:] - mx)/sx
            elif normData=="block1024M":
                mx = np.mean(eit_seglist[idx,:,:], axis=(0, 1,2))
                eit_seglist[idx,:,:] = (eit_seglist[idx,:,:] - mx)

            elif normData=="block2":
                mx = np.mean(eit_seglist[idx,:,:], axis=(0, 1))
                sx = np.std(eit_seglist[idx, :, :])
                eit_seglist[idx,:,:] = (eit_seglist[idx,:,:] - mx) / sx

            elif normData=="block2_1024":
                mx = np.mean(eit_seglist[idx,:,:], axis=(0, 1,2))
                sx = np.std(eit_seglist[idx, :, :])
                eit_seglist[idx,:,:] = (eit_seglist[idx,:,:] - mx) / sx

            elif normData=="blockmoving":
                idx= np.array(idx).flatten()
                moving_window = 10
                i = 0
                for i in range(0, len(idx) - moving_window, moving_window):
                    idx_windowed = idx[i:i + moving_window]
                    mx = np.mean(eit_seglist[idx_windowed, :, :], axis=(0, 1))
                    eit_seglist[idx_windowed, :, :] = (eit_seglist[idx_windowed, :, :] - mx)

                i = i + moving_window
                idx_windowed = idx[i:]
                mx = np.mean(eit_seglist[idx_windowed, :, :], axis=(0, 1))
                eit_seglist[idx_windowed, :, :] = (eit_seglist[idx_windowed, :, :] - mx)

                sx = np.std(eit_seglist[idx, :, :])
                eit_seglist[idx, :, :] = (eit_seglist[idx, :, :]) / sx

 #   plt.plot(eit_seglist[0][:, 9])
 #   plt.grid()
  #  plt.show()

    for j in range(len(eit_seglist)):
        eit_seglist[j] =eit_seglist[j].astype(np.float32)
        aorta_seglist[j] =aorta_seglist[j].astype(np.float32)

    if bCreateTimeStamps:
        return eit_seglist, aorta_seglist, starts, timestamps
    return eit_seglist, aorta_seglist, starts


def load_paras(paraPath, piglist,iLeaveOut=1, paratype="Linear"):

    blocks = ["01", "02", "03", "04", "05", "06", "07", "08", "09"]
    blocks = ["06"]

    aorta_paras= []
    for pig in piglist:

        fpath = paraPath + pig + "_kalibrierteRuhephasen_" + paratype + "model.mat"
        print("Loading para from {0}".format(fpath))
      #  fPara = h5py.File(fpath, 'r')
        fP = load_stored_pigmat(fpath, paratype)

        for block in blocks:
            b = "Block" +block
            aorta_p = fP["PigData"][pig][b]["AortaParameters"]
            aorta_paras = aorta_paras + list(aorta_p)

    if iLeaveOut>1:
        aorta_paras = aorta_paras[::iLeaveOut]
    return np.array(aorta_paras)



def preprocess_eit_data(eit_segs, bHPEIT=False, bLPEIT=False, normData="block", resample_data=True):
    if bHPEIT:
        eit_segs = highpass_eit(eit_segs)

    if bLPEIT:
        eit_segs = lowpass_eit(eit_segs)

    if resample_data:
        eit_segs = resample_eit(eit_segs, 64)


    if normData!=None:
        if normData=="block":
            mx = np.mean(eit_segs, axis=(0, 1))
            sx = np.std(eit_segs, axis=(0, 1))
            eit_segs= (eit_segs - mx) / sx
        elif normData=="block2":
            mx = np.mean(eit_segs, axis=(0, 1))
            sx = np.std(eit_segs)
            eit_segs= (eit_segs - mx) / sx
    for j in range(len(eit_segs)):
        eit_segs[j] = eit_segs[j].astype(np.float32)
    return eit_segs



# ------------------------------------------------------------------------------------------------------------------- #
# Segmentation ------------------------------------------------------------------------------------------------------ #
def segment_with_index(fdata, fsegment, data, pnum, block, bAorta, biSyncCut=True):
    if biSyncCut:
        cuts, fs = load_data(fdata, pnum, block, "CutsEIT")
        iSync = int(cuts[0][0])
    else:
        iSync = 0
    aorta_index, aorta_len, eit_index, eit_len, aorta_rejected_index, aorta_rejected_len = load_segmentation_index(
        fsegment, pnum, block, True, iSync)
    Nsegments = 0
    segments = []
    index_used = []
    len_used = []
    # data sampled with fsAdinstruments
    if bAorta:
        if aorta_index.any() == None:
            return np.array(aorta_index), segments, aorta_len, Nsegments
        for i in range(len(aorta_index)):
            if aorta_len[i] > 150:
                s = data[aorta_index[i]:aorta_index[i] + aorta_len[i]]
                segments.append(s)
                Nsegments += 1
                index_used.append(aorta_index[i])
                len_used.append(aorta_len[i])

        return np.array(index_used), segments, np.array(len_used), Nsegments
    # data sampled with fsEIT
    else:
        aorta_index = np.append(aorta_index, aorta_rejected_index)
        if aorta_index.any() != None:
            aorta_index = np.sort(aorta_index)
            for i in range(len(eit_index)):
                s = data[eit_index[i]:eit_index[i] + eit_len[i]]
                segments.append(s)
                Nsegments += 1

            return eit_index, segments, eit_len, Nsegments, aorta_index
        else:
            return np.array([None]), np.array([None]), np.array([None]), 0, aorta_index


# ------------------------------------------------------------- #
def segment_with_index_eit(fsegment, data_eit, data_aorta, pnum, block):
    aorta_index, aorta_len, eit_index, eit_len, aorta_rejected_index, aorta_rejected_len = load_segmentation_index(
        fsegment, pnum, block, True, False)
    Nsegments = 0
    segments = []
    index_used = []
    len_used = []
    aorta_index_used = []

    ds_r = data_aorta.shape[0] / data_eit.shape[0]
    if aorta_index.any() != None:
        for i in range(len(aorta_index)):
            if aorta_len[i] > 150:
                start_aorta = aorta_index[i]
                end_aorta = aorta_index[i] + aorta_len[i]
                eit_ind = int(start_aorta / ds_r)
                eit_end = int(end_aorta / ds_r)
                eit_len = eit_end - eit_ind
                s = data_eit[eit_ind:eit_end]
                segments.append(s)
                Nsegments += 1
                index_used.append(eit_ind)
                aorta_index_used.append(aorta_index[i])
                len_used.append(eit_len)
        return np.array(index_used), segments, np.array(len_used), Nsegments, np.array(aorta_index_used)
    else:
        return np.array([None]), np.array([None]), np.array([None]), 0, aorta_index

# ------------------------------------------------------------- #
def segment_with_index_eit_total(aorta_index,aorta_len, data_eit, data_aorta, bSubtract1=False):
    if bSubtract1:
        aorta_index = aorta_index -1
    Nsegments = 0
    index_used = []
    len_used = []

    ds_r = data_aorta.shape[0] / data_eit.shape[0]
    if aorta_index.any() != None:
        for i in range(len(aorta_index)):
                start_aorta = aorta_index[i]
                end_aorta = aorta_index[i] + aorta_len[i]
                eit_ind = int(start_aorta / ds_r)
                eit_end = int(end_aorta / ds_r)
                eit_len = eit_end - eit_ind
                Nsegments += 1
                index_used.append(eit_ind)
                len_used.append(eit_len)
        return np.array(index_used), np.array(len_used), Nsegments,
    else:
        return np.array([None]), np.array([None]), 0


def get_ventilation_signal(fSeg,  pnum, block, type="middle"):
    aorta_index, aorta_len, eit_index, eit_len, aorta_rejected_index, aorta_rejected_len = load_segmentation_index(
        fSeg, pnum, block, True, False)
    data = fSeg.get('PigData')
    try:
        if type=="middle":
            vents = np.array(data[pnum][block]["Ventilation_MidSeg"]).flatten()
        else:
            vents = np.array(data[pnum][block]["Ventilation_StartSeg"]).flatten()
    except:
        return np.zeros(len(aorta_index))
    vents_used = []
    for k in range(len(aorta_index)):
        if aorta_len[k]>150:
            vents_used.append(vents[k])
    return np.array(vents_used)


# ------------------------------------------------------------- #
# todo
def segment_eit_aorta(eit, posMin, segLen, fsAorta, fsEit):
    eit_seg_pos = []
    eit_len = []
    last_seg = -5
    max_eit_seg = len(eit) - 1
    for j in range(len(posMin)):
        m = posMin[j]
        m_eit = int(fsEit * m / fsAorta)
        if m_eit <= last_seg:
            print("cuts")
            print(str(m_eit) + "   " + str(last_seg))
            m_eit = last_seg + 1
        l_eit = int(fsEit * segLen[j] / fsAorta)
        eit_len.append(l_eit)
        eit_seg_pos.append(m_eit)
        last_seg = m_eit + l_eit - 1

    if last_seg > max_eit_seg:
        error = last_seg - max_eit_seg
        print("error " + str(error))
        eit_len[-1] = eit_len[-1] + error

    return np.array(eit_seg_pos), np.array(eit_len)
    # Eigene Segmentierung für bestimmte Blöcke laufen lassen, so umwandeln wie Henryks format. Andere Blöcke ergänzen


# ------------------------------------------------------------- #
def segment_aorta_dual(bLoadSegments, aorta, hr, fdata, fsegment, pig, block, minlength, mindist, bplot):
    if bLoadSegments:
        positionsMinima, aorta_seg, segment_widths, Nsegments = segment_with_index(fdata, fsegment, aorta, pig, block,
                                                                                   True, False)
        if bplot:
            if positionsMinima.any() != None:
                plot_segmentation(segment_widths, Nsegments, aorta_seg)
    else:
        positionsMinima, aorta_seg, segment_widths, Nsegments = segment_aorta_with_hr(minlength, mindist, aorta, hr,
                                                                                      bplot)
    return positionsMinima, aorta_seg, segment_widths, Nsegments


# ------------------------------------------------------------- #
def segment_aorta_with_hr(minlength, mindistance, aorta_sig, hr, bplot=True):
    Nsegments = int(aorta_sig.shape[0] / minlength * 3)  # Maxnumber of segments
    posMinima = np.zeros(Nsegments, dtype=int)  # Positions of Minima
    posMaxima = np.zeros(Nsegments, dtype=int)  # Positions of Maxima
    seg_widths = np.zeros(Nsegments)  # Width of found segments
    aorta_seg = []  # Found segments
    segments = np.zeros((Nsegments, 2), dtype=int)  #
    hr = hr.flatten()
    # find first minimum
    posMinima[0] = np.argmin(aorta_sig[:minlength])
    posMaxima[0] = np.argmax(aorta_sig[:minlength])
    cont = True
    i = 1
    too_long = []
    while cont:
        # Get current signal
        min_di, max_di = determine_min_and_maxlength(
            hr[np.max([posMinima[i - 1], 0]):np.min([int(posMinima[i - 1] + 100), len(aorta_sig)])])
        curr_seg = aorta_sig[posMinima[i - 1] + min_di: posMinima[i - 1] + max_di]

        # Find exact position of minimum (start of segment)
        # find exact start of segment by looking for strongest rise, then step back in time
        # to its zero crossing, which is the minimum near the strongest rise in amplitude
        segdiffs = curr_seg[1:] - curr_seg[:-1]  # 1st order derivative of aorta sig segment
        strongest = np.argmax(segdiffs)  # strongest rise
        pos = strongest
        while segdiffs[pos] >= 0:  # go back to find zero crossing in derivative
            pos = pos - 1
        posMinima[i] = pos + posMinima[i - 1] + min_di

        # Find position of maximum in segment
        posMaxima[i] = np.argmax(curr_seg) + posMinima[i - 1] + min_di
        # Determine segment widths
        seg_widths[i - 1] = posMinima[i] - posMinima[i - 1]
        aorta_seg.append(aorta_sig[posMinima[i - 1]:posMinima[i]])

        if posMinima[i] + minlength > aorta_sig.shape[0]:
            cont = False
        else:
            i += 1

    # Shorten vectors to actual length
    Nsegments = i
    posMinima = posMinima[:Nsegments + 1]
    posMaxima = posMaxima[:Nsegments + 1]
    seg_widths = seg_widths[:Nsegments]
    if bplot:
        #   plot_segment_width(seg_widths)
        plot_segmentation(seg_widths, Nsegments, aorta_seg)

    return posMinima, aorta_seg, seg_widths, Nsegments


# ------------------------------------------------------------- #
def segment_aorta(minlength, mindistance, aorta_sig, bplot=True):
    Nsegments = int(aorta_sig.shape[0] / minlength * 2)  # Maxnumber of segments
    posMinima = np.zeros(Nsegments, dtype=int)  # Positions of Minima
    posMaxima = np.zeros(Nsegments, dtype=int)  # Positions of Maxima
    seg_widths = np.zeros(Nsegments)  # Width of found segments
    aorta_seg = []  # Found segments
    segments = np.zeros((Nsegments, 2), dtype=int)  #

    # find first minimum
    posMinima[0] = np.argmin(aorta_sig[:minlength])
    posMaxima[0] = np.argmax(aorta_sig[:minlength])
    cont = True
    i = 1

    while cont:
        # Get current signal
        curr_seg = aorta_sig[posMinima[i - 1] + mindistance: posMinima[i - 1] + minlength]
        # Find exact position of minimum (start of segment)
        # find exact start of segment by looking for strongest rise, then step back in time
        # to its zero crossing, which is the minimum near the strongest rise in amplitude
        segdiffs = curr_seg[1:] - curr_seg[:-1]  # 1st order derivative of aorta sig segment
        strongest = np.argmax(segdiffs)  # strongest rise
        pos = strongest
        while segdiffs[pos] >= 0:  # go back to find zero crossing in derivative
            pos = pos - 1
        posMinima[i] = pos + posMinima[i - 1] + mindistance

        # Find position of maximum in segment
        posMaxima[i] = np.argmax(curr_seg) + posMinima[i - 1] + mindistance

        # Determine segment widths
        seg_widths[i - 1] = posMinima[i] - posMinima[i - 1]
        aorta_seg.append(aorta_sig[posMinima[i - 1]:posMinima[i]])

        if posMinima[i] + minlength > aorta_sig.shape[0]:
            cont = False
        else:
            i += 1

    # Shorten vectors to actual length
    Nsegments = i
    posMinima = posMinima[:Nsegments + 1]
    posMaxima = posMaxima[:Nsegments + 1]
    seg_widths = seg_widths[:Nsegments]
    mark_outliners(Nsegments, aorta_seg, seg_widths)
    if bplot:
        plot_segment_width(seg_widths)
        plot_segmentation(seg_widths, Nsegments, aorta_seg)

    return posMinima, aorta_seg, seg_widths, Nsegments


# ------------------------------------------------------------------------------------------------------------------- #
# Plotter ----------------------------------------------------------------------------------------------------------- #
def plot_segment_width(seg_widths):
    a = 500
    b = 600
    plt.figure(figsize=(9, 4))
    plt.plot(seg_widths[:-1], 'b-x')
    plt.xlim(a, b)
    plt.grid()
    plt.title('widths of detected segments')
    print(np.mean(seg_widths[:-1]))


# ------------------------------------------------------------- #
def plot_segments(segs):
    plt.figure(figsize=(15, 9))
    for i in np.arange(len(segs)):
        plt.plot(segs[i], 'b-', linewidth=0.2)

    # plt.plot(aorta_seg_mean, 'r-', linewidth=1)
    plt.grid()
    plt.show()


# ------------------------------------------------------------- #
def plot_segmentation(seg_widths, Nseg, aorta_segs):
    # plot aorta segments
    min_length = int(np.amin(seg_widths[:-1]))
    max_length = int(np.amax(seg_widths[:-1]))

    mean_length = int(np.mean(seg_widths[:-1]))
    std_length = int(np.std(seg_widths[:-1]))
    print("minlength " + str(min_length))
    print("maxlength " + str(max_length))

    print("meanlength " + str(mean_length))
    print("std " + str(std_length))

    plt.figure(figsize=(13, 9))
    l = mean_length - std_length
    aorta_seg_mean = np.zeros(mean_length - std_length)
    for i in np.arange(Nseg - 1):
        plt.plot(aorta_segs[i], 'b-', linewidth=0.1)

        lextend = l - len(aorta_segs[i])
        if lextend > 0:
            ext = np.zeros(lextend)
            seg = np.append(aorta_segs[i][:].flatten(), ext)
        else:
            seg = aorta_segs[i][:l].flatten()
        aorta_seg_mean += seg
    aorta_seg_mean /= Nseg
    plt.plot(aorta_seg_mean, 'r-', linewidth=1)
    plt.grid()
    # plt.title("P05 Block09")
    plt.show()


# ------------------------------------------------------------------------------------------------------------------- #
# Help functions ---------------------------------------------------------------------------------------------------- #
# ------------------------------------------------------------- #
def determine_min_and_maxlength(hr, vari=50):
    hr_max = np.max(hr.flatten())

    min_dist_hr = int(60 * 1000 / hr_max)
    hr_min = np.min(hr)
    max_dist_hr = int(60 * 1000 / hr_min)
    max_dist_hr = np.min([1400, max_dist_hr])
    min_dist_hr = np.min([800, min_dist_hr])
    return min_dist_hr - vari, max_dist_hr + vari


# ------------------------------------------------------------- #
def update_eit_and_aorta(keep_eit, keep_aorta, eit_index, eit_seg, eit_len, Nseg_eit, aorta_index, aorta_seg, aorta_len,
                         Nseg_aorta):
    eit_index = eit_index[keep_eit]
    eit_seg_new = []

    for i in keep_eit:
        eit_seg_new.append(eit_seg[i])
    eit_len = eit_len[keep_eit]
    Nseg_eit = len(eit_index)
    aorta_index = aorta_index[keep_aorta]
    aorta_seg_new = []
    for i in keep_aorta:
        aorta_seg_new.append(aorta_seg[i])
    aorta_len = aorta_len[keep_aorta]
    Nseg_aorta = len(aorta_index)
    return eit_index, eit_seg_new, eit_len, Nseg_eit, aorta_index, aorta_seg_new, aorta_len, Nseg_aorta


# ------------------------------------------------------------- #
def compare_sort(eit_aorta_ind, aorta_ind):
    keep_eit = []
    keep_aorta = []
    eit_idx = 0
    for j in range(len(aorta_ind)):
        while eit_aorta_ind[eit_idx] < aorta_ind[j]:
            eit_idx += 1
        if eit_aorta_ind[eit_idx] == aorta_ind[j]:
            keep_aorta.append(j)
            keep_eit.append(eit_idx)
            eit_idx += 1
            continue
        else:
            print("For the aorta_index {0} was not found in the EIT segmentation".format(aorta_ind[j]))
    return np.array(keep_eit), np.array(keep_aorta)


# ------------------------------------------------------------- #
def mark_outliners(Nsegments, segs, widths):
    Nmarked = 0
    Nunmarked = 0
    marked = []
    seg_unmarked = []
    mean_len = np.mean(widths[:-1])
    std = np.std(widths[:-1])
    # Sort for length
    print(mean_len - 5 * std)
    print(mean_len + 5 * std)
    for i in range(Nsegments):
        if mean_len - 5 * std < widths[i] < mean_len + 5 * std:
            Nunmarked += 1
            seg_unmarked.append(segs[i])
        else:
            Nmarked += 1
            marked.append(segs[i])

    mean_len = int(mean_len)
    std = int(std)
    lmean = mean_len - 1 * std
    aorta_seg_mean = np.zeros(int(mean_len) - 1 * int(std))
    for i in np.arange(Nunmarked):
        #  plt.plot(segs[i], 'b-', linewidth=0.1)
        lextend = lmean - len(seg_unmarked[i])
        if lextend > 0:
            ext = np.zeros(lextend)
            seg = np.append(seg_unmarked[i][:].flatten(), ext)
        else:
            seg = seg_unmarked[i][:lmean].flatten()
        aorta_seg_mean += seg
    aorta_seg_mean /= Nunmarked
    errors = np.zeros(Nunmarked)
    for i in range(Nunmarked):
        l = len(seg_unmarked[i])
        if l < lmean:
            e_vec = np.abs(seg_unmarked[i] - aorta_seg_mean[:l])
        else:
            e_vec = np.abs(seg_unmarked[i][:lmean] - aorta_seg_mean)
        errors[i] = np.sum(e_vec)
        if errors[i] > 0.9 * (10 ** 7):
            r = i
    error_cutoff = np.mean(errors) + 8 * np.std(errors)
    j = 0
    while j < Nunmarked:
        if errors[j] > error_cutoff:
            Nmarked += 1
            marked.append(seg_unmarked[j])
            seg_unmarked = seg_unmarked[:j] + seg_unmarked[j + 1:]
            Nunmarked -= 1
            errors = np.append(errors[:j], errors[j + 1:])
        else:
            j += 1

    plt.plot(errors)
    plt.plot(error_cutoff, "red")
    plt.show()
    plt.figure(figsize=(9, 4))
    for i in np.arange(Nunmarked):
        plt.plot(seg_unmarked[i], 'b-', linewidth=0.1)
    for i in np.arange(Nmarked):
        plt.plot(marked[i], 'g-', linewidth=3)
    # plt.plot(seg_unmarked[r], "orange", linewidth=2)
    plt.plot(aorta_seg_mean, 'r-', linewidth=1)
    plt.grid()
    plt.show()


# ------------------------------------------------------------- #
def split_ecg_like_aorta(minPositions, ecg):
    ecg_segments = []
    for j in range(1, len(minPositions)):
        ecg_segments.append(ecg[minPositions[j - 1]: minPositions[j]])


# ------------------------------------------------------------- #
class ErrorMeasure:
    def __init__(self, limit):
        self.limit = limit

    def mae_vector(self, original, reconst):
        v = np.abs(original - reconst)
        return np.sum(v)

    def mse_vector(self, original, reconst):
        v = (np.abs(original - reconst)) ** 2
        return np.sum(v)

    def diff_min_max(self, original, reconst):
        min_diff = np.abs(np.min(original) - np.min(reconst))
        max_diff = np.abs(np.max(original) - np.max(reconst))
        return min_diff, max_diff

    def calc_error(self, original, reconst):
        return self.mae_vector(original, reconst), self.mse_vector(original, reconst), self.diff_min_max(original,
                                                                                                         reconst)
