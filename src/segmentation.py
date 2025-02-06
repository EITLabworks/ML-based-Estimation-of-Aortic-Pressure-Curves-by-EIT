#  Author Patricia Fuchs
import numpy as np


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
def load_block_no_eit(p, block, f, fseg, nameLoading="Aorta"):
    aorta, fs_ecg = load_data(f, p, block, nameLoading)
    aorta = aorta.flatten()

    if aorta.any() == None:
        print("\n" + p + " - " + block + str(" failed."))
        return np.array([None]), np.array([None]), None, np.array([None]), None
    else:
        print("\nStarting " + p + " - " + block)

    if p == "P09" and block == "09":
        return np.array([None]), np.array([None]), None, np.array([None]), None
        # Block segmentation: Load segmentation indices or own segmentation

    # Load segmentation for Aorta
    aorta_index, aorta_segs, aorta_len, Nseg_aorta = segment_with_index(f, fseg, aorta, p, block, True, True)
    if aorta_index.any() == None:
        return np.array([None]), np.array([None]), None, np.array([None]), None
    return  aorta, aorta_index, aorta_segs, aorta_len, Nseg_aorta


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
