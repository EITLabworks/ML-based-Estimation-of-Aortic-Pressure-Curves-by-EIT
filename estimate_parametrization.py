"""
Author:  Patricia Fuchs
"""
import os
from src.parameter_estimation import *
from src.segmentation import load_block_no_eit
from src.save_paras import READ_ME_TEXT
from src.hierarchical import HierarchicalApprox
import warnings
import h5py
import time
warnings.filterwarnings('ignore')

# Estimation methods ------------------------------------------------------------------------------- #
bResampleAorta = True
iAortaLen = 1024

blockname = "Block"
Pigs = ["P01", "P02", "P03", "P04", "P05", "P06", "P07", "P08", "P09", "P10"]
blocks = ["01", "02", "03", "04", "05", "06", "07", "08", "09"]
pressure_sig = "Aorta"

data_path = 'C:/Users/pfuchs/Documents/Data/EIT/PulHypStudie/DataOriginal/'
segmentPath = "C:/Users/pfuchs/Documents/Data/Segmentierung_Heartbeats/PulHyp_Segs_neu3/"

fMatPath = 'C:/Users/pfuchs/Documents/Data/EIT/PulHypStudie/'

fOrigin = "_kalibrierteRuhephasen.mat"
matpaths = {}
bPlotOutliers = True
bResultPlots = True
bSaveToMat = True

fSeg = None
afterError = "Lin"


nmsedBLimit = -30
Klin = 21
Kspecest = 14
Khier = 16
Rval = "abs"
bLin = True
bSpecEst = True
bHier = True



# Start loading config -------------------------------------------------------------------------------------------- #
if bSaveToMat:
    dirname = "Parameters" + time.strftime("%Y%m%d-%H%M%S")
    fMatPath = os.path.join(fMatPath, dirname)
    os.mkdir(fMatPath)
    fMatPath = fMatPath+"/"

    # Open the file in write mode ('w')
    with open(fMatPath+"readme.txt", 'w') as file:
        # Write some text into the file
        file.write("Data path:" + data_path + "\n")
        file.write("Segment path:" + segmentPath + "\n\n")
        file.write("Piecewise Linear Regression with NMSELimit={0} dB and KLin={1} \n".format(nmsedBLimit, Klin))
        file.write("Spectral Estimation with NMSELimit={0} dB and KSpecest={1} and Rval ={2} \n".format(nmsedBLimit, Kspecest, Rval))
        file.write("Hierarchical Approximation with NMSELimit={0} dB and KHier={1} \n".format(nmsedBLimit, Khier))
        file.write(READ_ME_TEXT)

if bHier:
    h = HierarchicalApprox("Poly", "Lin", 0)
    if bSaveToMat:
        dirname = "Paras_k" + str(Khier) + "_DataHierarchical"
        f1 = os.path.join(fMatPath, dirname)
        os.mkdir(f1)
        dirname = dirname+"/"
        matpaths.update({"PolyHierarchical": dirname})


if bLin and bSaveToMat:
    dirname = "Paras_k" + str(Klin) + "_DataLinear"
    f1 = os.path.join(fMatPath, dirname)
    os.mkdir(f1)
    dirname = dirname + "/"
    matpaths.update({"Linear": dirname})

if bSpecEst and bSaveToMat:
    dirname = "Paras_k" + str(Kspecest) + "_DataSpectralEstimation"
    f1 = os.path.join(fMatPath, dirname)
    os.mkdir(f1)
    dirname = dirname + "/"
    matpaths.update({"SpectralEstimation": dirname})



# Main loop ------------------------------------------------------------------------------------------------------- #
for pig in Pigs:
    fname = pig + fOrigin
    fpath = data_path + fname
    print(fpath)
    f = h5py.File(fpath, 'r')
    fSegname = segmentPath + pig + "_Ruhephasen_HB_Segments.mat"
    fSeg = h5py.File(fSegname, "r")

    savetoMat = {"Linear": {}, "SpectralEstimation": {}, "PolyHierarchical":{}}

    for block in blocks:
        b = blockname + block
        aorta, aorta_index, aorta_seg, aorta_len, Nsegments = load_block_no_eit(pig, b, f, fSeg)

        if aorta_index.any() == None:
            print("\n"+pig + " - " + b + str(" failed."))
            continue
        else:
            print("\nStarting " + pig + " - " + b)

        if bResampleAorta:
            aorta_seg, len_vec = resample_aorta(aorta_seg, iAortaLen)
        else:
            len_vec = aorta_len

        aorta_seg = aorta_seg[0:50]
        Nsegments= 50
        # Estimate parametric representations
        if bLin:

            blockmat = estimate_linear_total(Nsegments, aorta_seg, aorta_index, Klin-1, Klin-1, nmsedBLimit,len_vec, bPlot=bResultPlots, bSaveToMat=bSaveToMat, bPlotOutliers=bPlotOutliers, bResampled=bResampleAorta)
            savetoMat["Linear"].update(({b: blockmat}))

        if bSpecEst:
            blockmat = estimate_cauchylorentz_total(Nsegments, aorta_seg, aorta_index,  Kspecest, Kspecest, nmsedBLimit,len_vec, bPlot=bResultPlots, bSaveToMat=bSaveToMat, bPlotOutliers=bPlotOutliers, Rval=Rval, bResampled=bResampleAorta)
            savetoMat["SpectralEstimation"].update(({b: blockmat}))

        if bHier:
            blockmat = estimate_hierarchical_total_linear(h, Nsegments, aorta_seg, aorta_index,
                                                           Khier, Khier, nmsedBLimit, len_vec,
                                                           bPlot=bResultPlots, bSaveToMat=bSaveToMat,
                                                           bPlotOutliers=bPlotOutliers, Rval=Rval,
                                                           bResampled=bResampleAorta)

            savetoMat["PolyHierarchical"].update(({b: blockmat}))

    finish_pig_calculation(fMatPath, matpaths, pig, bSaveToMat, savetoMat)

print("Parametrization finished.")