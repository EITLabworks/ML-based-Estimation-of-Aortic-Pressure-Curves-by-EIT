"""
Author: Volker Kuehn; Patricia Fuchs
"""
# Cauchy-lorentz spectral estimation


import numpy as np
from src.cauchy_lorentz import annihilatingCauchyLorentz, reconstructCauchyLorentzPeriodic, sort_cl_pulses, \
    remove_interval


# ------------------------------------------------------------------------------------------------------------------- #
# Spectral Estimation for Cauchy-Lorentz Pulses --------------------------------------------------------------------- #
# ------------------------------------------------------------- #
def cl_estimation(seg, Kinit=1, Kmax=80, nmsedBLim=-40, interval=0, R=float(1 / 200)):
    """
    Parametrization for minimally Kinit and maximally Kmax CL-pulses, goal reached for NMSE < nmsedBLim
    :return: x_seg_zeromean_hat: Parametric signal, K_pulses: Number of used pulses, pos: Pulse positions,
             sigma: STDs, amp: Amplitudes , dk: 2.Amplitudes, nmse: reached NMSE, nmsedB: Reached NMSE in DB
    """
    K_pulses = Kinit
    # get current segment of aorta signal
    Lseg = len(seg)
    # time axis for current segment
    t = np.arange(Lseg)
    previous = []
    saving = []
    cont = True
    while cont:
        pos, sigma, amp, dk = annihilatingCauchyLorentz(seg, K_pulses, R=R)
        # construct aorta signal from detected pulses considering periodic repetitions (output does not contain repetitions)
        x_seg_zeromean_hat = reconstructCauchyLorentzPeriodic(t, pos, amp, sigma, dk)
        saving.append(x_seg_zeromean_hat)
        # compute normalized MSE of found model
        nmse, nmsedB = calc_nmse(seg, x_seg_zeromean_hat, interval)

        if interval != 0:
            x_seg_zeromean_hat = x_seg_zeromean_hat[interval:-interval]
            pos = remove_interval(pos, interval)
        #
        # vary model order if nmse is larger than threshold
        if (nmsedB > nmsedBLim and K_pulses < Kmax):
            K_pulses += 1
            previous = [x_seg_zeromean_hat, K_pulses - 1, pos, sigma, amp, dk, nmse, nmsedB]

        elif K_pulses == Kmax and nmsedB > nmsedBLim:
            if previous != []:
                nmse_dbprev = previous[7]

            else:
                pos_p, sigma_p, amp_p, dk_p = annihilatingCauchyLorentz(seg, K_pulses - 1, R=R)
                x_seg_zeromean_hat_p = reconstructCauchyLorentzPeriodic(t, pos_p, amp_p, sigma_p, dk_p)
                # compute normalized MSE of found model
                nmse_p, nmse_dbprev = calc_nmse(seg, x_seg_zeromean_hat_p, interval)
                if interval != 0:
                    x_seg_zeromean_hat_p = x_seg_zeromean_hat_p[interval:-interval]
                    pos_p = remove_interval(pos_p, interval)

                previous = [x_seg_zeromean_hat_p, K_pulses - 1, pos_p, sigma_p, amp_p, dk_p, nmse_p, nmse_dbprev]
            if nmse_dbprev < nmsedB:
                saving.append(previous[0])
                pos, sigma, ak, dk = sort_cl_pulses(previous[2], previous[3], previous[4], previous[5])

                return previous[0], previous[1], pos, sigma, ak, dk, previous[6], previous[7], saving
            else:
                cont = False

        else:
            cont = False
    pos, sigma, amp, dk = sort_cl_pulses(pos, sigma, amp, dk)
    return x_seg_zeromean_hat, K_pulses, pos, sigma, amp, dk, nmse, nmsedB, saving




# ------------------------------------------------------------- #
def cl_estimation_block(Nseg, aorta_seg, Kinit=1, Kmax=80, nmsedBLim=-40, R=float(1 / 200)):
    """
    FOr the entire block of Segments:
    Parametrization for minimally Kinit and maximally Kmax CL-pulses, goal reached for NMSE < nmsedBLim
    :return: aorta_seg_model:Parametric signals, K_pulses : Number of used pulses, nmsedB: Reached NMSE in DB, paras:
    List of parameter vectors, pos_cl: Pulse positions, sigma_cl: STDs, amp_cl: Amplitudes, dk_cl: 2. Amplitudes
    """
    Lseg = np.zeros(Nseg, dtype=int)
    K_pulses = np.zeros(Nseg, dtype=int)
    nmse = np.zeros(Nseg)
    nmsedB = np.zeros(Nseg)
    aorta_seg_mean = np.zeros(Nseg)

    pos_cl = [None] * Nseg  # tk
    amp_cl = [None] * Nseg  # nach paper ck
    sigma_cl = [None] * Nseg  # nach paper ak
    dk_cl = [None] * Nseg  # dk
    aorta_seg_model = [None] * Nseg
    paras = [None] * Nseg

    for run_seg in range(Nseg):

        # get current segment of aorta signal
        Lseg[run_seg] = aorta_seg[run_seg].shape[0]
        x_seg = aorta_seg[run_seg].flatten()
        aorta_seg_mean[run_seg] = np.mean(x_seg)
        x_seg_zeromean = x_seg - aorta_seg_mean[run_seg]

        x_seg_zeromean_hat, K, pos, sigma, amp, dk, nmse_val, nmsedB_val, saving = cl_estimation(x_seg_zeromean,
                                                                                                      Kinit, Kmax,
                                                                                                      nmsedBLim,
                                                                                                      interval=0, R=R)

        aorta_seg_model[run_seg] = x_seg_zeromean_hat + aorta_seg_mean[run_seg]
        update_values(run_seg, K_pulses, pos_cl, amp_cl, dk_cl, sigma_cl, nmse, nmsedB, K, pos, amp, dk, sigma,
                      nmse_val, nmsedB_val)
        paras[run_seg] = pack_cl_paras(Lseg[run_seg], aorta_seg_mean[run_seg], K, pos, amp, dk, sigma)
    return aorta_seg_model, K_pulses, nmsedB, paras, pos_cl, sigma_cl, amp_cl, dk_cl


# ------------------------------------------------------------------------------------------------------------------- #
# Help functions ---------------------------------------------------------------------------------------------------- #

# ------------------------------------------------- #
def calc_nmse(seg, seg_hat, interval):
    if interval != 0:
        nmse = np.mean((seg_hat[interval:-interval] - seg[interval:-interval]) ** 2) / np.var(seg[interval:-interval])
    else:
        nmse = np.mean((seg_hat - seg) ** 2) / np.var(seg)
    nmsedB = 10 * np.log10(nmse)
    return nmse, nmsedB


# ------------------------------------------------- #
def update_values(i, Kvec, pvec, avec, dvec, sigmavec, nmsevec, nmsedBvec, K, p, a, d, sigma, nm, nmdB):
    Kvec[i] = K
    pvec[i] = p
    avec[i] = a
    dvec[i] = d
    sigmavec[i] = sigma
    nmsevec[i] = nm
    nmsedBvec[i] = nmdB


# ------------------------------------------------- #
def pack_cl_paras(L, sig_mean_val, K, pos, amp, dk, sigma):
    z = np.zeros(2 + 4 * K)
    z[0] = L
    z[1] = sig_mean_val
    offset = 2
    for q in range(K):
        z[offset + q * 4] = pos[q]
        z[offset + q * 4 + 1] = amp[q]
        z[offset + q * 4 + 2] = dk[q]
        z[offset + q * 4 + 3] = sigma[q]
    return z
