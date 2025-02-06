from scipy.linalg import toeplitz
import numpy as np


# ------------------------------------------------------------------------------------------------------------------- #
# Cauchy-Lorentz Spectral Estimation Functions ---------------------------------------------------------------------- #
def annihilatingCauchyLorentz(x_seg, K_pulses, R=1 / 200):
    """
    Determines Cauchy-Lorentz spectral estimation parameters with annihilating filter approach
    :param x_seg: Signal
    :param K_pulses: Number of cauchy-Lorentz pulses to use
    :param R: Paramter for handling instable poles
    """
    Lseg = x_seg.shape[0]
    # R = 1/200

    # time axis for current segment
    t = np.arange(Lseg)
    # determine FFT length without zero-padding
    X_seg = np.fft.fft(x_seg)

    # Offset to stay in positive frequency range for w
    offset = K_pulses - 1
    Xr = np.flipud(X_seg[-K_pulses + 1 + offset: offset + 1])
    Xc = X_seg[offset: offset + K_pulses + 10]
    Ftoeplitz = toeplitz(Xc, Xr)
    X_vec = X_seg[offset + 1: offset + K_pulses + 1].flatten()

    # Solve linear equation system
    try:
        af_symCL = (-np.linalg.pinv(Ftoeplitz)) @ X_vec
    except:
        afg = Ftoeplitz[0:len(X_vec), 0:len(X_vec)]
        af_symCL = (- np.linalg.pinv(afg) @ X_vec).flatten()

    # determine roots of annihilating filter in spectral domain to obtain pulse positions
    # precede a_0=1 to vector with coefficients af
    af_cl_roots = np.roots(np.block([1, af_symCL]))

    # determine position of pulses from phases of roots
    angles_cl = np.angle(af_cl_roots, deg=False)
    absolutes_cl = np.abs(af_cl_roots)

    pos_cl = np.mod(- angles_cl / (2 * np.pi), 1) * t[-1]  # == tk
    sigma_cl = -t[-1] * np.log(absolutes_cl) / (2 * np.pi)  # sigma cl or ak in paper has to be >0 else instable

    # check for stability sigma_cl>0
    if R == "abs":
        for q in range(len(sigma_cl)):
            if sigma_cl[q] < 0:
                sigma_cl[q] = np.abs(sigma_cl[q])
            if sigma_cl[q] < 1:
                sigma_cl[q] = 1

    elif type(R) == float:
        for q in range(len(sigma_cl)):
            if sigma_cl[q] < R:
                sigma_cl[q] = R
    # else:
    # print("Instable pole disregard")

    # determine amplitudes of Cauchy Lorentz pulses
    V = np.vander(af_cl_roots, K_pulses + 1, increasing=True).T
    offset = 0
    try:
        amp_cl = np.linalg.pinv(V) @ X_seg[offset:offset + K_pulses + 1]
    except:
        blub = X_seg[offset:offset + K_pulses + 1]
        Vs = V[0:len(blub), 0:len(blub)]
        amp_cl = np.linalg.pinv(Vs) @ blub

    amp_real = np.real(amp_cl).flatten()  # ck
    amp_dk = -1 * np.imag(amp_cl).flatten()  # dk
    return pos_cl, sigma_cl, amp_real, amp_dk


# ------------------------------------------------- #
def reconstructCauchyLorentzPeriodic(t, pos, amp, sigma, dk):
    """
    Reconstruct Cauchy-Lorentz estimated signal by regarding the previously assumed periodicity
    :param t: time vector including repetitions (start of center block at time t=0)
    :param pos: pulse positions
    :param amp: pulse amplitudes
    :param sigma: width of Cauchy-Lorentz pulses
    :param dk: pulse amplitudes
    :return: Reconstructed signal
    """
    K_pulses = pos.shape[0]
    Lseg = t.shape[0]
    Tseg = t[-1]
    aorta_seq_est = np.zeros(Lseg)

    for k in range(K_pulses):
        zk = np.exp(2 * np.pi / Tseg * (-sigma[k] + 1j * (t - pos[k])))
        aorta_seq_est += (amp[k] * (1 - (np.abs(zk)) ** 2) + 2 * dk[k] * np.imag(zk)) / (
                    (1 - zk) * (1 - np.conj(zk))).astype(float)
    aorta_seq_est *= 1 / Tseg
    return aorta_seq_est[0: Lseg]


# ------------------------------------------------------------------------------------------------------------------- #
# Help functions ---------------------------------------------------------------------------------------------------- #
def sort_cl_pulses(pos, sigma, amp, dk):
    paired_data = list(zip(pos, sigma, amp, dk))
    sorted_data = sorted(paired_data, key=lambda x: x[0])
    pos_new = [element[0] for element in sorted_data]
    sigma_new = [element[1] for element in sorted_data]
    amp_new = [element[2] for element in sorted_data]
    dk_new = [element[3] for element in sorted_data]
    return pos_new, sigma_new, amp_new, dk_new


# ------------------------------------------------- #
def remove_interval(pos, interval):
    if interval != 0:
        for j in range(len(pos)):
            pos[j] = pos[j] - interval
    return pos
