import numpy as np
from nn.aorta import AortaNormalizer


# ------------------------------------------------------------------------------------------------------------------- #
# Overall reconstruction -------------------------------------------------------------------------------------------- #
def reconstruct_paras(paras, paratype, segLen=1024):
    if paratype == "CauchyLorentz":
        return reconstruct_cl(paras)
    elif paratype == "Linear":
        return reconstruct_lin(paras)
    elif paratype == "PolyHierarchicalLin":
        return reconstruct_hierarchical_lin(paras, "Poly")
    elif paratype == "PolyHierarchicalLinCurve":
        return reconstruct_hierarchical_curve(paras, "Poly")
    elif paratype.endswith("LinRest"):
        return reconstruct_hierarchical_restlin(paras, "Poly", segLen)



# ------------------------------------------------------------- #
def recon_paras_block(parablock, paratype, bUncorrected=False, segLen=1024, bScale=None,Denorm="none", normcoeffs=[0,0], bReorderLinParas=False):
    recon_block = []
    if Denorm!= "none":
        if Denorm=="fixed":
            AN = AortaNormalizer(paratype=paratype, mode=Denorm)
            parablock = AN.normalize_inverse(np.array(parablock))
    for u in range(len(parablock)):
        re = reconstruct_paras(parablock[u], paratype, segLen=segLen)
        recon_block.append(re)
    return recon_block


# ------------------------------------------------------------------------------------------------------------------- #
# Specific reconstruction ------------------------------------------------------------------------------------------- #
def reconstruct_cl(p):
    p = list(p)
    offset = 2
    p = _check_for_zero_pulses_cl(p, offset, limit=0.9)
    Lsegment = np.abs(int(p[0]))
    Lmean = float(p[1])
    #Lsegment=1024

    Tseg = Lsegment - 1
    t = np.linspace(0, Tseg, Lsegment)
    K_pulses = (len(p) - offset) // 4
    aorta_seq_est = np.zeros(Lsegment)
    for k in range(K_pulses):
        v = np.array(p[offset + k * 4:offset + (k + 1) * 4])
        v[3] = np.abs(v[3])
        if v[3] == 0:
            continue
        zk = np.exp(2 * np.pi / Tseg * (-v[3] + 1j * (t - v[0])))
        aorta_seq_est += (v[1] * (1 - (np.abs(zk)) ** 2) + 2 * v[2] * np.imag(zk)) / (
                (1 - zk) * (1 - np.conj(zk))).astype(float)
    aorta_seq_est *= 1 / Tseg
    aorta_seq_est = np.array(aorta_seq_est)
    aorta_seq_est += Lmean
    aorta_seq_est[np.isnan(aorta_seq_est)] = 50
    aorta_seq_est[aorta_seq_est > 800] = Lmean
    aorta_seq_est[aorta_seq_est < 0] = 0
    return aorta_seq_est

# ------------------------------------------------------------- #
def reconstruct_lin(p):
    p = list(p)
    p = _check_for_appended_zeros(p)
    try:
        aorta_seg = np.zeros(int(p[-2] + 1))
        for i in range(0, len(p) - 2, 2):
            aorta_seg[int(p[i])] = p[i + 1]
            xpoints = [int(np.abs(p[i])), int(np.abs(p[i + 2]))]
            ypoints = [p[i + 1], p[i + 3]]
            # if xpoints[1] >= xpoints[0]:

            x_interp = np.arange(xpoints[0] + 1, xpoints[1])
            if len(x_interp) != 0:
                y_interp = np.interp(x_interp, xpoints, ypoints)
                aorta_seg[x_interp] = y_interp
        aorta_seg[int(xpoints[1])] = ypoints[1]
    except:
        for i in range(len(p), 2):
            p[i] = int(p[i])
    return aorta_seg



# ------------------------------------------------------------- #
def reconstruct_hierarchical_curve(p, curvetype):
    Lsegment = int(p[0])
    Lmean = p[1]
    Tseg = Lsegment - 1
    t = np.linspace(0, 1, Lsegment)
    if curvetype == "Poly":
        curve_paras = 5
        fit = p[2:2 + curve_paras]
        t = np.linspace(0, 1, Lsegment)
        curve= np.polyval(fit, t)
    else:
        raise ValueError("The curve type is not defined for the reconstruction of hierarchical parameters.")
    aorta_seq_est = np.zeros(Lsegment)
    aorta_seq_est += Lmean
    aorta_seq_est += curve
    aorta_seq_est[np.isnan(aorta_seq_est)] = 50
    aorta_seq_est[aorta_seq_est > 800] = Lmean
  #  aorta_seq_est[aorta_seq_est <200] = 200
    return aorta_seq_est



# ------------------------------------------------------------- #
def reconstruct_hierarchical_restlin(p, curvetype, Lsegment):
    if curvetype == "Poly":
        curve_paras = 5
    else:
        curve_paras = 5

    p_lin = list(p[curve_paras+2:])
    p_lin = _check_for_appended_zeros(p_lin)
    try:
        aorta_seg = np.zeros(Lsegment)
        for i in range(0, len(p_lin) - 2, 2):
            aorta_seg[int(p_lin[i])] = p_lin[i + 1]
            xpoints = [int(np.abs(p_lin[i])), int(np.abs(p_lin[i + 2]))]
            ypoints = [p_lin[i + 1], p_lin[i + 3]]
            # if xpoints[1] >= xpoints[0]:

            x_interp = np.arange(xpoints[0] + 1, xpoints[1])
            if len(x_interp) != 0:
                y_interp = np.interp(x_interp, xpoints, ypoints)
                aorta_seg[x_interp] = y_interp
        aorta_seg[int(xpoints[1])] = ypoints[1]
    except:
        for i in range(len(p), 2):
            p[i] = int(p[i])
    aorta_seg[np.isnan(aorta_seg)] = 50
    aorta_seg[aorta_seg > 800] = 50
    #   aorta_seg[aorta_seg <200] = 200
    return aorta_seg


# ------------------------------------------------------------- #
def reconstruct_hierarchical_lin(p, curvetype, errortype="CL"):
    Lsegment = int(p[0])
    Lmean = p[1]
    Tseg = Lsegment - 1
    t = np.linspace(0, Tseg, Lsegment)
    if curvetype == "Poly":
        curve_paras = 5
        fit = p[2:2 + curve_paras]
        t = np.linspace(0, 1, Lsegment)
        curve= np.polyval(fit, t)

    else:
        raise ValueError("The curve type is not defined for the reconstruction of hierarchical parameters.")
    offset = curve_paras + 2
    p_lin = list(p[offset:])
    p_lin = _check_for_appended_zeros(p_lin)
    try:
        aorta_seg = np.zeros(Lsegment)
        for i in range(0, len(p_lin) - 2, 2):
            aorta_seg[int(p_lin[i])] = p_lin[i + 1]
            xpoints = [int(np.abs(p_lin[i])), int(np.abs(p_lin[i + 2]))]
            ypoints = [p_lin[i + 1], p_lin[i + 3]]
            # if xpoints[1] >= xpoints[0]:

            x_interp = np.arange(xpoints[0] + 1, xpoints[1])
            if len(x_interp) != 0:
                y_interp = np.interp(x_interp, xpoints, ypoints)
                aorta_seg[x_interp] = y_interp
        aorta_seg[int(xpoints[1])] = ypoints[1]
    except:
        for i in range(len(p), 2):
            p[i] = int(p[i])
    aorta_seg += Lmean
    aorta_seg += curve
    aorta_seg[np.isnan(aorta_seg)] = 50
    aorta_seg[aorta_seg > 800] = Lmean
    #   aorta_seg[aorta_seg <200] = 200
    return aorta_seg



# ------------------------------------------------------------------------------------------------------------------- #
# Help functions ---------------------------------------------------------------------------------------------------- #
def _check_for_zero_pulses_cl(p, offset=2, limit=0.9):
    q = len(p) - 4
    minlen = offset + 5 * 4
    while q >= minlen:
        # pulse position smaller than limit(0.9) -> only noise
        if p[q] < limit:
            p = p[:q]
            q -= 4
        else:
            q = 0  # once a valid pulse is found, no more shortening
    return p


# ------------------------------------------------------------- #
def _check_for_appended_zeros(p, lim=1):
    q = 2
    p[0] = 0
    while q < len(p):
        if p[q] < p[q - 2]:
            p = p[:q] + p[q + 2:]
        else:
            q += 2
    return p


def split_y(Y, curvelen=7, bLeaveLen=True):
    # Y=[numsamples, lenparas]
    if bLeaveLen:
        Ycurve = Y[:, 1:curvelen]
    else:
        Ycurve = Y[:, :curvelen]
    Yerror = Y[:, curvelen:]
    return Ycurve, Yerror


def append_y(ycurve, yresidual, bAddLength=True, iLen=1024):
    y = np.append(ycurve, yresidual, axis=1)
    if bAddLength:
        L = np.ones((len(ycurve), 1)) * iLen
        y = np.append(L, y, axis=1)
    return y
