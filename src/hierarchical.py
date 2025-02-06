import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from src.linear_approx import *

# ------------------------------------------------------------------------------------------------------------------- #
# Fit functions ----------------------------------------------------------------------------------------------------- #
ramp = lambda u: np.maximum(u, 0)
step = lambda u: (u > 0).astype(float)


# ------------------------------------------------------------- #
def linearParametrization(seg, seg_ori, curve, kInit, nMax=80, nmsedBLim=-40, bSave=False):
    # Define parameters
    l = len(seg)
    nMaxLines = nMax
    nmsedBLimit = nmsedBLim
    x = np.linspace(0, l - 1, l)

    extra_bp = False
    sec_extra_bp = False
    cycleCount = 0

    # Initialize breakpoints for linear pieces
    bp = [l // 8, l // 4, int(l * 3 / 8), l // 2, int(l * 5 / 8), int(3 / 4 * l),
          int(l * 7 / 8)]

    kPieces = len(bp) + 1
    numBps = [kPieces - 1]

    if kPieces > nMaxLines:
        print(nMaxLines)
        print(kPieces)
        raise ValueError("Linear Parametrization: Kpieces bigger than the maximal amount of lines")

    # Iterate per segment to find optimal linear approximation
    cont = True
    while cont:
        if len(bp) + 1 > nMaxLines:
            print("hello")
        # Linear Regression for optimal breakpoint placement
        xiter, yiter = SegmentedLinearReg(x, seg, bp, nIterationsMax=15)

        # Check for invalid(doubled) breakpoints and redirect the x_points to integer values
        xiter_eqsampled, yiter = redirect_xpoints(xiter, yiter, x[-1])
        kPieces = len(xiter_eqsampled) - 1

        # Option 2
        signal, piecelen, pieceerror = recon_interpolate(seg, xiter_eqsampled, yiter)
        nmse = np.mean(((signal + curve) - seg_ori) ** 2) / np.var(seg_ori)
        nmsedB = 10 * np.log10(nmse)

        # Check for convergence
        if nmsedB > nmsedBLimit and kPieces < nMaxLines or kInit > kPieces:

            # Check for cycles /loops in the iteration
            cycleDetected = check_cycles(xiter_eqsampled, numBps)
            if len(xiter_eqsampled) == len(bp) and len(xiter_eqsampled) < nMaxLines - 2:
                extra_bp = True
            elif len(xiter_eqsampled) < len(bp) and len(xiter_eqsampled) < nMaxLines - 3:
                sec_extra_bp = True
            # Determine the new breakpoints
            bp = xiter_eqsampled[1:-1]
            a = np.argmax(pieceerror)
            if extra_bp:
                create_bp([1 / 3, 2 / 3, 1 / 2], bp, xiter_eqsampled, a)
                extra_bp = False
            elif kPieces < nMaxLines - 1:
                create_bp([1 / 3, 2 / 3], bp, xiter_eqsampled, a)
            else:
                create_bp([1 / 2], bp, xiter_eqsampled, a)

            if sec_extra_bp:
                create_sec_max_bp(a, pieceerror, bp, xiter_eqsampled, 1 / 2)
                sec_extra_bp = False

            # Precaution measurement if cycles are detected
            if cycleDetected:
           #     print("CycleDetected")
                cycleCount += 1
                if len(bp) + 4 < nMaxLines:
                    bp.append(int(0.2 * l))
                    bp.append(int(0.8 * l))
                    bp.append(int(0.6 * l))
                else:
                    if len(bp) + 1 == nMaxLines:
                        bp = bp[:-1]
                    create_sec_max_bp(a, pieceerror, bp, xiter_eqsampled, 2 / 3)
                cycleDetected = False
                if cycleCount > 10:
                    print("Left for cycle")
                    cont = False
        else:
            cont = False
    return signal, kPieces, nmsedB, xiter_eqsampled, yiter


# ------------------------------------------------------------------------------------------------------------------- #
# Hierarchical estimation --------------------------------------------------------------------------- #
class HierarchicalApprox:
    def __init__(self, typeOnePulse="Poly", afterError="Lin", interval=0):
        self.onePulse = typeOnePulse
        self.afterError = afterError
        self.interval = interval
        if self.onePulse == "Poly":
            self.onePulseFct = self.one_pulse_poly
            self.meanFct = self.mean_removal
            self.iPulseParas = 5

        if self.afterError == "Lin":
            self.AfterFct = linearParametrization

    # ------------------------------------------------------------- #
    def hier_block_linear(self, Nseg, aorta_seg, Kinit=1, Kmax=80, nmsedBLim=-35):
        Lseg = np.zeros(Nseg, dtype=int)
        K_pulses = np.zeros(Nseg, dtype=int)
        nmsedB = np.zeros(Nseg)
        aorta_seg_mean = np.zeros(Nseg)

        pos_cl = [None] * Nseg
        amp_cl = [None] * Nseg
        aorta_seg_model = [None] * Nseg
        paras = [None] * Nseg
        curveParas = [None] * Nseg

        for run_seg in range(Nseg):
            # get current segment of aorta signal
            Lseg[run_seg] = aorta_seg[run_seg].shape[0]
            x_seg = aorta_seg[run_seg].flatten()
            x_seg_zeromean, m = self.meanFct(x_seg)
            aorta_seg_mean[run_seg] = m

            pa, recon = self.onePulseFct(x_seg_zeromean, Lseg[run_seg])
            x_pulsefree = x_seg_zeromean - recon
            curveParas[run_seg] = pa
            rec, K, nmsedBval, pos, amp = self.AfterFct(x_pulsefree, x_seg_zeromean, recon, Kinit, Kmax, nmsedBLim)

            K_pulses[run_seg] = K
            self.update_values_lin(run_seg, K_pulses, pos_cl, amp_cl, nmsedB, K, pos, amp, nmsedBval)

            x_zeromean_hat = recon + rec
            aorta_seg_model[run_seg] = x_zeromean_hat + m
            p = self.pack_paras_lin_after(Lseg[run_seg], m, K, pa, pos, amp)
            paras[run_seg] = p

        return aorta_seg_model, K_pulses, nmsedB, paras, curveParas, pos_cl, amp_cl

    # ------------------------------------------------------------- #
    def mean_removal(self, seg):
        ymean = np.mean(seg)
        s = seg - ymean
        return s, ymean

    # ------------------------------------------------------------- #
    def one_pulse_poly(self, seg, Lseg, K=5):
        x = np.linspace(0, 1, Lseg)
        K = 4
        fit = np.polyfit(x, seg, K)
        fit_fn = np.poly1d(fit)
        return fit, fit_fn(x)

    # ------------------------------------------------------------- #
    def pack_paras_lin_after(self, L, sig_mean_val, K, curveParas, pos, amp, ):
        offset = 2 + self.iPulseParas
        z = np.zeros(offset + 2 * K)
        z[0] = L
        z[1] = sig_mean_val
        z[2:2 + self.iPulseParas] = curveParas
        for q in range(K):
            z[offset + q * 2] = pos[q]
            z[offset + q * 2 + 1] = amp[q]
        return z

    # ------------------------------------------------------------- #
    def update_values_lin(self, i, Kvec, posvec, ampvec, nmsedBvec, K, p, a, nmdB):
        Kvec[i] = K
        posvec[i] = p
        ampvec[i] = a
        nmsedBvec[i] = nmdB
