import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from parametrization.cauchy_lorentz import annihilatingCauchyLorentz, reconstructCauchyLorentzPeriodic, sort_cl_pulses, \
    remove_interval
from parametrization.cl_estimation import cl_estimation, add_periodic_interval, calc_nmse
from parametrization.gaussian_estimation import gaussian_estimation
from parametrization.linear_approx import *

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
          int(l * 7 / 8)]  # todo better initialization breakpoints
    #  bp = [l // 8,int(l * 3 / 8), l // 2, int(l * 5 / 8),
    #      int(l * 7 / 8)]  # todo better initialization breakpoints

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
                print("CycleDetected")
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


# ------------------------------------------------------------- #
def cl_estimation_test(seg, seg_ori, curve, Kinit=1, Kmax=80, nmsedBLim=-40, interval=30, R=float(1 / 200)):
    K_pulses = Kinit
    # get current segment of aorta signal
    Lseg = len(seg)
    R = "abs"
    if interval != 0:
        seg = add_periodic_interval(seg, interval)
        Lseg = len(seg)

    # time axis for current segment
    t = np.arange(Lseg)
    previous = []
    cont = True
    while cont:
        pos, sigma, amp, dk = annihilatingCauchyLorentz(seg, K_pulses, R=R)
        # construct aorta signal from detected pulses considering periodic repetitions (output does not contain repetitions)
        #   x_seg_zeromean_hat = reconstructCauchyLorentz(t, pos, amp, sigma, dk)
        x_seg_zeromean_hat = reconstructCauchyLorentzPeriodic(t, pos, amp, sigma, dk)
        #  x_seg_zeromean_hat = reconstructCauchyLorentz_FD(t, pos, amp, sigma, dk)
        # compute normalized MSE of found model

        si_ = x_seg_zeromean_hat[interval:-interval] + curve
        # nmse, nmsedB = calc_nmse(seg, x_seg_zeromean_hat, interval)
        nmse, nmsedB = calc_nmse(seg_ori, si_, 0)

        if interval != 0:
            x_seg_zeromean_hat = x_seg_zeromean_hat[interval:-interval]
            pos = remove_interval(pos, interval)

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
                pos, sigma, ak, dk = sort_cl_pulses(previous[2], previous[3], previous[4], previous[5])
                return previous[0], previous[1], pos, sigma, ak, dk, previous[6], previous[7]
            else:
                cont = False
        else:
            cont = False

    pos, sigma, amp, dk = sort_cl_pulses(pos, sigma, amp, dk)
    return x_seg_zeromean_hat, K_pulses, pos, sigma, amp, dk, nmse, nmsedB


# ------------------------------------------------------------- #
def cl_time(t, ak, dk, sigma, tk):
    res = 1 / np.pi * (ak * sigma + dk * (t - tk)) / (sigma ** 2 + (t - tk) ** 2)
    return res


# ------------------------------------------------------------- #
def gaussian_fct(t, ak, sigma, tk):
    res = ak / (np.sqrt(2 * np.pi * sigma ** 2)) * np.exp(-(t - tk) ** 2 / (2 * sigma ** 2))
    return res


# ------------------------------------------------------------- #
def gaussian_curve(seg, K, nmseDblim):
    Lseg = len(seg)
    t = np.linspace(0, Lseg - 1, Lseg)
    tk = np.argmax(seg)
    sigma = 250
    ak = np.max(seg) * np.sqrt(2 * np.pi * sigma ** 2)
    popt, pcov = curve_fit(gaussian_fct, t, seg, [ak, sigma, tk])
    plt.scatter(t, seg, label='data', color='r')
    plt.plot(t, gaussian_fct(t,popt[0], popt[1], popt[2]), color='b', label='fit')
    plt.legend(loc='upper left')
    plt.title("Gaussian")
    plt.show()


# ------------------------------------------------------------- #
def approximate_with_CL(seg, K, nmsedBLim):
    Lseg = len(seg)
    t = np.linspace(0, Lseg - 1, Lseg)
    tk = np.argmax(seg)
    sigma = 250
    ak = np.max(seg) * np.pi * sigma - 20
    dk = ak / 2
    pulse = cl_time(t, ak, dk, sigma, tk)
    plt.scatter(t, seg, label='data', color='r')
    plt.plot(t, pulse, color='b', label='fit')
    plt.legend(loc='upper left')
    plt.title("CL")
    plt.show()


# ------------------------------------------------------------- #
def approximate_with_CL_new(seg, K, nmsedBLim):
    Lseg = len(seg)
    t = np.linspace(0, Lseg - 1, Lseg)
    tk = np.argmax(seg)
    sigma = 350
    ak = np.max(seg) * np.pi * sigma
    dk = ak / 4
    popt, pcov = curve_fit(cl_time, t, seg, [ak, dk, sigma, tk])

    #  pulse =cl_time(t, ak, dk, sigma, tk)
    plt.scatter(t, seg, label='data', color='r')
    plt.plot(t, cl_time(t, popt[0], popt[1], popt[2], popt[3]), color='b', label='fit')
    e = seg - cl_time(t, popt[0], popt[1], popt[2], popt[3])
    plt.plot(t, e, color="green", label="Error")
    plt.legend(loc='upper left')
    plt.title("CL")
    plt.show()


# ------------------------------------------------------------- #
ramp = lambda u: np.maximum(u, 0)
def weibull(t, lam, k, amp, tk):
    t_new = t - tk
    t_new = ramp(t_new)
    res = k / lam * (t_new / lam) ** (k - 1) * np.exp(-(t_new / lam) ** k)
    res = res * amp
    return res


# ------------------------------------------------------------- #
def weibull_fit(seg, K, nmseDbLim):
    Lseg = len(seg)
    t = np.linspace(0, Lseg - 1, Lseg)
    k = 5
    l = 200
    tk = np.argmax(seg)

    popt, pcov = curve_fit(weibull, t, seg, [l, k, np.max(seg) * l, tk])
    plt.scatter(t, seg, label='data', color='r')
    plt.plot(t, weibull(t, popt[0], popt[1], popt[2], popt[3]), color='b', label='fit')
    e = seg - weibull(t, popt[0], popt[1], popt[2], popt[3])
    plt.plot(t, e, color="green", label="Error")
    plt.legend(loc='upper left')
    plt.title("Weibull")
    plt.show()


# ------------------------------------------------------------- #
# seg is zeromean
def approximate_with_polynomial(seg, K, nmsedBLim):
    l = len(seg)
    x = np.linspace(0, l - 1, l)
    fit = np.polyfit(x, seg, K)
    # print(fit)
    fit_fn = np.poly1d(fit)
    #   print(fit_fn)
    plt.scatter(x, seg, label='data', color='r')
    plt.plot(x, fit_fn(x), color='b', label='fit')
    e = seg - fit_fn(x)
    plt.plot(x, e, color="green", label="Error")
    plt.legend(loc='upper left')
    plt.title("Polynomial")
    plt.show()


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

        pos_cl = [None] * Nseg  # tk
        amp_cl = [None] * Nseg  # nach paper ck
        aorta_seg_model = [None] * Nseg
        paras = [None] * Nseg
        curveParas = [None] * Nseg
        prob_segments = 0

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

            #  rec, K, nmsedBval, xitereq, yiter = self.AfterFct(x_pulsefree, Kinit, Kmax, nmsedBLim)
            K_pulses[run_seg] = K
            #
            self.update_values_lin(run_seg, K_pulses, pos_cl, amp_cl, nmsedB, K, pos, amp, nmsedBval)

            x_zeromean_hat = recon + rec
            aorta_seg_model[run_seg] = x_zeromean_hat + m
            p = self.pack_paras_lin_after(Lseg[run_seg], m, K, pa, pos, amp)
            paras[run_seg] = p
            if nmsedB[run_seg] > nmsedBLim:
                prob_segments += 1

        if prob_segments > 0:
            print("Hierarchical-Estimation problem segment amount was " + str(prob_segments))

        return aorta_seg_model, K_pulses, nmsedB, paras, curveParas, pos_cl, amp_cl
        # plot_outliers(model, aorta_seg, nmsedB, nmsedBLim)

    # ------------------------------------------------------------- #
    def mean_removal(self, seg):
        ymean = np.mean(seg)
        s = seg - ymean
        return s, ymean

    # ------------------------------------------------------------- #
    def minline_removal(self, seg):
        ymin = np.min(seg)
        s = seg - ymin
        return s, ymin

    # ------------------------------------------------------------- #
    def one_pulse_poly(self, seg, Lseg, K=5):
        x = np.linspace(0, 1, Lseg)
        K = 4
        fit = np.polyfit(x, seg, K)
        fit_fn = np.poly1d(fit)
        return fit, fit_fn(x)

    # ------------------------------------------------------------- #
    def one_pulse_lin(self, seg, Lseg, K=5, l=200):
        t = np.linspace(0, Lseg - 1, Lseg)
        tk = np.argmax(seg)
        yk = seg[tk]
        popt, pcov = curve_fit(lin_try, t, seg, [seg[0], tk, yk, seg[-1]])
        return popt[0:4], lin_try(t, popt[0], popt[1], popt[2], popt[3])

    # ------------------------------------------------------------- #
    def one_pulse_weibull(self, seg, Lseg, K=5, l=200):
        t = np.linspace(0, Lseg - 1, Lseg)
        tk = np.argmax(seg)
        try:
            popt, pcov = curve_fit(weibull, t, seg, [l, K, np.max(seg) * l, tk])
        except:
            popt = [l, K, np.max(seg) * l, tk]
            print("No optimal Curve fit found!")
        return popt[0:4], weibull(t, popt[0], popt[1], popt[2], popt[3])

    # ------------------------------------------------------------- #
    def one_pulse_gaussian(self, seg, Lseg, K=5, sigma=250):
        t = np.linspace(0, Lseg - 1, Lseg)
        tk = np.argmax(seg)
        ak = np.max(seg) * np.sqrt(2 * np.pi * sigma ** 2)
        try:
            popt, pcov = curve_fit(gaussian_fct, t, seg, [ak, sigma, tk])
        except:
            popt = [ak, sigma, tk]
            print("No optimal Curve fit found!")
        return popt[0:3], gaussian_fct(t, popt[0], popt[1], popt[2])

    # ------------------------------------------------------------- #
    def one_pulse_cl(self, seg, Lseg, K=5, sigma=350):
        t = np.linspace(0, Lseg - 1, Lseg)
        tk = np.argmax(seg)
        ak = np.max(seg) * np.pi * sigma
        dk = ak / 4
        try:
            popt, pcov = curve_fit(cl_time, t, seg, [ak, dk, sigma, tk])
        except:
            print("No optimal Curve fit found!")
            popt = [ak, dk, sigma, tk]
        return popt[:4], cl_time(t, popt[0], popt[1], popt[2], popt[3])

    # ------------------------------------------------------------- #
    def pack_paras(self, L, sig_mean_val, K, curveParas, pos, amp, dk, sigma):
        offset = 2 + self.iPulseParas
        z = np.zeros(offset + 4 * K)
        z[0] = L
        z[1] = sig_mean_val
        z[2:2 + self.iPulseParas] = curveParas
        for q in range(K):
            z[offset + q * 4] = pos[q]
            z[offset + q * 4 + 1] = amp[q]
            z[offset + q * 4 + 2] = dk[q]
            z[offset + q * 4 + 3] = sigma[q]
        return z

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
    def update_values(self, i, Kvec, pvec, avec, dvec, sigmavec, nmsevec, nmsedBvec, K, p, a, d, sigma, nm, nmdB):
        Kvec[i] = K
        pvec[i] = p
        avec[i] = a
        dvec[i] = d
        sigmavec[i] = sigma
        nmsevec[i] = nm
        nmsedBvec[i] = nmdB

    # ------------------------------------------------------------- #
    def update_values_lin(self, i, Kvec, posvec, ampvec, nmsedBvec, K, p, a, nmdB):
        Kvec[i] = K
        posvec[i] = p
        ampvec[i] = a
        nmsedBvec[i] = nmdB


# ------------------------------------------------------------------------------------------------------------------- #
# Help functions ---------------------------------------------------------------------------------------------------- #

# ------------------------------------------------------------- #
def lin_try(t, y0, x1, y1, y2):
    yout = np.zeros(len(t))
    yout[0] = y0
    #  print(x1)
    yout[int(x1)] = y1
    yout[len(t) - 1] = y2
    x = [0, x1]
    y = [y0, y1]
    x_interp = np.arange(1, int(x1))
    y_interp = np.interp(x_interp, x, y)
    # print(x_interp)
    # print(y_interp)
    yout[x_interp] = y_interp

    x = [x1, len(t) - 1]
    y = [y1, y2]

    x_interp = np.arange(int(x1 + 1), int(len(t) - 1))
    y_interp = np.interp(x_interp, x, y)
    yout[x_interp] = y_interp
    return yout

# note poylfit kann nicht gut die Klappenbewegung darstellen, auch nicht bei bei sehr hoher Polynomordnung
