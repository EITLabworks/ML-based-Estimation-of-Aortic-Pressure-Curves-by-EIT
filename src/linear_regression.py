# Author Patricia Fuchs
from src.linear_approx import *

# ------------------------------------------------------------------------------------------------------------------- #
# Linear Regression Function ---------------------------------------------------------------------------------------- #
ramp = lambda u: np.maximum(u, 0)
step = lambda u: (u > 0).astype(float)

# ------------------------------------------------------------- #
def linearParametrization(seg, kInit, nMax=80, nmsedBLim=-40, bSave=False):
    # Define parameters
    l = len(seg)
    nMaxLines = nMax
    nmsedBLimit = nmsedBLim
    x = np.linspace(0, l - 1, l)

    extra_bp = False
    sec_extra_bp = False
    cycleDetected = False
    cycleCount = 0

    # Initialize breakpoints for linear pieces
    bp = [l // 8, l // 4, int(l * 3 / 8), l // 2, int(l * 5 / 8), int(3 / 4 * l),
          int(l * 7 / 8)]
    bp = [l // 8, int(l * 3 / 8), l // 2, int(l * 5 / 8),
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
        nmse = np.mean((signal - seg) ** 2) / np.var(seg)
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
def linear_parametrization_block(Nseg, aorta_seg, kInit, Nmax=80, nmsedBLim=-40):
    nmse_vec = np.zeros(Nseg)
    nLines = np.zeros(Nseg)
    sig_segments = []
    points = []
    xpoints_total = []
    ypoints_total = []
    for i in range(Nseg):
        y_1 = aorta_seg[i].flatten()

        y_1_mean = np.mean(y_1)
        y_1_zeromean = y_1 - y_1_mean
        sig, Num, nmse, xpoints, ypoints = linearParametrization(y_1_zeromean, kInit, Nmax, nmsedBLim)
        ypoints += y_1_mean
        xpoints_total.append(xpoints)
        ypoints_total.append(ypoints)

        z = []
        for k in range(len(xpoints)):
            z.append(xpoints[k])
            z.append(ypoints[k])

        sig_mean = sig + y_1_mean
        sig_segments.append(sig_mean)
        nLines[i] = Num
        nmse_vec[i] = nmse
        points.append(z)
    return sig_segments, nLines, nmse_vec, points, xpoints_total, ypoints_total

