# Author Patricia Fuchs
import numpy
from numpy.linalg import lstsq
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------------------------------------------- #
# Linear Regression Function ---------------------------------------------------------------------------------------- #
ramp = lambda u: np.maximum(u, 0)
step = lambda u: ((u > 0).astype(numpy.float64) * -1)


def SegmentedLinearReg(X, Y, breakpoints, nIterationsMax=25):
    # based on Muggeo, V. M. (2003). Estimating regression models with unknown breakpoints. Statistics in medicine,
    # 22(19), 3055-3071.
    # and https://datascience.stackexchange.com/questions/8457/python-library-for-segmented-regression-a-k-a-piecewise-regression
    # nIterationMax = 10
    breakpoints = np.sort(np.array(breakpoints))
    xmax = np.max(X)
    dt = np.min(np.diff(X))
    ones = np.ones_like(X)
    i = 0
    while i < nIterationsMax:
        i += 1
        isValid, index = check_valid_number(breakpoints)
        if not isValid:
            for k in index:
                breakpoints[k] = k * (xmax / len(breakpoints)) - 1
        breakpoints = np.sort(np.array(breakpoints))
        if breakpoints[-1] > xmax:
            breakpoints[-1] = xmax - 1
        if breakpoints[0] < 0:
            breakpoints[0] = 1
        # Linear regression:  solve A*p = Y
        Rk = [ramp(X - xk) for xk in breakpoints]
        Sk = [step(X - xk) for xk in breakpoints]

        A = np.array([ones, X] + Rk + Sk)
        try:
            p = lstsq(A.transpose(), Y, rcond=None)[0]

            # Parameters identification:
            a, b = p[0:2]
            ck = p[2:2 + len(breakpoints)]
            dk = p[2 + len(breakpoints):]

            # Estimation of the next break-points:
            newBreakpoints = breakpoints + dk / ck

            # Stop condition
            if np.max(np.abs(newBreakpoints - breakpoints)) < dt / 5:
                break

            breakpoints = newBreakpoints
            if i == nIterationsMax:
                isValid, index = check_valid_number(breakpoints)
                if not isValid:
                    for k in index:
                        breakpoints[k] = k * (xmax / len(breakpoints)) - 1
                    i -= 1

        except Exception as e:
            print(e)
            isValid, index = check_valid_number(breakpoints)
            if not isValid:
                for k in index:
                    breakpoints[k] = k * (xmax / len(breakpoints)) - 1

    else:
        v = 'maximum iteration reached'

    # Compute the final segmented fit:
    Xsolution = np.insert(np.append(breakpoints, max(X)), 0, min(X))
    ones = np.ones_like(Xsolution)
    Rk = [c * ramp(Xsolution - x0) for x0, c in zip(breakpoints, ck)]

    Ysolution = a * ones + b * Xsolution + np.sum(Rk, axis=0)
    z = sorted(zip(Xsolution, Ysolution))
    Xsolution, Ysolution = zip(*z)
    return Xsolution, Ysolution


# ------------------------------------------------------------------------------------------------------------------- #
# Help functions ---------------------------------------------------------------------------------------------------- #
def create_sec_max_bp(a, pieceerror, bp, xiter_eqsampled, point):
    anewslice = np.argpartition(pieceerror, -2)[-2:]
    if anewslice[0] == a:
        anew = anewslice[1]
    else:
        anew = anewslice[0]
    create_bp([point], bp, xiter_eqsampled, anew)


# ------------------------------------------------------------- #
def check_valid_number(a):
    valid = True
    indices = []
    for i in range(len(a)):
        if np.isnan(a[i]):
            valid = False
            indices.append(i)
    return valid, indices




# ------------------------------------------------------------- #
def check_cycles(xp, bplist):
    currLen = len(xp) - 1  # kPieces = len -1
    bplist.append(currLen)
    count = 0
    for k in range(len(bplist) - 1):
        if bplist[k] == currLen:
            count += 1
    if count > 2:
        # Cycle detected
        return True
    else:
        return False


# ------------------------------------------------------------- #
def create_bp(bplist, bp, xiter, point):
    if point > 0:
        lowerlimit = bp[point - 1]
    else:
        lowerlimit = 0
    if point < len(bp):
        upperlimit = bp[point]
    else:
        upperlimit = xiter[point + 1]
    for val in bplist:
        bp.append((upperlimit - lowerlimit) * val + lowerlimit)





# ------------------------------------------------------------- #
def redirect_xpoints(xpoints, ypoints, xmax):
    try:
        j = 0
        while round(xpoints[j]) < 0:
            j += 1
        newy = [ypoints[j]]
        newx = [round(xpoints[j])]
        for i in range(j + 1, len(xpoints)):
            if (round(xpoints[i]) != newx[-1] and round(xpoints[i]) <= xmax) and round(xpoints[i]) >= 0:
                newx.append(round(xpoints[i]))
                newy.append(ypoints[i])
        if newx[-1] != xmax:
            newx[-1] = xmax
            if newx[-1] == newx[-2]:
                newx = newx[:-1]
                newy = newy[:-1]
    except:
        print(xpoints)
        print(ypoints)
        raise ValueError

    return newx, newy


# ------------------------------------------------------------- #
def recon_interpolate(original, xpoints, ypoints):
    recon = np.zeros(int(xpoints[-1] + 1))
    Npieces = len(xpoints)
    Piecelen = np.zeros(Npieces - 1)
    PieceError = np.zeros(Npieces - 1)
    for k in range(Npieces - 1):
        recon[int(xpoints[k])] = ypoints[k]
        x = [int(xpoints[k]), int(xpoints[k + 1])]
        y = ypoints[k:k + 2]
        x_interp = np.arange(x[0] + 1, x[1])
        if len(x_interp) != 0:
            y_interp = np.interp(x_interp, x, y)
            recon[x_interp] = y_interp
        Piecelen[k] = x[1] - x[0]
        w = np.sum((original[x[0]:x[1]] - recon[x[0]: x[1]]) ** 2)
        PieceError[k] = (original[x[1]] - y[1]) ** 2 + w

    recon[x[1]] = y[1]
    return recon, Piecelen, PieceError

