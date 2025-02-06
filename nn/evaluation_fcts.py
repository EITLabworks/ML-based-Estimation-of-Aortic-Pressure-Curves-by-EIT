import numpy as np
import matplotlib.pyplot as plt
from src.reconstruction import recon_paras_block
from scipy.stats.stats import pearsonr
from sklearn.preprocessing import LabelEncoder
TITLE_SIZE = 24
FONT_SIZE = 20
LEGEND_SIZE = 16


# ------------------------------------------------------------------------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------- #
# Help functions
# ------------------------------------------------------------------------------------------------------------- #
def append_aorta(aorta_seg, aorta_estimated=None):
    """
    Appends aortas segments into a longer time piece, optionally appends a second list of segments appended for
    each segment to have the same length
    :param aorta_seg: list or aorta segments, each segment being a numpy array
    :param aorta_estimated: list or aorta segments, each segment being a numpy array, each segment is appendend, so that
    segments of both imput list have the same length
    :return: the single array of appended segmends or two arrays of appended segments
    """
    vec = np.array([])
    if aorta_estimated == None:
        for i in range(len(aorta_seg)):
            vec = np.append(vec, aorta_seg[i])
        return vec
    else:
        vec_estimated = np.array([])
        aorta_seg = aorta_seg.copy()
        aorta_estimated = aorta_estimated.copy()
        for i in range(len(aorta_seg)):
            # check length
            dist = len(aorta_seg[i]) - len(aorta_estimated[i])
            if dist > 0:
                aorta_estimated[i] = np.append(aorta_estimated[i], aorta_estimated[i][-1] * np.ones(dist))
            elif dist < 0:
                aorta_seg[i] = np.append(aorta_seg[i], aorta_seg[i][-1] * np.ones(np.abs(dist)))
            vec = np.append(vec, aorta_seg[i])
            vec_estimated = np.append(vec_estimated, aorta_estimated[i])
        return vec, vec_estimated


# ------------------------------------------------------------------------------------------------------------- #
def calc_nmse(seg, ori):
    """
    Calculates the NMSE (normalized mean squared error)
    :param seg: Estimated data
    :param ori: Original data
    :return: NMSE
    """
    return np.mean((seg - ori) ** 2) / np.var(ori)


# ------------------------------------------------------------------------------------------------------------- #
def calc_mse(seg, ori):
    """
    Calculates the MSE (mean squared error)
    :param seg: Estimated data
    :param ori: Original data
    :return: MSE
    """
    return np.mean((seg - ori) ** 2)


# ------------------------------------------------------------------------------------------------------------- #
def calc_mae(seg, ori):
    """
    Calculates the MAE (mean absolute error)
    :param seg: Estimated data
    :param ori: Original data
    :return: MAE
    """
    return np.mean(np.abs((seg - ori)))



# ------------------------------------------------------------------------------------------------------------- #
def make_int(v):
    """
    Makes every value of the input vector to an integer number
    """
    for q in range(len(v)):
        v[q] = int(v[q])
    return v



# ------------------------------------------------------------------------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------- #
# Plotter functions



# ------------------------------------------------------------------------------------------------------------- #
def plot_parameters(paras, parasori, title, Smax=100, bShow=True, bSave=False, fSavePath="C:/"):
    """
    Plots Smax  aorta parameters (representing the curve) overlaying over each other
    :param paras: Estimated aortic pressure parameters as list of segments [Num segments L x Para length]
    :param parasori: Original aortic parameters as list of segments [Num segments L x Para length]
    :param title: Plot title
    :param Smax: Max number of curves to plot
    :param bShow: if to show the plot
    :param bSave: if to save the plot
    :param fSavePath: Path to save the plot
    """
    Smax = Smax
    if len(paras) < Smax:
        Smax = len(paras)
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(17, 9))
    plt.subplots_adjust(left=0.079, top=0.936, right=0.957, bottom=0.08)
    for axi in ax.flat:
        for axis in ["top", "bottom", "left", "right"]:
            axi.spines[axis].set_linewidth(2)
    for q in range(Smax):
        ax[1].plot(paras[q], "x", color="blue", linewidth="0.8")
        ax[0].plot(parasori[q], "x", color="red", linewidth="0.8")
    ax[0].grid(linewidth=0.8)
    ax[1].grid(linewidth=0.8)
    ax[1].set_xlabel("Parameter index", loc="right", fontsize=FONT_SIZE)
    ax[1].set_title("Estimated parameters " + title, fontsize=TITLE_SIZE)
    ax[0].set_title("Original parameters " + title, fontsize=TITLE_SIZE)
    if bSave:
        fig.savefig(fSavePath + title + "ParaEstimation.png")
    if bShow:
        plt.show()




# ------------------------------------------------------------------------------------------------------------- #
def plot_history(history, type, bShow=True, bSave=False, fSavePath="C:/"):
    """
    Plot Training history of the neural network
    :param history: History strucutre
    :param type: Measurement type to plot
    :param bShow: if to show the plot
    :param bSave: if to save the plot
    :param fSavePath: Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(17, 9))
    plt.subplots_adjust(left=0.079, top=0.936, right=0.957, bottom=0.1)
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(2)
    ax.plot(history.history[type], "b", linewidth=2)
    ax.plot(history.history['val_' + type], "r", linewidth=2)
    ax.set_title('Model ' + type, fontsize=TITLE_SIZE)
    ax.set_ylabel(type, loc="top", fontsize=FONT_SIZE)
    ax.set_xlabel('Epoch', loc="right", fontsize=FONT_SIZE)
    ax.legend(['Training', 'Validation'], loc='upper left', fontsize=LEGEND_SIZE)
    ax.grid(linewidth=0.8)
    if bSave:
        fig.savefig(fSavePath + type + "TrainingHistory.png")
    if bShow:
        plt.show()


# ------------------------------------------------------------------------------------------------------------- #
def plot_4_recon_curves_paper(curves, curveoris, title, paratype, segment=[np.array([None])], recon_given=True,
                        ind=[5, 25, 40, 50], bShow=True, bSave=False, fSavePath="C:/"):
    """
    Plots 4 reconstructed aorta curves into on plot
    :param curves: Estimated aortic pressure curves as list of segments [Num segments L x individual segment len] or only paras
    :param curveoris: Original parametric aortic pressure curves [Num segments L x individual segment len] or only paras
    :param title: Plot title
    :param paratype: Type of paras for reconstruction
    :param segment: Original parametric aortic pressure curves [Num segments L x individual segment len]
    :param recon_given: If the reconstructed curves are already given /or if reconstruction should be performed
    :param ind: Indices from curves and curvesoris for which segments to use
    :param bShow: if to show the plot
    :param bSave: if to save the plot
    :param fSavePath: Path to save the plot
    """
    curves_used = []
    curves_used_ori = []
    for i in ind:
        curves_used.append(curves[i])
     #   curves_used_ori.append(curveoris[i])
    #   FONT_SIZE = 18
    # Für Graphik die beide columns überspannt 18, sonst 36
    FONT_SIZE = 18

    curve = curves_used
  #  curveori = curves_used_ori
    curveori = curveoris
    fontSize = 22
    title_Size = 24
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rc('xtick', labelsize=fontSize)
    plt.rc('ytick', labelsize=fontSize)
    # Figure
    fig, ax = plt.subplots(2, 2, figsize=(17, 9))

    plt.subplots_adjust(left=0.062, top=0.986, right=0.99, bottom=0.08, wspace=0.05, hspace=0.07)
    r = 0
    c = 0
    for axi in ax.flat:
        for axis in ["top", "bottom", "left", "right"]:
            axi.spines[axis].set_linewidth(2)
        axi.yaxis.set_tick_params(labelsize=FONT_SIZE)
        axi.xaxis.set_tick_params(labelsize=FONT_SIZE)
     #   axi.set_facecolor("whitesmoke")
    for q in range(len(ind)):
   #     ax[r,c].set_ylim([70,122])
        ax[r, c].plot(curve[q], color="maroon", linewidth=2, label="Estimated CAP Curve")
        ax[r, c].plot(curveori[q], color="steelblue", linewidth=2, label="CAP Curve")
        if segment[0].any() != None:
            ax[r, c].plot(segment[q], color="goldenrod", linewidth=2, label="Original Curve")
        ax[r, c].grid(linewidth=0.8)
        ax[r, c].legend(fontsize=FONT_SIZE)
    #    ax[r, c].set_title(title, fontsize=FONT_SIZE)
        c += 1
        if c == 2:
            c = 0
            r += 1
    ax[0,0].set_xticklabels([])
    ax[0,1].set_xticklabels([])
    ax[1,1].set_yticklabels([])
    ax[0,1].set_yticklabels([])

    ax[1, 1].set_xlabel("Samples", fontsize=FONT_SIZE, loc="right")
    ax[0, 0].set_ylabel("Pressure [mmHg]", fontsize=FONT_SIZE, loc="top")
    if bSave:
        fig.savefig(fSavePath + title + "4Plot.png")
        fig.savefig(fSavePath + title + "4Plot.svg")
      #  fig.savefig(fSavePath + title + "4Plot.pgf")
    #if bShow:
    plt.show()

#todo gucken ob so fertig, dann min mean max und relative error soweit vorbereiten



# ------------------------------------------------------------------------------------------------------------- #
def plot_appended_recon_curves(curves, curveoris, title, paratype, segment=np.array([None]), recon_given=True,
                               bShow=True, bSave=False, fSavePath="C:/", ):
    """
    Plots appended reconstructed aorta curves
    :param curves: Estimated aortic pressure curves as list of segments [Num segments L x individual segment len] or only paras
    :param curveoris: Original parametric aortic pressure curves [Num segments L x individual segment len] or only paras
    :param title: Plot title
    :param paratype: Type of paras for reconstruction
    :param segment: Original parametric aortic pressure curves [Num segments L x individual segment len]
    :param recon_given: If the reconstructed curves are already given /or if reconstruction should be performed
    :param bShow: if to show the plot
    :param bSave: if to save the plot
    :param fSavePath: Path to save the plot
    """
    FONT_SIZE=18
    if not recon_given:
        curves = recon_paras_block(curves, paratype)
        curveoris = recon_paras_block(curveoris, paratype)
    plt.rcParams["font.family"] = "Times New Roman"
    curve_original, curve_estimated = append_aorta(curveoris, curves)
    segment = append_aorta(segment)

    fig, ax = plt.subplots(figsize=(17, 9))
    plt.subplots_adjust(left=0.064, top=0.98, right=0.99, bottom=0.08)
    for axis in ["top", "bottom", "left", "right"]:
        ax.spines[axis].set_linewidth(2)
    ax.plot(curve_original, color="steelblue", linewidth=3, label="CAP Curves")
    ax.plot(curve_estimated, color="maroon", linewidth=3, label="Estimated CAP Curves")

    if segment.any() != None:
        ax.plot(segment, color="goldenrod", linewidth=3, label="Original Curves")
  #  ax.grid(linewidth=0.8)
    ax.grid()
    ax.legend(fontsize=LEGEND_SIZE)
    ax.legend(fontsize=FONT_SIZE, loc="lower right")
    ax.set_ylabel("Pressure [mmHg]", loc="top", size=FONT_SIZE)
    ax.set_xlabel("Samples", loc="right", size=FONT_SIZE)
    ax.yaxis.set_tick_params(labelsize=FONT_SIZE)
    ax.xaxis.set_tick_params(labelsize=FONT_SIZE)
    ax.set_xticklabels([])
  #  ax.set_title(title, size=22)
    if bSave:
        fig.savefig(fSavePath + title + " AppendRecon.png")
    if bShow:
        plt.show()





def Pearson_correlation(Y_true,Y_pred)->list:
    """
    The Pearson correlation coefficient [1]_ measures the linear relationship
    between two datasets. Like other correlation
    coefficients, this one varies between -1 and +1 with 0 implying no
    correlation. Correlations of -1 or +1 imply an exact linear relationship.
    Positive correlations imply that as x increases, so does y. Negative
    correlations imply that as x increases, y decreases.

    Return:
    list of Pearson correlation coefficients
    """
    corr_number = list()
    for true,pred in zip(Y_true,Y_pred):
        p_nr, _ = pearsonr(true, pred)
        corr_number.append(p_nr)
    corr_number = np.array(corr_number)
    print("Pearson correlation coefficient mean", np.mean(corr_number))
    return corr_number



def plot_relative_error_aorta(Y_true, Y_pred, std, var, mean, s_name=None, bShow=False, bSave=False, fSavePath="C:/"):
    mean_err = np.mean(np.abs((Y_pred - Y_true) / Y_true), axis=0) * 100
    print(mean_err.shape)
    std_err = np.std((Y_pred - Y_true) / Y_true, axis=0) * 100  # evtl mit abs
    std_err_abs = np.std(np.abs((Y_pred - Y_true) / Y_true), axis=0) * 100
    var_err = np.var((Y_pred - Y_true) / Y_true, axis=0) * 100  #evtl mit abs
    L = len(Y_true)
    MAE = np.sum(np.abs(Y_true - Y_pred)) / 1024 /L
    MSE = np.sum((Y_true - Y_pred) ** 2) / 1024 /L
    PN = np.mean(Pearson_correlation(Y_true, Y_pred))

    print("MSE", MSE)
    print("MAE", MAE)
    print("Pearson number", PN)
    plt.figure(figsize=(6, 2))
    if mean:
        plt.plot(mean_err, label="Relative mean absolute deviation")
    if std:
        plt.plot(std_err, label="Standard deviation")
        plt.plot(std_err_abs, label="Standard deviation of absolute error",color="red")
    if var:
        plt.plot(var_err, label="Variance")
    plt.legend()
    plt.xlabel("Aorta pressure curve index")
    plt.xticks(ticks=np.linspace(0, 1024, 5), labels=np.linspace(0, 1024, 5, dtype=int))
    plt.ylabel("Relative error (%)")
    plt.grid()
    plt.tight_layout()
    if bSave:
        plt.savefig(fSavePath + s_name + " RelativeError.png")
    if bShow:
        plt.show()
    return MAE, MSE, PN





def consecutive_error_blockwise(y, yhat, piginfo):
    # y has to be unshuffled
    le_p = LabelEncoder()
    le_b = LabelEncoder()
    le_p.fit(piginfo[:, 0])
    curvelen= len(y[0])
    counter= 0
    mae= 0
    mse= 0
    for p in le_p.classes_:
        idx_p = np.where(piginfo[:, 0] == p)
        le_b.fit(np.squeeze(piginfo[idx_p, 1]))
        for b in le_b.classes_:
            idx = np.array(np.where((piginfo[:, 0] == p) & (piginfo[:, 1] == b))[0])
            print(idx)
            error_init = np.mean(y[idx[0]]) - np.mean(yhat[idx[0]])
            print(error_init)
            y_partial = y[idx[1:]] +error_init
            E = (y[idx[1:]] -(yhat[idx[1:]] +error_init))
            counter += len(idx)-1
            mae += np.sum(np.abs(E))
            mse += np.sum(np.square(E))
    mae = mae / counter /curvelen
    mse = mse / curvelen/  counter
    print("MSE with blockwise adjusted mean", mse)
    print("MAE with blockwise adjusted mean ", mae)
