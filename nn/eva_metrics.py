import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (mean_squared_error, mean_absolute_error, median_absolute_error, r2_score,
                             explained_variance_score, mean_tweedie_deviance, d2_absolute_error_score, d2_tweedie_score,
                             PredictionErrorDisplay)
from scipy.stats.stats import pearsonr
from scipy.stats import spearmanr
import json
import csv
import gc
import psutil
process=psutil.Process()
# -------------------------------------------------------------------------------------------------------------------- #
"""
Class to calculate certain metrics to evaluate the performance of a neural network
"""


class EvaMetrics:
    # -------------------------------------- #
    def __init__(self, fSavePath:str):
        """
        :param fSavePath: Path where the metrics should be shaved
        """
        self.fSavePath = fSavePath

        # Different categories
        self.byMeanForm = True
        self.bStandardMetrics = True

        # Metrics
        self.bMAE = True
        self.bMSE = True
        self.bMedianAE = True
        self.bMeanError = True
        self.bSTD = True  # and variance of the error
        self.bRScore = True  # Pearson coefficient tscore and p-value

        # Storage
        self.metrics = {}

    # -------------------------------------- #
    def gather_info(self):
        """
        Creates a collection of the most important metrics
        """
        d = {}
        for t in ["Test", "Vali"]:
            w = {}
            for p in ["Curve","Recon", "Paras"]:
                l = {}
                q = t + p
                if q in self.metrics:
                    l.update({"MAE": self.metrics[q]["Standard"]["MAE"]})
                    l.update({"MSE": self.metrics[q]["Standard"]["MSE"]})
                    l.update({"PearsonR": self.metrics[q]["Standard"]["PearsonR"]})
                    l.update({"PearsonP": self.metrics[q]["Standard"]["PearsonP"]})
                    if p == "Curve":
                        l.update({"MeanError": self.metrics[q]["Standard"]["MeanError"]})
                        l.update({"MAEmean": self.metrics[q]["ByMeanForm"]["Mean"]["MAE"]})
                        l.update({"Rmean": self.metrics[q]["ByMeanForm"]["Mean"]["PearsonR"]})
                        l.update({"MAEform": self.metrics[q]["ByMeanForm"]["Form"]["MAE"]})
                        l.update({"Rform": self.metrics[q]["ByMeanForm"]["Form"]["PearsonR"]})
                        l.update({"MAEform":round(self.metrics["TestCurve"]["ByMeanForm"]["Form"]["FormMAE"], 4)})
                    w.update({p: l})
            if t == "Test":
                d.update({"Testing": w})
            else:
                d.update({"Validation": w})
        self.metrics.update({"Compact": d})
        line = self.cvs_line()
        print(line)
        # paras = MAE, MSE, R
        # curve ;MSE, Mae, mean,var, std, R,p, Rs, R2, MAe Mean, MAe form


    def cvs_line(self):
        l = [self.fSavePath[-16:]]
        try:
            l.append(round(self.metrics["TestCurve"]["Standard"]["MAE"],4))
            l.append(round(self.metrics["TestCurve"]["Standard"]["PearsonR"],4))
            l.append(round(self.metrics["TestCurve"]["ByMeanForm"]["Mean"]["MAE"],4))
            l.append(round(self.metrics["TestCurve"]["ByMeanForm"]["Mean"]["PearsonR"],4))
            l.append(round(self.metrics["TestCurve"]["ByMeanForm"]["Form"]["MAE"],4))
            l.append(round(self.metrics["TestCurve"]["ByMeanForm"]["Form"]["PearsonR"],4))
            l.append(round(self.metrics["TestCurve"]["ByMinMax"]["Min"]["MAE"],4))
            l.append(round(self.metrics["TestCurve"]["ByMinMax"]["Min"]["PearsonR"],4))
            l.append(round(self.metrics["TestCurve"]["ByMinMax"]["Max"]["MAE"],4))
            l.append(round(self.metrics["TestCurve"]["ByMinMax"]["Max"]["PearsonR"],4))
            l.append(round(self.metrics["TestCurve"]["Standard"]["MSE"],4))
            l.append(round(self.metrics["ValiCurve"]["Standard"]["MAE"],4))
            l.append(round(self.metrics["TestCurve"]["ByMeanForm"]["Form"]["FormMAE"],4))
            l.append(round(self.metrics["TestCurve"]["ByMeanForm"]["Form"]["FormMSE"],4))
            l.append(round(self.metrics["ValiCurve"]["ByMeanForm"]["Form"]["FormMAE"],4))
            l.append(round(self.metrics["ValiCurve"]["ByMeanForm"]["Form"]["FormMSE"],4))
            l.append(round(self.metrics["TestRecon"]["Standard"]["MAE"],4))
            l.append(round(self.metrics["TestRecon"]["Standard"]["PearsonR"],4))
            l.append(round(self.metrics["TestRecon"]["ByMeanForm"]["Form"]["FormMAE"],4))
            l.append(round(self.metrics["TestCurve"]["ByMeanForm"]["Form"]["FormMAENorm"], 4))
        except:
            print("CSV Line could not be filled")
        return l



    # -------------------------------------- #
    def reset_metrics(self):
        """
        Resets the stored metrics
        """
        self.metrics = {}

    # -------------------------------------- #
    def save_metrics(self):
        """
        Saves all stored metrics to self.fSavePath/metrics.json
        """
        self.gather_info()
        with open(self.fSavePath + "metrics.json", "w") as outfile:
            json.dump(self.metrics, outfile)
        line= self.cvs_line()
        csvpath = self.fSavePath[:-16]
        with open(csvpath+"resultsNN.csv", 'a') as c_file:
            writer_o = csv.writer(c_file)
            writer_o.writerow(line)
            c_file.close()


    # -------------------------------------- #
    def calc_mae(self, y_true, y_predict):
        """
        Calculates the MAE (of 1D or 2D data)
        """
        if self.bMAE:
            mae = mean_absolute_error(y_true, y_predict)
            return mae

    # -------------------------------------- #
    def calc_mse(self, y_true, y_predict):
        """
        Calculates the MSE (of 1D or 2D data)
        """
        if self.bMSE:
            mse = mean_squared_error(y_true, y_predict)
            return mse

    # -------------------------------------- #
    def calc_median_ae(self, y_true, y_predict):
        """
        Calculates the median average error (of 1D or 2D data)
        """
        if self.bMedianAE:
            mae = median_absolute_error(y_true, y_predict)
            return mae

    # -------------------------------------- #
    def calc_mean_error(self, y_true, y_predict):
        """
        Calculates the mean error, the variance and standard deviation of the error (of 1D or 2D data)
        """
        if self.bMeanError:
            error = y_true - y_predict
            mean = np.mean(error, axis=None)
            var = np.var(error, axis=None)
            std = np.std(error, axis=None)

            return mean, var, std
 # -------------------------------------- #
    def calc_pearson(self, y_true, y_predict):
        """
        Calculates the Pearson coefficient (R-value) with t value and p value (of only 1D data)
        """
        if self.bRScore:
            r, p = pearsonr(y_true, y_predict)
            t = r / np.sqrt((1 - r ** 2) / (len(y_true) - 2))
            return r, t, p



    # -------------------------------------- #
    def calc_metrics(self, y_true, y_predict, paratype, name, bParas=False, bSave=True, bShow=False):
        """
        Calculates all the wanted metrics
        :param y_true: True values of data, 1D or 2D
        :param y_predict: Predicted values, 1D or 2D
        :param paratype: Type of parameters (Linear, CauchyLorentz, Hierarchical..)
        :param name: Name for saving and titles (Test/Vali)
        :param bParas: If the data are parameters or already resampled curves
        :param bSave: If the graphics should be saves
        :param bShow: If the graphics are shown
        """
        d = {}
        if self.bStandardMetrics:
            b = self.run_metrics(y_true, y_predict)
            d.update({"Standard": b})

        if self.byMeanForm and bParas == False:
            s = {}
            mean_true, mean_predict, form_true, form_predict = self.sort_mean_form(y_true, y_predict)
            s.update({"Mean": self.run_metrics(mean_true, mean_predict)})
            s.update({"Form": self.run_metrics(form_true, form_predict)})
            s["Form"].update({"FormMAE":self.form_error_block(form_true, form_predict, 1024, errortype="Absolute")})
            s["Form"].update({"FormMSE":self.form_error_block(form_true, form_predict, 1024, errortype="Squared")})
            s["Form"].update({"FormMAENorm":self.form_error_block(form_true, form_predict, 1024, errortype="Absolute", bnorm=True)})
            s["Form"].update({"FormMSENorm":self.form_error_block(form_true, form_predict, 1024, errortype="Squared", bnorm=True)})
            d.update({"ByMeanForm": s})

        self.metrics.update({name: d})



    # -------------------------------------- #
    def run_metrics(self, y_true, y_predict):
        """
        Call specific metric functions
        """
        di = dict()
        di.update({"MAE": self.calc_mae(y_true, y_predict)})
        di.update({"MSE": self.calc_mse(y_true, y_predict)})
        di.update({"MedianAE": self.calc_median_ae(y_true, y_predict)})
        m, v, s = self.calc_mean_error(y_true, y_predict)
        di.update({"MeanError": m})
        di.update({"VarError": v})
        di.update({"StdError": s})
        r, t, p = self.calc_pearson(y_true.flatten(), y_predict.flatten())
        di.update({"PearsonR": r})
        di.update({"PearsonT": t})
        di.update({"PearsonP": p})
        return di


    # -------------------------------------- #
    def sort_mean_form(self, y_true, y_predict):
        """
        Seperate each curve into a mean and a shape
        :return: True mean values, Predicted mean values, true shape vectors, estimated shape vectors
        """
        mean_true = np.mean(y_true, axis=1)
        mean_predict = np.mean(y_predict, axis=1)
        form_true = np.copy(y_true)
        form_predict = np.copy(y_predict)

        for k in range(len(y_true[0])):
            form_true[:, k] = form_true[:, k] - mean_true
            form_predict[:, k] = form_predict[:, k] - mean_predict
        return mean_true, mean_predict, form_true, form_predict


    # -------------------------------------- #
    def create_basic_curve(self, l=1024, height=40):
        x = np.linspace(0, 1, l)
        fit = [-560, 1374, -1158, 344, -17.48287563133016]
        fit_fn = np.poly1d(fit)
        return fit_fn(x)

    def calc_error_sig_abs(self, sigtrue, sigestimated):
        return np.trapz(np.abs(sigtrue - sigestimated), dx=1)

    def calc_error_sig_squared(self, sigtrue, sigestimated):
        return np.trapz((sigtrue - sigestimated) ** 2, dx=1)

    def form_error_block(self, segs, segs_pred, l, errortype="Absolute", bnorm=False):
        curve = self.create_basic_curve(l)
        if bnorm:
            curve=norm_to_one(curve)
        errors = np.zeros(len(segs))

        if errortype == "Absolute" and not bnorm:
            for i in range(len(segs)):
                curverror = self.calc_error_sig_abs(segs[i], curve)
                errors[i] = self.calc_error_sig_abs(segs[i], segs_pred[i])
                errors[i] = -1 / curverror * errors[i] + 1
        elif not bnorm:
            for i in range(len(segs)):
                curverror = self.calc_error_sig_squared(segs[i], curve)
                errors[i] = self.calc_error_sig_squared(segs[i], segs_pred[i])
                errors[i] = -1 / curverror * errors[i] + 1

        elif errortype == "Absolute" and bnorm:
            for i in range(len(segs)):
                s = norm_to_one(segs[i])
                s_pred = norm_to_one(segs_pred[i])
                curverror = self.calc_error_sig_abs(s, curve)
                errors[i] = self.calc_error_sig_abs(s, s_pred)
                if curverror == 0:
                    curverror = 1e-4
                errors[i] = -1 / curverror * errors[i] + 1

        else:
            for i in range(len(segs)):
                s = norm_to_one(segs[i])
                s_pred = norm_to_one(segs_pred[i])
                curverror = self.calc_error_sig_squared(s, curve)
                errors[i] = self.calc_error_sig_squared(s, s_pred)
                errors[i] = -1 / curverror * errors[i] + 1
        return np.mean(errors)


def norm_to_one(sig):
    return (sig - np.min(sig)) / (np.max(sig) - np.min(sig))
