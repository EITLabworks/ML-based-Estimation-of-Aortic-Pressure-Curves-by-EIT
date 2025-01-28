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
        self.bByIndex = False
        self.bByParatype = True
        self.byMeanForm = True
        self.byMinMax = True
        self.bVisualMetrics = True
        self.bStandardMetrics = True

        # Metrics
        self.bMAE = True
        self.bMSE = True
        self.bMedianAE = True
        self.bMeanError = True
        self.bSTD = True  # and variance of the error
        self.bRScore = True  # Pearson coefficient tscore and p-value
        self.bRsScore = True  # Spearman cooeficient
        self.bR2Score = True
        self.bExplainedvarianceScore = True
        self.bDScore = True  # Mean poisson gamma and tweedle variance
        self.bD2TweedleScore = True
        self.bD2AbsError = True

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
                        l.update({"SpearmanRs": self.metrics[q]["Standard"]["SpearmanRs"]})
                        l.update({"R2": self.metrics[q]["Standard"]["R2"]})
                        l.update({"MeanError": self.metrics[q]["Standard"]["MeanError"]})
                        l.update({"VarError": self.metrics[q]["Standard"]["VarError"]})
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
    def calc_varstd_quotient(self, y_true, y_predict):
        """
        Calculates the quotient of variance_predicted to variance_true and std_predicted/std_true (of 1D or 2D data)
        """
        var = np.var(y_true)
        var_pr = np.var(y_predict)
        std_true = np.std(y_true)
        std_pr = np.std(y_predict)
        return var_pr / var, std_pr / std_true

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
    def calc_spearman(self, y_true, y_predict):
        """
        Calculates the Spearman coefficient (Rs-value)  (of only 1D data)
        """
        if self.bRsScore:
            rs = spearmanr(y_true, y_predict)
            try:
                rs_val = rs.statistic
            except:
                rs_val = rs.correlation
            return rs_val

    # -------------------------------------- #
    def calc_r2(self, y_true, y_predict):
        """
        Calculates the R2 value (of 1D or 2D data)
        """
        if self.bR2Score:
            r2 = r2_score(y_true, y_predict, multioutput="variance_weighted")
            return r2

    # -------------------------------------- #
    def calc_explained_var_score(self, y_true, y_predict):
        """
        Calculates the explained variance value (of 1D or 2D data)
        """
        if self.bExplainedvarianceScore:
            EVS = explained_variance_score(y_true, y_predict, multioutput="variance_weighted")
            return EVS

    # -------------------------------------- #
    def calc_d(self, y_true, y_predict):
        """
        Calculates the Mean-Tweedie-deviance (of 1D only)
        """
        if self.bDScore:
            d = mean_tweedie_deviance(y_true, y_predict, power=1.9)
            return d

    # -------------------------------------- #
    def calc_d2_tweedie(self, y_true, y_predict):
        """
        Calculates the D2 -Tweedie score (of 1D only)
        """
        if self.bD2TweedleScore:
            d2 = d2_tweedie_score(y_true, y_predict, power=1.9)
            return d2

    # -------------------------------------- #
    def calc_d2_abserror(self, y_true, y_predict):
        """
        Calculates the D2-absolute error score (of 1D or 2D data)
        """
        if self.bD2AbsError:
            d2 = d2_absolute_error_score(y_true, y_predict)
            return d2

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
        print("1 Memory used " + str(process.memory_info().rss / 1024))
        d = {}
        if self.bStandardMetrics:
            b = self.run_metrics(y_true, y_predict)
            d.update({"Standard": b})
        print("2 Memory used " + str(process.memory_info().rss / 1024))

        if self.byMeanForm and bParas == False:
            s = {}
            mean_true, mean_predict, form_true, form_predict = self.sort_mean_form(y_true, y_predict)
            s.update({"Mean": self.run_metrics(mean_true, mean_predict)})
            s.update({"Form": self.run_metrics(form_true, form_predict)})
            print("2a1) Memory used " + str(process.memory_info().rss / 1024))
            s["Form"].update({"FormMAE":self.form_error_block(form_true, form_predict, 1024, errortype="Absolute")})
            s["Form"].update({"FormMSE":self.form_error_block(form_true, form_predict, 1024, errortype="Squared")})
            print("2a) Memory used " + str(process.memory_info().rss / 1024))
            s["Form"].update({"FormMAENorm":self.form_error_block(form_true, form_predict, 1024, errortype="Absolute", bnorm=True)})
            s["Form"].update({"FormMSENorm":self.form_error_block(form_true, form_predict, 1024, errortype="Squared", bnorm=True)})
            print("2b) Memory used " + str(process.memory_info().rss / 1024))
            d.update({"ByMeanForm": s})

          #  gc.collect()
        print("3Memory used " + str(process.memory_info().rss / 1024))

        if self.byMinMax and bParas==False:
            s = {}
            min_true, min_predict, max_true, max_predict = self.sort_min_max(y_true, y_predict)
            s.update({"Min": self.run_metrics(min_true, min_predict)})
            s.update({"Max": self.run_metrics(max_true, max_predict)})
            d.update({"ByMinMax": s})
            if self.byMeanForm:
                self.visu_mean_results( mean_true, mean_predict,min_true, min_predict, max_true, max_predict, name, d, bSave, bShow=bShow)
                del mean_true, mean_predict, form_true, form_predict
            del min_true, min_predict, max_true, max_predict
            gc.collect()
        print("4Memory used " + str(process.memory_info().rss / 1024))
        if self.bByIndex:
            s = {}
            for i in range(len(y_true[0])):
                b = self.run_metrics(y_true[:, i], y_predict[:, i])
                s.update({str(i): b})
            d.update({"ByIndex": s})

        print("5Memory used " + str(process.memory_info().rss / 1024))
        if self.bByParatype and bParas == True:
            s = {}
            l = self.sort_by_paratype(y_true, y_predict, paratype)
            for t, val in l.items():
                s.update({t: self.run_metrics(val[0], val[1])})
            d.update({"ByParatype": s})

        print("6Memory used " + str(process.memory_info().rss / 1024))

        self.metrics.update({name: d})
        if self.bVisualMetrics:
            self.visu_metrics(y_true, y_predict, name, d, bSave, bShow=bShow)
            print("6b) Memory used " + str(process.memory_info().rss / 1024))
            self.visu_by_index(y_true, y_predict, name, d, bSave, bShow=bShow)
        print("7Memory used " + str(process.memory_info().rss / 1024))

    # -------------------------------------- #
    def visu_metrics(self, y_true, y_predict, name, d, bSave, bShow=False):
        """
        Visualize certain metrics
        """
        fig, axs = plt.subplots(ncols=2, figsize=(15, 11))
        print("6a) Memory used " + str(process.memory_info().rss / 1024))

        axs[0].scatter(y_true,y_predict, alpha=0.1, s=25)
        print("6a2) Memory used " + str(process.memory_info().rss / 1024))

        plt.subplots_adjust(left=0.079, top=0.936, right=0.957, bottom=0.08)
        for axi in axs.flat:
            for axis in ["top", "bottom", "left", "right"]:
                axi.spines[axis].set_linewidth(2)
        for i in range(2):
            axs[i].set_facecolor("whitesmoke")

        axs[0].set_title("Actual vs. Predicted values", fontsize=20)
        axs[1].scatter(y_true, y_true-y_predict, alpha=0.1, s=25)
        print("6a3) Memory used " + str(process.memory_info().rss / 1024))

        axs[1].set_title("Residuals vs. Predicted Values", size=20)
        axs[0].set_xlabel("Actual values", fontsize=20, loc="right")
        axs[1].set_xlabel("Predicted values", fontsize=20, loc="right")
        axs[0].set_ylabel("Predicted values", fontsize=20, loc="top")
        axs[1].set_ylabel("Residuals (actual - predicted)", fontsize=20, loc="top")
        fig.suptitle("Plotting predictions for " + name, size=24)
        plt.tight_layout()
        if bSave:
            fig.savefig(self.fSavePath + name + "PredictionError.png")
        if bShow:
            plt.show()
        gc.collect()


    def visu_metrics2(self, ymean_true, ymean_predict, min_true, min_predict, max_true, max_predict, name, bSave, bShow=False):
        """
        Visualize certain metrics min max mean
        """
        fig, axs = plt.subplots(ncols=3, figsize=(20, 7.1))
        limits= [np.min(min_true)-5, np.max(min_true)+5]
        line= np.arange(limits[0], limits[1], 1)
        axs[0].plot(line, line, color="black", linewidth=2, linestyle="--")

        limits= [np.min(ymean_true)-5, np.max(ymean_true)+5]
        line= np.arange(limits[0], limits[1], 1)
        axs[1].plot(line, line, color="black", linewidth=2, linestyle="--")

        limits= [np.min(max_true)-5, np.max(max_true)+5]
        line= np.arange(limits[0], limits[1], 1)
        axs[2].plot(line, line, color="black", linewidth=2, linestyle="--")
        axs[0].scatter(min_true,min_predict, alpha=0.1, s=25, color="steelblue")

        plt.subplots_adjust(left=0.079, top=0.936, right=0.957, bottom=0.08)
        for axi in axs.flat:
            for axis in ["top", "bottom", "left", "right"]:
                axi.spines[axis].set_linewidth(2)
        for i in range(3):
            axs[i].set_facecolor("whitesmoke")

      #  fig.suptitle("Actual vs.Predicted values", fonsize=20)
        axs[0].set_title("Minima", fontsize=20)
        axs[1].scatter(ymean_true,ymean_predict, alpha=0.1, s=25, color="steelblue")
        axs[2].scatter(max_true,max_predict, alpha=0.1, s=25 ,color="steelblue")
        axs[1].set_title("Mean", size=20)
        axs[2].set_title("Maxima", size=20)
        for i in range(3):
            axs[i].set_xlabel("Actual values", fontsize=20, loc="right")
            axs[i].set_ylabel("Predicted values", fontsize=20, loc="top")
        #axs[1].set_ylabel("Residuals (actual - predicted)", fontsize=20, loc="top")
        fig.suptitle("Plotting predictions for " + name, size=24)
        plt.tight_layout()
        if bSave:
            fig.savefig(self.fSavePath + name + "PredictionErrorMinMaxMean.png")
        if bShow:
            plt.show()
        gc.collect()

    def visu_metrics_paper(self, ymean_true, ymean_predict, min_true, min_predict, max_true, max_predict, name, bSave, bShow=False):
        """
        Visualize certain metrics min max mean
        """
        plt.rcParams["font.family"] = "Times New Roman"
        fig, axs = plt.subplots(ncols=3, figsize=(20, 7.1))
        axs[0].scatter(min_true,min_predict, alpha=0.1, s=25, color="steelblue")

        plt.subplots_adjust(left=0.079, top=0.936, right=0.957, bottom=0.08)
        for axi in axs.flat:
            for axis in ["top", "bottom", "left", "right"]:
                axi.spines[axis].set_linewidth(2)
        for i in range(3):
            axs[i].set_facecolor("whitesmoke")

        limits= [np.min(min_true)-5, np.max(min_true)+5]
        line= np.arange(limits[0], limits[1], 1)
        axs[0].plot(line, line, color="black", linewidth=2, linestyle="--")

        limits= [np.min(ymean_true)-5, np.max(ymean_true)+5]
        line= np.arange(limits[0], limits[1], 1)
        axs[1].plot(line, line, color="black", linewidth=2, linestyle="--")

        limits= [np.min(max_true)-5, np.max(max_true)+5]
        line= np.arange(limits[0], limits[1], 1)
        axs[2].plot(line, line, color="black", linewidth=2, linestyle="--")


        #  fig.suptitle("Actual vs.Predicted values", fonsize=20)
        axs[0].set_title("Minima", fontsize=20)
        axs[1].scatter(ymean_true,ymean_predict, alpha=0.1, s=25, color="steelblue")
        axs[2].scatter(max_true,max_predict, alpha=0.1, s=25, color="steelblue")

        axs[1].set_title("Mean", size=20)
        axs[2].set_title("Maxima", size=20)
        for i in range(3):
            axs[i].set_xlabel("Real Pressures [mm Hg]", fontsize=20, loc="right")
            axs[i].set_ylabel("Estimated Pressures [mm Hg]", fontsize=20, loc="top")
        # axs[1].set_ylabel("Residuals (actual - predicted)", fontsize=20, loc="top")
    #    fig.suptitle("Plotting predictions for " + name, size=24)
        plt.tight_layout()
        if bSave:
            fig.savefig(self.fSavePath + name + "PredictionErrorMinMaxMean.png")
        if bShow:
            plt.show()

        #todo
        plt.show()


    def visu_by_index(self, y_true, y_predict, name, d, bSave, bShow=False):
        if self.bByIndex:
            mean_error = []
            var_error = []
            for i in range(len(y_true[0])):
                mean_error.append(d["ByIndex"][str(i)]["MeanError"])
                var_error.append(d["ByIndex"][str(i)]["VarError"])

            x = np.arange(1, len(y_true[0]) + 1)
            fig, ax1 = plt.subplots(figsize=(16, 10))
            ax1.set_facecolor("whitesmoke")
            plt.subplots_adjust(left=0.079, top=0.936, right=0.94, bottom=0.08)
            for axis in ["top", "bottom", "left", "right"]:
                ax1.spines[axis].set_linewidth(2)
            plot1 = ax1.plot(x, mean_error, color="blue", linewidth=2, label="Mean Error")
            ax1.set_xlabel("Index", fontsize=20, loc="right")
            ax1.set_ylabel("Mean Error", fontsize=20, loc="top")
            ax1.set_title("Mean Error and Error Variance per index", fontsize=24)
            ax2 = ax1.twinx()
            ax2.set_ylabel("Error Variance", fontsize=20, loc="top")
            plot2 = ax2.plot(x, var_error, color="crimson", linewidth=2, label="Error Variance")
            lns = plot1 + plot2
            labels = [l.get_label() for l in lns]
            plt.legend(lns, labels, loc=0)
            if bSave:
                fig.savefig(self.fSavePath + name + "MeanErrorVar.png")
            if bShow:
                plt.show()


    def visu_mean_results(self, ymean_true, ymean_predict, min_true, min_predict, max_true, max_predict, name, d, bSave, bShow=False):
        fontSize = 22
        title_Size = 24

        plt.rcParams["font.family"] = "Times New Roman"
        plt.rc('xtick', labelsize=fontSize)
        plt.rc('ytick', labelsize=fontSize)
        fig, ax = plt.subplots(figsize=(17, 12))
        # Creating plot
        ax.set_facecolor("whitesmoke")
        bp1 = ax.violinplot([min_true-min_predict, ymean_true-ymean_predict, max_true-max_predict], widths=0.5, showmeans=True)

        # ax.legend(fontsize=fontSize)
        ax.grid(linewidth=0.4)
        i = 0
        colors = ["lightblue", "steelblue", "teal"]
        for pc in bp1["bodies"]:
            pc.set_facecolor(colors[i])
            #    pc.set_edgecolor("black")
            pc.set_linewidth(3)
            pc.set_alpha(0.7)
            i += 1
        bp1["cmeans"].set_color("black")
        bp1["cmins"].set_color("black")
        bp1["cmaxes"].set_color("black")
        bp1["cbars"].set_color("black")

        def set_axis_style(ax, labels):
            ax.set_xticks(np.arange(1, len(labels) + 1))

        ax.set_xticklabels(["Minima", "Mean", "Maxima"], fontsize=fontSize)
        set_axis_style(ax, ["Minima", "Mean", "Maxima"])
        ax.set_title(name + " Min Max Mean from Aortic Pressure Curve Estimation", size=title_Size)
        ax.set_ylabel("Error [mm Hg]", size=fontSize, loc="top")
        # ax.set_xlabel("Extracted parameters ", size=fontSize, loc="right")

        plt.subplots_adjust(left=0.079, top=0.936, right=0.957, bottom=0.1)
        for axis in ["top", "bottom", "left", "right"]:
            ax.spines[axis].set_linewidth(2)

        if bSave:
            fig.savefig(self.fSavePath + name + "Error_of_mean_min_max.png")
        if bShow:
           plt.show()
        #todo
        plt.show()

        self.visu_metrics_paper(ymean_true, ymean_predict, min_true, min_predict, max_true, max_predict,name+" Paper", bSave, bShow=bShow)
        self.visu_metrics2(ymean_true, ymean_predict, min_true, min_predict, max_true, max_predict,name+" of Min,Mean and Max", bSave, bShow=bShow)
        self.visu_metrics(ymean_true, ymean_predict, name+" of the Mean ", d, bSave, bShow=bShow)


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
        vq, sq = self.calc_varstd_quotient(y_true, y_predict)
        di.update({"VarQ": vq})
        di.update({"StdQ": sq})
        r, t, p = self.calc_pearson(y_true.flatten(), y_predict.flatten())
        di.update({"PearsonR": r})
        di.update({"PearsonT": t})
        di.update({"PearsonP": p})
        di.update({"SpearmanRs": self.calc_spearman(y_true.flatten(), y_predict.flatten())})
        di.update({"R2": self.calc_r2(y_true, y_predict)})
        di.update({"EVS": self.calc_explained_var_score(y_true, y_predict)})
        #    di.update({"D": self.calc_d(y_true.flatten(), y_predict.flatten())})
        #  di.update({"D2Tweedie": self.calc_d2_tweedie(y_true.flatten(), y_predict.flatten())})
        di.update({"D2AE": self.calc_d2_abserror(y_true, y_predict)})
        return di

    # -------------------------------------- #
    def sort_by_paratype(self, y_true, ypredict, paratype):
        """
        Internal function to sort the values by parameter type (amplitudes, positions, means, standard_deviations...)
        """
        l = len(y_true[0])
        if paratype == "Linear":
            k = np.arange(0, l, 2)
            y1_true = y_true[:, k]
            y1_predict = ypredict[:, k]
            k = k + 1
            y2_true = y_true[:, k]
            y2_predict = ypredict[:, k]
            return {"Position": [y1_true, y1_predict], "Amplitude": [y2_true, y2_predict]}

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
    def sort_min_max(self, y_true, y_predict):
        """
        Calc min and ax of each curve
        :return: True mean values, Predicted mean values, true shape vectors, estimated shape vectors
        """
        min_true = np.min(y_true, axis=1)
        min_predict = np.min(y_predict, axis=1)
        max_true = np.max(y_true, axis=1)
        max_predict = np.max(y_predict, axis=1)
        return min_true, min_predict, max_true, max_predict

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
