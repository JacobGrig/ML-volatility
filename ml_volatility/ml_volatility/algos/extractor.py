import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance

from ml_volatility.model.model import (
    MemModel,
    HarModel,
    LstmModel,
    RandForestModel,
    HarModelWithJumps,
    LstmModelWithJumps,
    RandForestModelWithJumps,
    HarModelOLS,
    HarCJModelOLS,
    HarCJModModelOLS,
    LstmModModel,
)


class Extractor:
    def __init__(self, data_link, res_link):
        self.res_link = res_link

        self.model_list = []

        np.random.seed(0)

        self.model_list.append(MemModel(data_link))
        self.model_list.append(HarModel(data_link, "mse"))
        self.model_list.append(HarModel(data_link, "linex", alpha=0.5))
        self.model_list.append(HarModel(data_link, "als", alpha=0.7))
        self.model_list.append(LstmModel(data_link, "mse"))
        self.model_list.append(LstmModel(data_link, "linex", alpha=0.5))
        self.model_list.append(LstmModel(data_link, "als", alpha=0.7))
        self.model_list.append(RandForestModel(data_link))

        self.model_abs_error_list = []
        self.model_log_error_list = []
        self.model_abs_true_test_list = []
        self.model_abs_pred_test_list = []
        self.model_log_true_test_list = []
        self.model_log_pred_test_list = []

    def __estimate(self):
        for model in self.model_list:
            (
                abs_error,
                log_error,
                abs_test_vec,
                abs_pred_vec,
                log_test_vec,
                log_pred_vec,
            ) = model.estimate()
            self.model_abs_error_list.append(abs_error)
            self.model_log_error_list.append(log_error)
            self.model_abs_true_test_list.append(abs_test_vec)
            self.model_abs_pred_test_list.append(abs_pred_vec)
            self.model_log_true_test_list.append(log_test_vec)
            self.model_log_pred_test_list.append(log_pred_vec)

        return self

    def __save_results(self):

        col_vec = [
            "MEM with QMLE",
            "HAR with MSE",
            "HAR with LinEx",
            "HAR with ALS",
            "LSTM with MSE",
            "LSTM with LinEx",
            "LSTM with ALS",
            "Random Forest with MSE",
        ]

        res_df = pd.DataFrame(columns=col_vec)

        abs_error_series = pd.Series(
            dict(zip(col_vec, self.model_abs_error_list)), name="abs_error"
        )

        log_error_series = pd.Series(
            dict(zip(col_vec, self.model_log_error_list)), name="log_error"
        )

        res_df = res_df.append(abs_error_series)
        res_df = res_df.append(log_error_series)

        res_df.to_csv(self.res_link / "error.csv")

        for (i_col, col) in enumerate(col_vec):
            cur_abs_test_vec = self.model_abs_true_test_list[i_col]
            cur_abs_pred_vec = self.model_abs_pred_test_list[i_col]

            cur_log_test_vec = self.model_log_true_test_list[i_col]
            cur_log_pred_vec = self.model_log_pred_test_list[i_col]

            t_vec = np.arange(1, cur_abs_test_vec.size + 1)

            plt.plot(
                t_vec[:200],
                cur_abs_test_vec[:200],
                "r",
                linewidth=1,
                label="True values of $ RV_t $",
            )
            plt.plot(
                t_vec[:200],
                cur_abs_pred_vec[:200],
                "b",
                linewidth=1,
                label="Predicted values of $ RV_t $",
            )
            plt.grid()
            plt.xlabel("time")
            plt.ylabel("$ RV_t $")
            plt.legend()
            plt.savefig(self.res_link / f"{col.replace(' ', '_')}_RV.eps", format="eps")
            plt.close()

            plt.plot(
                t_vec[:200],
                cur_log_test_vec[:200],
                "r",
                linewidth=1,
                label="True values of $ log(RV_t) $",
            )
            plt.plot(
                t_vec[:200],
                cur_log_pred_vec[:200],
                "b",
                linewidth=1,
                label="Predicted values of $ log(RV_t) $",
            )
            plt.grid()
            plt.xlabel("time")
            plt.ylabel("$ log(RV_t) $")
            plt.legend()
            plt.savefig(
                self.res_link / f"{col.replace(' ', '_')}_log(RV).eps", format="eps"
            )
            plt.close()

        return self

    def run(self):

        return self.__estimate().__save_results()


class ExtractorWithJumps:
    def __init__(self, data_link, res_link):
        self.res_link = res_link

        self.model_list = []

        np.random.seed(0)

        self.model_list.append(HarModelWithJumps(data_link))
        self.model_list.append(LstmModelWithJumps(data_link))
        self.model_list.append(RandForestModelWithJumps(data_link))

        self.model_c_error_list = []
        self.model_j_error_list = []

        self.model_C_error_list = []
        self.model_J_error_list = []

        self.model_rv_error_list = []

        self.model_c_true_test_list = []
        self.model_c_pred_test_list = []

        self.model_j_true_test_list = []
        self.model_j_pred_test_list = []

        self.model_C_true_test_list = []
        self.model_C_pred_test_list = []

        self.model_J_true_test_list = []
        self.model_J_pred_test_list = []

    def __estimate(self):
        for model in self.model_list:
            (
                c_error,
                j_error,
                C_error,
                J_error,
                rv_error,
                c_test_vec,
                c_pred_vec,
                j_test_vec,
                j_pred_vec,
                C_test_vec,
                C_pred_vec,
                J_test_vec,
                J_pred_vec,
            ) = model.estimate()

            self.model_c_error_list.append(c_error)
            self.model_j_error_list.append(j_error)

            self.model_C_error_list.append(C_error)
            self.model_J_error_list.append(J_error)

            self.model_rv_error_list.append(rv_error)

            self.model_c_true_test_list.append(c_test_vec)
            self.model_c_pred_test_list.append(c_pred_vec)

            self.model_j_true_test_list.append(j_test_vec)
            self.model_j_pred_test_list.append(j_pred_vec)

            self.model_C_true_test_list.append(C_test_vec)
            self.model_C_pred_test_list.append(C_pred_vec)

            self.model_J_true_test_list.append(J_test_vec)
            self.model_J_pred_test_list.append(J_pred_vec)

        return self

    def __save_results(self):

        col_vec = [
            "HAR-CJ",
            "LSTM",
            "Random Forest",
        ]

        res_df = pd.DataFrame(columns=col_vec)

        c_error_series = pd.Series(
            dict(zip(col_vec, self.model_c_error_list)), name="c_error"
        )
        j_error_series = pd.Series(
            dict(zip(col_vec, self.model_j_error_list)), name="j_error"
        )

        C_error_series = pd.Series(
            dict(zip(col_vec, self.model_C_error_list)), name="C_error"
        )
        J_error_series = pd.Series(
            dict(zip(col_vec, self.model_J_error_list)), name="J_error"
        )

        rv_error_series = pd.Series(
            dict(zip(col_vec, self.model_rv_error_list)), name="rv_error"
        )

        res_df = res_df.append(c_error_series)
        res_df = res_df.append(j_error_series)

        res_df = res_df.append(C_error_series)
        res_df = res_df.append(J_error_series)

        res_df = res_df.append(rv_error_series)

        res_df.to_csv(self.res_link / "error.csv")

        for (i_col, col) in enumerate(col_vec):
            cur_c_test_vec = self.model_c_true_test_list[i_col]
            cur_c_pred_vec = self.model_c_pred_test_list[i_col]

            cur_j_test_vec = self.model_j_true_test_list[i_col]
            cur_j_pred_vec = self.model_j_pred_test_list[i_col]

            cur_C_test_vec = self.model_C_true_test_list[i_col]
            cur_C_pred_vec = self.model_C_pred_test_list[i_col]

            cur_J_test_vec = self.model_J_true_test_list[i_col]
            cur_J_pred_vec = self.model_J_pred_test_list[i_col]

            t_vec = np.arange(1, cur_c_test_vec.size + 1)

            plt.plot(
                t_vec[:200],
                cur_c_test_vec[:200],
                "r",
                linewidth=1,
                label="True values of $ c_t $",
            )
            plt.plot(
                t_vec[:200],
                cur_c_pred_vec[:200],
                "b",
                linewidth=1,
                label="Predicted values of $ c_t $",
            )
            plt.grid()
            plt.xlabel("time")
            plt.ylabel("$ c_t $")
            plt.legend()
            plt.savefig(self.res_link / f"{col.replace(' ', '_')}_c.eps", format="eps")
            plt.close()

            plt.plot(
                t_vec[:200],
                cur_j_test_vec[:200],
                "r",
                linewidth=1,
                label="True values of $ j_t $",
            )
            plt.plot(
                t_vec[:200],
                cur_j_pred_vec[:200],
                "b",
                linewidth=1,
                label="Predicted values of $ j_t $",
            )
            plt.grid()
            plt.xlabel("time")
            plt.ylabel("$ c_t $")
            plt.legend()
            plt.savefig(self.res_link / f"{col.replace(' ', '_')}_j.eps", format="eps")
            plt.close()

            plt.plot(
                t_vec[:200],
                cur_C_test_vec[:200],
                "r",
                linewidth=1,
                label="True values of $ C_t $",
            )
            plt.plot(
                t_vec[:200],
                cur_C_pred_vec[:200],
                "b",
                linewidth=1,
                label="Predicted values of $ C_t $",
            )
            plt.grid()
            plt.xlabel("time")
            plt.ylabel("$ C_t $")
            plt.legend()
            plt.savefig(self.res_link / f"{col.replace(' ', '_')}_C_large.eps", format="eps")
            plt.close()

            plt.plot(
                t_vec[:200],
                cur_J_test_vec[:200],
                "r",
                linewidth=1,
                label="True values of $ J_t $",
            )
            plt.plot(
                t_vec[:200],
                cur_J_pred_vec[:200],
                "b",
                linewidth=1,
                label="Predicted values of $ J_t $",
            )
            plt.grid()
            plt.xlabel("time")
            plt.ylabel("$ J_t $")
            plt.legend()
            plt.savefig(self.res_link / f"{col.replace(' ', '_')}_J_large.eps", format="eps")
            plt.close()

            plt.plot(
                t_vec[:200],
                cur_C_test_vec[:200] + cur_J_test_vec[:200],
                "r",
                linewidth=1,
                label="True values of $ RV^2_t $",
            )
            plt.plot(
                t_vec[:200],
                cur_C_pred_vec[:200] + cur_J_pred_vec[:200],
                "b",
                linewidth=1,
                label="Predicted values of $ RV^2_t $",
            )
            plt.grid()
            plt.xlabel("time")
            plt.ylabel("$ RV^2_t $")
            plt.legend()
            plt.savefig(self.res_link / f"{col.replace(' ', '_')}_RV.eps", format="eps")
            plt.close()

        return self

    def run(self):

        return self.__estimate().__save_results()


class FinalExtractor:
    def __init__(self, data_link, res_link):
        self.res_link = res_link

        self.model_list = []

        np.random.seed(0)

        self.model_list.append(HarModelOLS(data_link))
        self.model_list.append(HarCJModelOLS(data_link))
        self.model_list.append(HarCJModModelOLS(data_link))
        self.model_list.append(LstmModModel(data_link))
        self.model_list.append(LstmModelWithJumps(data_link))
        self.model_list.append(RandForestModel(data_link))
        self.model_list.append(RandForestModelWithJumps(data_link))

        self.model_c_error_list = []
        self.model_j_error_list = []

        self.model_C_error_list = []
        self.model_J_error_list = []

        self.model_RV_error_list = []
        self.model_rv_error_list = []

        self.model_c_true_test_list = []
        self.model_c_pred_test_list = []

        self.model_j_true_test_list = []
        self.model_j_pred_test_list = []

        self.model_C_true_test_list = []
        self.model_C_pred_test_list = []

        self.model_J_true_test_list = []
        self.model_J_pred_test_list = []

        self.model_rv_true_test_list = []
        self.model_rv_pred_test_list = []

        self.model_RV_true_test_list = []
        self.model_RV_pred_test_list = []

        self.model_c_list = []
        self.model_j_list = []
        self.model_rv_list = []

        self.model_feat_mat_list = []

    def __estimate(self):
        for (i, model) in enumerate(self.model_list):
            est = model.estimate()

            if i in (2, 4, 6):
                (
                    c_error,
                    j_error,
                    C_error,
                    J_error,
                    RV_error,
                    rv_error,
                    c_test_vec,
                    c_pred_vec,
                    j_test_vec,
                    j_pred_vec,
                    C_test_vec,
                    C_pred_vec,
                    J_test_vec,
                    J_pred_vec,
                    rv_test_vec,
                    rv_pred_vec,
                    model_c,
                    model_j,
                    feat_test_mat,
                ) = est

                self.model_c_error_list.append(c_error)
                self.model_j_error_list.append(j_error)

                self.model_C_error_list.append(C_error)
                self.model_J_error_list.append(J_error)

                self.model_c_true_test_list.append(c_test_vec)
                self.model_c_pred_test_list.append(c_pred_vec)

                self.model_j_true_test_list.append(j_test_vec)
                self.model_j_pred_test_list.append(j_pred_vec)

                self.model_C_true_test_list.append(C_test_vec)
                self.model_C_pred_test_list.append(C_pred_vec)

                self.model_J_true_test_list.append(J_test_vec)
                self.model_J_pred_test_list.append(J_pred_vec)

                if not (model_c is None):
                    self.model_c_list.append(model_c)
                    self.model_j_list.append(model_j)

            else:
                (
                    rv_error,
                    RV_error,
                    rv_test_vec,
                    rv_pred_vec,
                    RV_test_vec,
                    RV_pred_vec,
                    model_rv,
                    feat_test_mat,
                ) = est

                self.model_RV_true_test_list.append(RV_test_vec)
                self.model_RV_pred_test_list.append(RV_pred_vec)

                if not (model_rv is None):
                    self.model_rv_list.append(model_rv)

            self.model_RV_error_list.append(RV_error)
            self.model_rv_error_list.append(rv_error)

            self.model_rv_true_test_list.append(rv_test_vec)
            self.model_rv_pred_test_list.append(rv_pred_vec)

            if not (feat_test_mat is None):
                self.model_feat_mat_list.append(feat_test_mat)

        return self

    def __save_results(self):

        col1_vec = [
            "HAR",
            "HAR-CJ",
            "LSTM",
            "RF",
        ]

        ind_list = [0, 1, 3, 5]

        res1_df = pd.DataFrame(columns=col1_vec)

        rv_error_series = pd.Series(
            dict(zip(col1_vec, [self.model_rv_error_list[i] for i in ind_list])), name="rv_error"
        )
        RV_error_series = pd.Series(
            dict(zip(col1_vec, [self.model_RV_error_list[i] for i in ind_list])), name="RV_error"
        )

        res1_df = res1_df.append(rv_error_series)
        res1_df = res1_df.append(RV_error_series)

        res1_df.to_csv(self.res_link / "error1.csv")

        col2_vec = [
            "HAR-CJ",
            "LSTM",
            "RF",
        ]

        res2_df = pd.DataFrame(columns=col2_vec)

        ind_list = [2, 4, 6]

        c_error_series = pd.Series(
            dict(zip(col2_vec, self.model_c_error_list)), name="c_error"
        )
        j_error_series = pd.Series(
            dict(zip(col2_vec, self.model_j_error_list)), name="j_error"
        )

        C_error_series = pd.Series(
            dict(zip(col2_vec, self.model_C_error_list)), name="C_error"
        )
        J_error_series = pd.Series(
            dict(zip(col2_vec, self.model_J_error_list)), name="J_error"
        )

        rv_error_series = pd.Series(
            dict(zip(col2_vec, [self.model_rv_error_list[i] for i in ind_list])), name="rv_error"
        )

        RV_error_series = pd.Series(
            dict(zip(col2_vec, [self.model_RV_error_list[i] for i in ind_list])), name="RV_error"
        )

        res2_df = res2_df.append(c_error_series)
        res2_df = res2_df.append(j_error_series)

        res2_df = res2_df.append(C_error_series)
        res2_df = res2_df.append(J_error_series)

        res2_df = res2_df.append(rv_error_series)
        res2_df = res2_df.append(RV_error_series)

        res2_df.to_csv(self.res_link / "error2.csv")

        col3_vec = [
            "HAR",
            "HAR-CJ",
            "HAR-CJ (modified)",
            "LSTM",
            "LSTM (modified)",
            "RF",
            "RF (modified)"
        ]

        res3_df = pd.DataFrame(columns=col3_vec)

        rv_error_series = pd.Series(
            dict(zip(col3_vec, self.model_rv_error_list)), name="rv_error"
        )
        RV_error_series = pd.Series(
            dict(zip(col3_vec, self.model_RV_error_list)), name="RV_error"
        )

        res3_df = res3_df.append(rv_error_series)
        res3_df = res3_df.append(RV_error_series)

        res3_df.to_csv(self.res_link / "error3.csv")

        for (i_col, col) in enumerate(col2_vec):
            cur_c_test_vec = self.model_c_true_test_list[i_col]
            cur_c_pred_vec = self.model_c_pred_test_list[i_col]

            cur_j_test_vec = self.model_j_true_test_list[i_col]
            cur_j_pred_vec = self.model_j_pred_test_list[i_col]

            cur_C_test_vec = self.model_C_true_test_list[i_col]
            cur_C_pred_vec = self.model_C_pred_test_list[i_col]

            cur_J_test_vec = self.model_J_true_test_list[i_col]
            cur_J_pred_vec = self.model_J_pred_test_list[i_col]

            t_vec = np.arange(1, cur_c_test_vec.size + 1)

            plt.plot(
                t_vec[:200],
                cur_c_test_vec[:200],
                "r",
                linewidth=1,
                label="True values of $ c_t $",
            )
            plt.plot(
                t_vec[:200],
                cur_c_pred_vec[:200],
                "b",
                linewidth=1,
                label="Predicted values of $ c_t $",
            )
            plt.grid()
            plt.xlabel("time")
            plt.ylabel("$ c_t $")
            plt.legend()
            plt.savefig(self.res_link / f"{col.replace(' ', '_')}_c.eps", format="eps")
            plt.close()

            plt.plot(
                t_vec[:200],
                cur_j_test_vec[:200],
                "r",
                linewidth=1,
                label="True values of $ j_t $",
            )
            plt.plot(
                t_vec[:200],
                cur_j_pred_vec[:200],
                "b",
                linewidth=1,
                label="Predicted values of $ j_t $",
            )
            plt.grid()
            plt.xlabel("time")
            plt.ylabel("$ c_t $")
            plt.legend()
            plt.savefig(self.res_link / f"{col.replace(' ', '_')}_j.eps", format="eps")
            plt.close()

            plt.plot(
                t_vec[:200],
                cur_C_test_vec[:200],
                "r",
                linewidth=1,
                label="True values of $ C_t $",
            )
            plt.plot(
                t_vec[:200],
                cur_C_pred_vec[:200],
                "b",
                linewidth=1,
                label="Predicted values of $ C_t $",
            )
            plt.grid()
            plt.xlabel("time")
            plt.ylabel("$ C_t $")
            plt.legend()
            plt.savefig(self.res_link / f"{col.replace(' ', '_')}_C_large.eps", format="eps")
            plt.close()

            plt.plot(
                t_vec[:200],
                cur_J_test_vec[:200],
                "r",
                linewidth=1,
                label="True values of $ J_t $",
            )
            plt.plot(
                t_vec[:200],
                cur_J_pred_vec[:200],
                "b",
                linewidth=1,
                label="Predicted values of $ J_t $",
            )
            plt.grid()
            plt.xlabel("time")
            plt.ylabel("$ J_t $")
            plt.legend()
            plt.savefig(self.res_link / f"{col.replace(' ', '_')}_J_large.eps", format="eps")
            plt.close()

            plt.plot(
                t_vec[:200],
                cur_C_test_vec[:200] + cur_J_test_vec[:200],
                "r",
                linewidth=1,
                label="True values of $ RV^2_t $",
            )
            plt.plot(
                t_vec[:200],
                cur_C_pred_vec[:200] + cur_J_pred_vec[:200],
                "b",
                linewidth=1,
                label="Predicted values of $ RV^2_t $",
            )
            plt.grid()
            plt.xlabel("time")
            plt.ylabel("$ RV^2_t $")
            plt.legend()
            plt.savefig(self.res_link / f"{col.replace(' ', '_')}_RV_large.eps", format="eps")
            plt.close()

        for (i_col, col) in enumerate(col3_vec):

            cur_rv_test_vec = self.model_rv_true_test_list[i_col]
            cur_rv_pred_vec = self.model_rv_pred_test_list[i_col]

            t_vec = np.arange(1, cur_rv_test_vec.size + 1)

            plt.plot(
                t_vec[:200],
                cur_rv_test_vec[:200],
                "r",
                linewidth=1,
                label="True values of $ log(RV^2_t) $",
            )
            plt.plot(
                t_vec[:200],
                cur_rv_pred_vec[:200],
                "b",
                linewidth=1,
                label="Predicted values of $ log(RV^2_t) $",
            )
            plt.grid()
            plt.xlabel("time")
            plt.ylabel("$ log(RV^2_t) $")
            plt.legend()
            plt.savefig(self.res_link / f"{col.replace(' ', '_')}_rv.eps", format="eps")
            plt.close()

        for (i_mat, feat_mat) in enumerate(self.model_feat_mat_list):
            if i_mat == 0:
                mod = self.model_rv_list[-1]
                y_test = self.model_rv_true_test_list[-2]
                res = permutation_importance(mod, feat_mat, y_test, n_repeats=10, random_state=42, n_jobs=2)

                idx = ["$rv_{t-10}$", "$rv_{t-9}$", "$rv_{t-8}$", "$rv_{t-7}$", "$rv_{t-6}$", "$rv_{t-5}$", "$rv_{t-4}$", "$rv_{t-3}$", "$rv_{t-2}$", "$rv_{t-1}$"]

                forest_importance_series = pd.Series(res.importances_mean, index=idx)

                fig, ax = plt.subplots()
                forest_importance_series.plot.bar(yerr=res.importances_std, ax=ax)
                ax.set_ylabel("Mean accuracy decrease")
                fig.tight_layout()
                ax.grid()

                plt.savefig(self.res_link / "RF_rv_feat_imp.eps", format="eps")

            else:
                mod_c = self.model_c_list[-1]
                mod_j = self.model_j_list[-1]

                c_test = self.model_c_true_test_list[-1]
                j_test = self.model_j_true_test_list[-1]

                res_c = permutation_importance(mod_c, feat_mat, c_test, n_repeats=10, random_state=42, n_jobs=2)
                res_j = permutation_importance(mod_j, feat_mat, j_test, n_repeats=10, random_state=42, n_jobs=2)

                idx = ["$c_{t-10}$", "$c_{t-9}$", "$c_{t-8}$", "$c_{t-7}$", "$c_{t-6}$", "$c_{t-5}$", "$c_{t-4}$",
                       "$c_{t-3}$", "$c_{t-2}$", "$c_{t-1}$", "$j_{t-10}$", "$j_{t-9}$", "$j_{t-8}$", "$j_{t-7}$", "$j_{t-6}$", "$j_{t-5}$", "$j_{t-4}$",
                       "$j_{t-3}$", "$j_{t-2}$", "$j_{t-1}$"]

                forest_importance_c_series = pd.Series(res_c.importances_mean, index=idx)

                fig, ax = plt.subplots()
                forest_importance_c_series.plot.bar(yerr=res_c.importances_std, ax=ax)
                ax.set_ylabel("Mean accuracy decrease")
                fig.tight_layout()
                ax.grid()

                plt.savefig(self.res_link / "RF_c_feat_imp.eps", format="eps")

                forest_importance_j_series = pd.Series(res_j.importances_mean, index=idx)

                fig, ax = plt.subplots()
                forest_importance_j_series.plot.bar(yerr=res_j.importances_std, ax=ax)
                ax.set_ylabel("Mean accuracy decrease")
                fig.tight_layout()
                ax.grid()

                plt.savefig(self.res_link / "RF_j_feat_imp.eps", format="eps")

        for (i_mod, mod) in enumerate(self.model_rv_list[:-1]):
            print(mod.summary().as_latex(), file=open(self.res_link / f"HAR_{i_mod}.txt", "w"))

        print(self.model_c_list[0].summary().as_latex(), file=open(self.res_link / "HAR_c.txt", "w"))
        print(self.model_j_list[0].summary().as_latex(), file=open(self.res_link / "HAR_j.txt", "w"))

        return self

    def run(self):

        return self.__estimate().__save_results()
