import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ml_volatility.model.model import MemModel, HarModel, LstmModel, RandForestModel


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
            abs_error, log_error, abs_test_vec, abs_pred_vec, log_test_vec, log_pred_vec = model.estimate()
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

            plt.plot(t_vec[:200], cur_abs_test_vec[:200], "r", linewidth=1, label="True values of $ RV_t $")
            plt.plot(t_vec[:200], cur_abs_pred_vec[:200], "b", linewidth=1, label="Predicted values of $ RV_t $")
            plt.grid()
            plt.xlabel("time")
            plt.ylabel("$ RV_t $")
            plt.legend()
            plt.savefig(self.res_link / f"{col.replace(' ', '_')}_RV.eps", format="eps")
            plt.close()

            plt.plot(t_vec[:200], cur_log_test_vec[:200], "r", linewidth=1, label="True values of $ log(RV_t) $")
            plt.plot(t_vec[:200], cur_log_pred_vec[:200], "b", linewidth=1, label="Predicted values of $ log(RV_t) $")
            plt.grid()
            plt.xlabel("time")
            plt.ylabel("$ log(RV_t) $")
            plt.legend()
            plt.savefig(self.res_link / f"{col.replace(' ', '_')}_log(RV).eps", format="eps")
            plt.close()

        return self

    def run(self):

        return self.__estimate().__save_results()
