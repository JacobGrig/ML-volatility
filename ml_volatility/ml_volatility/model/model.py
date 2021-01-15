import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from scipy.optimize import fmin_slsqp


class BaseModel:
    def __init__(self, data_path):
        self.data_path = data_path
        self.rv_df = pd.read_csv(data_path / "data.csv")
        self.rv_vec = np.sqrt(self.rv_df.loc[self.rv_df["Symbol"] == ".SPX"]["rv5"].values)


class MemModel(BaseModel):
    def __init__(self, data_path):
        super().__init__(data_path)

    def __prepare_data(self):

        self.rv_vec *= 1000

        train_test_index = np.int(0.7 * self.rv_vec.size)

        self.rv_train_vec = self.rv_vec[:train_test_index]
        self.rv_test_vec = self.rv_vec[train_test_index:]

        self.init_mean = self.rv_train_vec.mean()
        self.init_var = self.rv_train_vec.var(ddof=1)

        self.start_vec = np.array([self.init_mean * 0.01, 0.02, 0.9])

        self.bound_vec = np.array([(0.0, 2 * self.init_mean), (0.0, 1.0), (0.0, 1.0)])

        self.train_size = self.rv_train_vec.size
        self.test_size = self.rv_test_vec.size

        self.psi_vec = np.ones(shape=self.train_size) * self.init_mean

        return self

    def __log_like(self, par_vec, ret_format=False):

        epsilon_vec = np.ones(shape=self.train_size)

        for t in range(1, self.train_size):
            epsilon_vec[t - 1] = self.rv_train_vec[t - 1] / self.psi_vec[t - 1]
            self.psi_vec[t] = par_vec.dot(
                [1, self.rv_train_vec[t - 1], self.psi_vec[t - 1]]
            )

        log_like_vec = self.rv_train_vec / self.psi_vec + np.log(self.psi_vec)

        if not ret_format:
            return np.sum(log_like_vec)
        else:
            return np.sum(log_like_vec), log_like_vec, np.copy(self.psi_vec)

    def __optimize(self):

        self.estimate_vec = fmin_slsqp(
            self.__log_like,
            self.start_vec,
            f_ieqcons=lambda par_vec, ret_format=False: np.array(
                [1 - par_vec[1] - par_vec[2]]
            ),
            bounds=self.bound_vec,
        )

        return self

    def __predict(self):

        n_test = self.rv_test_vec.size

        psi_vec = np.zeros(n_test + 1)
        psi_vec[0] = self.rv_train_vec[-1]

        for i_pred in np.arange(n_test):
            psi_vec[i_pred + 1] = self.estimate_vec.dot(
                [1, self.rv_test_vec[i_pred], psi_vec[i_pred]]
            )

        self.rv_pred_vec = psi_vec[1:]

        return self

    def __error(self):

        self.rv_log_test_vec = np.log(self.rv_test_vec / 1000 + 1e-10)
        self.rv_log_pred_vec = np.log(self.rv_pred_vec / 1000 + 1e-10)

        abs_error = np.mean((self.rv_test_vec / 1000 - self.rv_pred_vec / 1000) ** 2)
        log_error = np.mean((self.rv_log_test_vec - self.rv_log_pred_vec) ** 2)

        return abs_error, log_error, self.rv_test_vec / 1000, self.rv_pred_vec / 1000, self.rv_log_test_vec, self.rv_log_pred_vec

    def estimate(self):

        return self.__prepare_data().__optimize().__predict().__error()


class HarModel(BaseModel):
    def __init__(self, data_path, loss="mse", alpha=0.5):
        super().__init__(data_path)

        self.MONTH = 22
        self.WEEK = 5
        self.DAY = 1

        self.loss = loss
        self.alpha = alpha

        self.learning_rate = 0.01
        self.tol = 1e-6

    def __prepare_data(self):

        self.rv_vec = np.log(self.rv_vec + 1e-10)

        self.rv_month_vec = (
            np.convolve(self.rv_vec, np.ones(self.MONTH, dtype=int), "valid")[:-1]
            / self.MONTH
        )
        self.sample_size = self.rv_month_vec.size

        self.rv_week_vec = (
            np.convolve(self.rv_vec, np.ones(self.WEEK, dtype=int), "valid")[-self.sample_size - 1: -1]
            / self.WEEK
        )
        self.rv_day_vec = self.rv_vec[-self.sample_size - 1: -1]

        self.feat_mat = np.stack(
            [
                np.ones(shape=self.sample_size),
                self.rv_day_vec,
                self.rv_week_vec,
                self.rv_month_vec,
            ]
        ).T
        self.target_vec = self.rv_vec[-self.sample_size:]

        train_test_index = np.int(0.7 * self.sample_size)

        self.feat_train_mat = self.feat_mat[:train_test_index]
        self.feat_test_mat = self.feat_mat[train_test_index:]

        self.target_train_vec = self.target_vec[:train_test_index]
        self.target_test_vec = self.target_vec[train_test_index:]

        self.init_mean = self.feat_train_mat.mean(axis=0)
        self.init_var = self.feat_train_mat.var(axis=0, ddof=1)

        self.start_vec = np.array(
            [
                self.init_mean[0] * 0.01,
                0.9,
                0.1,
                0.1
            ]
        )

        self.weight_vec = self.start_vec

        return self

    def __gradient(self):

        target_est_vec = self.feat_train_mat @ self.weight_vec

        delta_vec = np.reshape(self.target_train_vec.flatten() - target_est_vec, newshape=(-1, 1))

        if self.loss == "mse":
            grad_vec = -2 * self.feat_train_mat.T @ delta_vec / self.feat_train_mat.shape[0]
            error = np.sum(delta_vec ** 2)

        elif self.loss == "linex":
            grad_vec = self.feat_train_mat.T @ (
                np.ones(shape=(self.feat_train_mat.shape[0], 1)) * self.alpha
                - self.alpha * np.exp(self.alpha * delta_vec)
            ) / self.feat_train_mat.shape[0]
            error = np.mean(np.exp(self.alpha * delta_vec) - self.alpha * delta_vec - 1)

        elif self.loss == "als":
            grad_vec = -(
                2 * self.feat_train_mat.T
                @ (delta_vec * np.abs(self.alpha - np.int64(np.less(delta_vec, 0))))
                / self.feat_train_mat.shape[0]
            )

            error = np.sum(delta_vec ** 2 * np.abs(self.alpha - np.int64(np.less(delta_vec, 0))))
        else:
            grad_vec = None
            error = None

        return grad_vec, error

    def __optimize(self):

        iteration = 0

        while True:
            iteration += 1
            grad_vec, delta = self.__gradient()

            if iteration % 1000 == 0:
                print(f"Iteration: {iteration}, loss: {delta}")
            grad_vec = grad_vec.flatten()
            weight_vec = self.weight_vec - self.learning_rate * grad_vec

            if np.sum(np.abs(weight_vec - self.weight_vec)) < self.tol:
                self.estimate_vec = weight_vec

                return self

            self.weight_vec = weight_vec

    def __predict(self):

        self.target_pred_vec = (self.feat_test_mat @ self.weight_vec).flatten()

        return self

    def __error(self):

        delta_vec = np.exp(self.target_test_vec) - np.exp(self.target_pred_vec)

        delta_log_vec = self.target_test_vec - self.target_pred_vec

        if self.loss == "mse":
            abs_error = np.mean(delta_vec ** 2)
            log_error = np.mean(delta_log_vec ** 2)

        elif self.loss == "linex":
            abs_error = np.mean(np.exp(self.alpha * delta_vec) - self.alpha * delta_vec - 1)
            log_error = np.mean(np.exp(self.alpha * delta_log_vec) - self.alpha * delta_log_vec - 1)

        elif self.loss == "als":
            abs_error = np.mean(delta_vec ** 2 * np.abs(
                self.alpha - np.int64(np.less(delta_vec, 0))
            ))
            log_error = np.mean(delta_log_vec ** 2 * np.abs(
                self.alpha - np.int64(np.less(delta_log_vec, 0))
            ))

        else:
            abs_error = None
            log_error = None

        return abs_error, log_error, np.exp(self.target_test_vec), np.exp(self.target_pred_vec), self.target_test_vec, self.target_pred_vec

    def estimate(self):

        return self.__prepare_data().__optimize().__predict().__error()


class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size)

        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (
            torch.zeros(1, 1, self.hidden_layer_size),
            torch.zeros(1, 1, self.hidden_layer_size),
        )

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(
            input_seq.view(len(input_seq), 1, -1), self.hidden_cell
        )
        prediction_vec = self.linear(lstm_out.view(len(input_seq), -1))
        return prediction_vec[-1]


class LstmModel(BaseModel):
    def __init__(self, data_path, loss="mse", alpha=0.5):
        def linex_loss(pred_vec, target_vec):
            delta_vec = target_vec - pred_vec
            return torch.sum(
                torch.exp(self.alpha * delta_vec) - self.alpha * delta_vec - 1
            )

        def als_loss(pred_vec, target_vec):
            delta_vec = target_vec - pred_vec
            return torch.mean(
                delta_vec ** 2 * torch.abs(self.alpha - torch.less(delta_vec, 0).type(torch.DoubleTensor))
            )

        super().__init__(data_path)

        self.loss = loss
        self.alpha = alpha

        self.learning_rate = 0.01
        self.tol = 1e-5

        self.depth = 10
        self.n_epochs = 15

        if self.loss == "mse":
            self.loss_function = nn.MSELoss()
        elif self.loss == "linex":
            self.loss_function = linex_loss
        elif self.loss == "als":
            self.loss_function = als_loss
        else:
            self.loss_function = nn.MSELoss()

    @staticmethod
    def __create_inout_sequences(input_vec, window_size):
        inout_list = []
        input_size = np.array(input_vec.size())[0]
        for i in np.arange(input_size - window_size):
            train_seq = input_vec[i: i + window_size]
            train_label = input_vec[i + window_size: i + window_size + 1]
            inout_list.append((train_seq, train_label))

        return inout_list

    def __prepare_data(self):

        self.rv_vec = np.log(self.rv_vec + 1e-10)

        train_test_index = np.int(0.7 * self.rv_vec.size)

        self.rv_train_vec = self.rv_vec[:train_test_index]
        self.rv_test_vec = self.rv_vec[train_test_index:]

        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.rv_train_scaled_vec = self.scaler.fit_transform(self.rv_train_vec.reshape(-1, 1))

        self.rv_test_scaled_vec = self.scaler.transform(self.rv_test_vec.reshape(-1, 1))

        self.rv_train_scaled_vec = torch.FloatTensor(self.rv_train_scaled_vec).view(-1)

        self.train_list = self.__create_inout_sequences(
            self.rv_train_scaled_vec, self.depth
        )

        return self

    def __optimize(self):

        self.model = LSTM()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        single_loss = 0

        for i_epoch in np.arange(self.n_epochs):
            for seq_vec, label in self.train_list:
                optimizer.zero_grad()
                self.model.hidden_cell = (
                    torch.zeros(1, 1, self.model.hidden_layer_size),
                    torch.zeros(1, 1, self.model.hidden_layer_size),
                )

                pred = self.model(seq_vec)

                single_loss = self.loss_function(pred, label)
                single_loss.backward()
                optimizer.step()

            print(f'epoch: {i_epoch:3}, loss: {single_loss.item():10.8f}')

        return self

    def __predict(self):

        self.model.eval()

        self.rv_test_list = self.rv_train_scaled_vec[-self.depth:].tolist() + self.rv_test_scaled_vec.flatten().tolist()
        self.rv_pred_vec = []

        for i_elem in np.arange(self.rv_test_scaled_vec.size):
            seq_vec = torch.FloatTensor(self.rv_test_list[i_elem: i_elem + self.depth])
            with torch.no_grad():
                self.model.hidden = (
                    torch.zeros(1, 1, self.model.hidden_layer_size),
                    torch.zeros(1, 1, self.model.hidden_layer_size),
                )
                self.rv_pred_vec.append(self.model(seq_vec).item())

        self.rv_pred_vec = self.scaler.inverse_transform(np.array(self.rv_pred_vec).reshape(-1, 1))

        return self

    def __error(self):

        delta_vec = np.exp(self.rv_test_vec) - np.exp(self.rv_pred_vec)

        delta_log_vec = self.rv_test_vec - self.rv_pred_vec

        if self.loss == "mse":
            abs_error = np.mean(delta_vec ** 2)
            log_error = np.mean(delta_log_vec ** 2)

        elif self.loss == "linex":
            abs_error = np.mean(np.exp(self.alpha * delta_vec) - self.alpha * delta_vec - 1)
            log_error = np.mean(np.exp(self.alpha * delta_log_vec) - self.alpha * delta_log_vec - 1)

        elif self.loss == "als":
            abs_error = np.mean(delta_vec ** 2 * np.abs(
                self.alpha - np.int64(np.less(delta_vec, 0))
            ))
            log_error = np.mean(delta_log_vec ** 2 * np.abs(
                self.alpha - np.int64(np.less(delta_log_vec, 0))
            ))

        else:
            abs_error = None
            log_error = None

        return abs_error, log_error, np.exp(self.rv_test_vec), np.exp(self.rv_pred_vec), self.rv_test_vec, self.rv_pred_vec

    def estimate(self):

        return self.__prepare_data().__optimize().__predict().__error()


class RandForestModel(BaseModel):
    def __init__(self, data_path):
        super().__init__(data_path)

        self.learning_rate = 0.01
        self.tol = 1e-5

        self.depth = 10
        self.n_epochs = 150

    @staticmethod
    def __split_sequence(seq_vec, n_steps):

        data_mat, target_vec = list(), list()

        for idx in np.arange(seq_vec.size):
            end_ix = idx + n_steps

            if end_ix > len(seq_vec) - 1:
                break

            seq_x, seq_y = seq_vec[idx:end_ix], seq_vec[end_ix]
            data_mat.append(seq_x)
            target_vec.append(seq_y)

        return np.array(data_mat), np.array(target_vec)

    def __prepare_data(self):

        self.feat_mat, self.target_vec = self.__split_sequence(self.rv_vec, self.depth)

        train_test_index = np.int(0.7 * self.target_vec.size)

        self.feat_train_mat = self.feat_mat[:train_test_index]
        self.feat_test_mat = self.feat_mat[train_test_index:]

        self.target_train_vec = self.target_vec[:train_test_index]
        self.target_test_vec = self.target_vec[train_test_index:]

        return self

    def __optimize(self):

        self.model = RandomForestRegressor(random_state=0)

        self.model.fit(self.feat_train_mat, self.target_train_vec)

        return self

    def __predict(self):

        self.target_pred_vec = self.model.predict(self.feat_test_mat)

        return self

    def __error(self):

        self.target_log_test_vec = np.log(self.target_test_vec + 1e-10)
        self.target_log_pred_vec = np.log(self.target_pred_vec + 1e-10)

        abs_error = np.mean((self.target_test_vec - self.target_pred_vec) ** 2)
        log_error = np.mean((self.target_log_test_vec - self.target_log_pred_vec) ** 2)

        return abs_error, log_error, self.target_test_vec, self.target_pred_vec, self.target_log_test_vec, self.target_log_pred_vec

    def estimate(self):

        return self.__prepare_data().__optimize().__predict().__error()


if __name__ == "__main__":
    pass
