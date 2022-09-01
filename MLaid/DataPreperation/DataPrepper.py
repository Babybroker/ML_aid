import numpy as np
import pandas as pd

from xgboost import DMatrix
from sklearn.preprocessing import StandardScaler


def create_artifact_folder():
    from datetime import datetime
    from pathlib import Path
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f'logs/artifacts/model_run{now}/'
    Path(folder_name).mkdir(parents=True, exist_ok=True)
    return folder_name, now


class DataPrepper:
    def __init__(
            self,
            target_column: str,
            x_cols: list,
            x_norm_cols: list = None,
            scale_values: bool = True,
            convert_to: str = 'tf_tensor'):
        self.x_cols = x_cols
        self.x_norm_cols = x_norm_cols
        self.target_col = target_column
        self.scale_values = scale_values
        self.x_scaler, self.y_scaler = StandardScaler(), StandardScaler(),
        self.convert_to = convert_to

    def clip_values(self, dataf: pd.DataFrame, cols_to_clip: list = None, window_length=30):
        if cols_to_clip is None:
            cols_to_clip = self.x_norm_cols

        for col in cols_to_clip:
            std_3 = dataf[col].rolling(window=window_length).std() * 3
            mean = dataf[col].rolling(window=window_length).mean()
            dataf[col] = dataf[col].clip(lower=mean - std_3, upper=mean + std_3)
        return dataf

    def _x_norm(self, dataset, is_train=False):
        norm_slice = dataset[self.x_norm_cols].values
        if len(dataset) == 1:
            norm_slice.reshape(1, -1)
        dataset[self.x_norm_cols] = self.x_scaler.fit_transform(norm_slice) if is_train else self.x_scaler.transform(norm_slice)
        return dataset

    def _y_norm(self, dataset):
        dataset[self.target_col] = self.y_scaler.fit_transform(np.array(dataset[self.target_col]).reshape(-1, 1))
        return dataset

    def _get_matrix(self, df, add_target=True):
        return DMatrix(data=df[self.x_cols], label=df[self.target_col] if add_target else None)

    def create_scaled_dataset(self, dataset, is_train=False):
        dataset = dataset.copy()
        if self.scale_values:
            dataset = self._x_norm(dataset, is_train=is_train)
            dataset = self._y_norm(dataset)
        if self.convert_to == 'tf_tensor':
            from tensorflow import convert_to_tensor
            return convert_to_tensor(dataset[self.x_cols]), dataset[self.target_col].values

        elif self.convert_to == 'xgb_matrix':
            return self._get_matrix(dataset)

        elif self.convert_to is None:
            return dataset[self.x_cols + [self.target_col]]

        else:
            print('Unknown conversion request')

    def scale_x_values(self, dataset):
        if self.convert_to != 'xgb_matrix':
            return [self._x_norm(dataset, is_train=False)]
        else:
            return self._get_matrix(self._x_norm(dataset, is_train=False), add_target=False)

    def inverse_transform_y_val(self, y_slice):
        if len(y_slice) == 1:
            y_slice = y_slice.reshape(1, -1)
        if y_slice.shape[0] != 1:
            y_slice = y_slice.reshape(-1, 1)
        return self.y_scaler.inverse_transform(y_slice)
