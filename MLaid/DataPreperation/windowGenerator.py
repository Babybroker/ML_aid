import pandas as pd
from numpy import array


class WindowGenerator:
    def __init__(self,
                 categorical_features,
                 ts_cols,
                 target_col: list,
                 label_width: int = 1,
                 input_width: int = 12,
                 shift: int = 1,
                 scale_window_by_col: str = None,
                 scale_cols: list = None,
                 use_max: bool = True
                 ):
        """

        :param categorical_features:
        :param ts_cols:
        :param target_col:
        :param label_width:
        :param input_width:
        :param shift:
        :param scale_window_by_col:
        :param scale_cols:
        :param use_max:
        """
        self.label_width = label_width
        self.categorical_features = categorical_features
        self.ts_cols = ts_cols
        self.target_col = target_col
        self.input_width = input_width
        self.shift = shift
        self.scale_window_by_col = scale_window_by_col
        self.scale_cols = scale_cols
        self.use_max = use_max

    # convert history into inputs and outputs
    def to_windows(self,
                   dataf: pd.DataFrame,

                   ) -> tuple:
        """

        :param dataf:
        :return:
        """

        if isinstance(self.ts_cols[0], list):
            nmbr_ts_arrays = len(self.ts_cols)
            x_ts_list = [list() for _ in range(nmbr_ts_arrays)]
        else:
            nmbr_ts_arrays = 1

        if isinstance(self.input_width, list):
            input_width_max = max(self.input_width)
            mult_input_width = True
        else:
            input_width_max = self.input_width
            mult_input_width = False
        shift = self.shift - 1  # subtract one to solve that the index starts at 0
        y_start = input_width_max + shift

        X_ts, X_cat, Y_ts = [], [], []
        # step over the entire history one time step at a time
        if nmbr_ts_arrays == 1:
            ts_df = dataf[self.ts_cols]
        elif nmbr_ts_arrays > 1:
            ts_dfs = [dataf[cols] for cols in self.ts_cols]

        if self.label_width == 1:
            y_df = dataf[y_start:][self.target_col].values
            cat_df = dataf[y_start:][self.categorical_features].values
        else:
            y_df = dataf[self.target_col]
            cat_df = dataf[self.categorical_features]
        for i in range(0, len(dataf) - shift - input_width_max - self.label_width + 1):
            # define the end of the input sequence
            in_end = i + input_width_max
            if nmbr_ts_arrays == 1:
                ts_selection = ts_df[i:in_end].copy()
                if self.scale_window_by_col is not None:
                    if self.use_max:
                        ts_selection[self.scale_cols] = ts_selection[self.scale_cols] / ts_selection[
                            self.scale_window_by_col].max()
                    else:
                        ts_selection[self.scale_cols] = ts_selection[self.scale_cols].div(
                            ts_selection[self.scale_window_by_col], axis=0)
                X_ts.append(ts_selection.values)
            else:
                for q in range(len(x_ts_list)):
                    if mult_input_width:
                        i_start = in_end - self.input_width[q]
                    else:
                        i_start = i
                    ts_selection = ts_dfs[q][i_start:in_end].copy()
                    if self.scale_window_by_col is not None and len(self.scale_cols[q]) > 0:
                        if self.use_max:
                            ts_selection[self.scale_cols[q]] = ts_selection[self.scale_cols[q]] / ts_selection[
                                self.scale_window_by_col].max()
                        else:
                            ts_selection[self.scale_cols[q]] = ts_selection[self.scale_cols[q]].div(
                                ts_selection[self.scale_window_by_col], axis=0)
                    x_ts_list[q].append(ts_selection.values)

            if self.label_width != 1:
                Y_ts.append(y_df[in_end + shift:shift + self.label_width + in_end].values)
                X_cat.append(cat_df[in_end + shift:shift + self.label_width + in_end].values)
        if self.label_width != 1:
            y_df = array(Y_ts)
            cat_df = array(X_cat)
        if nmbr_ts_arrays == 1:
            return [array(X_ts), cat_df], y_df
        else:
            output = [array(x_ts) for x_ts in x_ts_list]
            output.append(cat_df)
            return output, y_df
