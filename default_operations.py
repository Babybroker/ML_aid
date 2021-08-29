"""
The MLaid project is build by Kevin Dankers and released under the licence CC BY-SA. Updates can be downloaded from
the project github https://github.com/Babybroker/ML_aid
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def split_data(dataframe, target_col, batch_size=None, use_validation=False, is_timeseries=False, split_xy=True):
    n = len(dataframe)
    train_size = 0.8
    if use_validation:
        train_size = 0.7
    train_length = int(round(n * train_size, 0))
    if batch_size is not None:
        train_length = train_length - train_length % batch_size

    shuffle = not is_timeseries
    if split_xy:
        ydata = dataframe.pop(target_col)
        xdata = dataframe
        x_train, x_test, y_train, y_test = train_test_split(xdata, ydata, train_size=train_size, shuffle=shuffle)
        dfs = [x_train, y_train]
        if use_validation:
            x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, train_size=0.7, shuffle=shuffle)
            dfs.extend([x_test, y_test, x_val, y_val])
        else:
            dfs.extend([x_test, y_test])
    else:
        train, test = train_test_split(dataframe, train_size=train_size, shuffle=shuffle)
        dfs = [train]
        if use_validation:
            test, val = train_test_split(test, train_size=0.7, shuffle=shuffle)
            dfs.extend([test, val])
        else:
            dfs.extend([test])
    print('Data has been split')
    return dfs


def normalize(xtrain_df, xtest_df, xval_df=None, normalization_method='standard_score', not_norm_cols=None,
              window_normalization=False, normalization_window_len=2500):
    """norm_param1: either the mean or the minimum value of the training set.
     norm_param2 is either the std or the maximum value of the training set"""

    def standard_score_normalize(dataframe, mean_val, std_val):
        norm_df = dataframe.copy()
        if not_norm_cols:
            norm_cols = np.setdiff1d(norm_df.columns, not_norm_cols)
            norm_df.loc[:, norm_cols] = (norm_df.loc[:, norm_cols] - mean_val[norm_cols]) / std_val[norm_cols]
        else:
            norm_df = (norm_df - mean_val) / std_val
        return norm_df

    def unity_based_normalize(dataframe, min_value, max_value):
        norm_df = dataframe.copy()
        if not_norm_cols:
            norm_cols = np.setdiff1d(norm_df.columns, not_norm_cols)
            norm_df.loc[:, norm_cols] = (norm_df.loc[:, norm_cols] - min_value[norm_cols]) / (
                    max_value[norm_cols] - min_value[norm_cols])
        else:
            norm_df = (norm_df - min_value) / (max_value - min_value)
        return norm_df

    def non_windowed_normalization():
        normed_train, normed_test, normed_val, scaler = None, None, None, None

        if normalization_method == 'standard_score':
            scaler = StandardScaler()
            scaler.fit(xtrain_df)

            normed_train = scaler.transform(xtrain_df)
            normed_test = scaler.transform(xtest_df)
            if xval_df is not None:
                normed_val = scaler.transform(xval_df)

        if normalization_method == 'unity_based':
            min_val = train_df.min()
            max_val = train_df.max()
            normed_train = unity_based_normalize(train_df, min_val, max_val)
            normed_test = unity_based_normalize(test_df, min_val, max_val)
            if xval_df is not None:
                normed_val = unity_based_normalize(val_df, min_val, max_val)
            denormalize_vals = min_val.to_frame('min').join(max_val.to_frame('max'))

        if xval_df is None:
            return [normed_train, normed_test, scaler]
        else:
            return [normed_train, normed_test, normed_val, scaler]

    def windowed_normalization(dataf, is_test=False):
        dataf_len = len(dataf)
        rest = dataf_len % normalization_window_len
        end = dataf_len - rest
        normalized_df = pd.DataFrame([])

        if is_test:
            if normalization_method == 'standard_score':
                mean = dataf.mean()
                std = dataf.std()
                denormalize_vals = mean.to_frame('mean').join(std.to_frame('std'))
                normalized_df = normalized_df.append(standard_score_normalize(dataf, mean, std))
            elif normalization_method == 'unity_based':
                min_val = dataf.min()
                max_val = dataf.max()
                denormalize_vals = min_val.to_frame('min').join(max_val.to_frame('max'))
                normalized_df = normalized_df.append(unity_based_normalize(dataf, min_val, max_val))
            return normalized_df, denormalize_vals
        else:
            for i in range(0, end, normalization_window_len):
                subset = dataf[i:i + normalization_window_len]
                if normalization_method == 'standard_score':
                    mean = subset.mean()
                    std = subset.std()
                    normalized_df = normalized_df.append(standard_score_normalize(subset, mean, std))
                elif normalization_method == 'unity_based':
                    min_val = subset.min()
                    max_val = subset.max()
                    normalized_df = normalized_df.append(unity_based_normalize(subset, min_val, max_val))

            subset = dataf[end:]
            if normalization_method == 'standard_score':
                mean = subset.mean()
                std = subset.std()
                normalized_df = normalized_df.append(standard_score_normalize(subset, mean, std))
            elif normalization_method == 'unity_based':
                min_val = subset.min()
                max_val = subset.max()
                normalized_df = normalized_df.append(unity_based_normalize(subset, min_val, max_val))
            return normalized_df

    normalization_possibilities = ['unity_based', 'standard_score']
    if normalization_method not in normalization_possibilities:
        raise print(f'That is not a recognized normalization, possibilities are: {normalization_possibilities}')

    if window_normalization:
        normed_train = windowed_normalization(train_df)
        normed_test, denormalize_val = windowed_normalization(test_df, is_test=True)
        if val_df is not None:
            normed_val = windowed_normalization(val_df)
            return [normed_train, normed_test, normed_val, denormalize_val]
        else:
            return [normed_train, normed_test, denormalize_val]

    else:
        return non_windowed_normalization()


def split_and_normalize(dataframe, target_col, batch_size=None, use_validation=False, is_timeseries=False,
                        normalization_method='standard_score', not_norm_cols=None,
                        window_normalization=False, normalization_window_len=2500, split_xy=True):
    splitted_data = split_data(dataframe, batch_size=batch_size,
                               use_validation=use_validation,
                               is_timeseries=is_timeseries,
                               target_col=target_col,
                               split_xy=split_xy
                               )
    yvals = [splitted_data[1], splitted_data[3]]
    if use_validation:
        yvals.extend([splitted_data[5]])
    if not use_validation:
        splitted_data.append(None)

    normalized_data = normalize(xtrain_df=splitted_data[0], xtest_df=splitted_data[2], xval_df=splitted_data[4],
                                normalization_method=normalization_method, not_norm_cols=not_norm_cols,
                                window_normalization=window_normalization,
                                normalization_window_len=normalization_window_len,
                                )
    yvals.extend(normalized_data)
    return yvals


def make_predictions(model, x_test, y_test, target_col, is_tf=False, index=None):
    # Make predictions
    predictions = pd.DataFrame(model.predict(x_test), columns=['prediction'])
    if is_tf:
        y_test = pd.DataFrame(y_test, columns=[target_col])
    else:
        try:
            y_test = y_test.to_frame().reset_index()
        except:
            y_test = pd.DataFrame(y_test)

    if index is None:
        try:
            index = x_test.index
        except AttributeError:
            index = y_test.index
    result_df = predictions.join(y_test).set_index(index).sort_index()

    # Print the actual scores
    print('XGBR Results')
    print('MSE actual values:', mean_squared_error(result_df[target_col], result_df.prediction))
    print('MAE actual values:', mean_absolute_error(result_df[target_col], result_df.prediction))
    print('R2 score:', r2_score(result_df[target_col], result_df.prediction))
    return result_df


def plot_timesteps(result_df, target_col, amnt_of_plots, ylabel, title=None):
    import matplotlib.pyplot as plt
    from random import randint
    fig2, ax2 = plt.subplots(amnt_of_plots)
    fig2.suptitle(title)
    for i in range(amnt_of_plots):
        following_timesteps = 24
        n = randint(0, len(result_df) - following_timesteps)
        data_slice = result_df[n:n + following_timesteps]
        ax2[i].scatter(data_slice.index, data_slice['prediction'], linewidths=0.5, marker='x')
        ax2[i].scatter(data_slice.index, data_slice[target_col], s=20, facecolors='none', edgecolors='black')
        ax2[i].set_ylabel(ylabel)
    fig2.legend(['Prediction', 'Actual'], loc='center right')
    plt.tight_layout()
    plt.show()


def make_windows(xdata, ydata, window_len, shift, target_len, reduced_stepsize=False, data_in_both_windows=True):
    """Creates windows"""
    amnt_data_points = len(xdata)
    print('Started making windows')
    print(f'There are {amnt_data_points} datapoints')

    total_win_len = window_len + shift + target_len - 1
    cutoff = amnt_data_points - amnt_data_points % total_win_len

    xdata = xdata[0:cutoff]  # cutoff to create equal parts
    ydata = ydata[0:cutoff]
    if data_in_both_windows:
        ydataset = ydata.to_frame()
        total = pd.DataFrame(xdata, index=ydataset.index).join(ydataset)

    x_data = []
    y_data = []

    if reduced_stepsize:
        step_size = shift + target_len - 1
    else:
        step_size = shift
    for i in range(0, len(xdata) - total_win_len, step_size):
        if data_in_both_windows:
            x_data.append(total[i:(i + window_len)])
            y_data.append(total[i + window_len:i + total_win_len])  # takes the last values of the temp set
        else:
            x_data.append(xdata[i:(i + window_len)])
            y_data.append(ydata[i + window_len:i + total_win_len])  # takes the last values of the temp set
    return [x_data, y_data]


def prepare_windowed_data(dataframe, target_col, window_len, target_len,
                          shift=1, data_in_both_windows=True,
                          batch_size=None, use_validation=False, reduced_stepsize=True,
                          normalization_method='standard_score', not_norm_cols=None,
                          window_normalization=False, normalization_window_len=2500
                          ):
    splitted_data = split_and_normalize(dataframe, target_col, batch_size=batch_size,
                                        use_validation=use_validation,
                                        is_timeseries=True,
                                        normalization_method=normalization_method,
                                        not_norm_cols=not_norm_cols,
                                        window_normalization=window_normalization,
                                        normalization_window_len=normalization_window_len
                                        )
    windowed_data = []
    end = 2
    if use_validation:
        end = 3
    for i in range(end):
        data = (make_windows(ydata=splitted_data[i],
                             xdata=splitted_data[i + end],
                             window_len=window_len,
                             shift=shift,
                             target_len=target_len,
                             reduced_stepsize=reduced_stepsize,
                             data_in_both_windows=data_in_both_windows
                             )
                )
        windowed_data.extend(data)
    return windowed_data, splitted_data[-1]
