"""
The MLaid project is build by Kevin Dankers and released under the licence CC BY-SA. Updates can be downloaded from
the project github https://github.com/Babybroker/ML_aid
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def split_data(dataframe, batch_size=None, use_validation=False, is_timeseries=False):
    def split_non_timeseries():
        train_df = dataframe.iloc[idx[:train_length]]
        dfs = [train_df]
        if use_validation:
            test_length = int(round(n * 0.2, 0))
            train_test_length = test_length + train_length

            test_df = dataframe.iloc[idx[train_length:train_test_length]]
            val_df = dataframe.iloc[idx[train_test_length:]]
            dfs.extend([test_df, val_df])
        else:
            test_df = dataframe.iloc[idx[train_length:]]
            dfs.extend([test_df])
        print('Data has been split')
        return dfs

    def split_timeseries():
        train_df = dataframe[:train_length].reset_index(drop=True)
        dfs = [train_df]
        if use_validation:
            val_df = dataframe[int(n * 0.7):int(n * 0.9)].reset_index(drop=True)
            test_df = dataframe[int(n * 0.9):].reset_index(drop=True)
            dfs.extend([test_df, val_df])
        else:
            test_df = dataframe[train_length:].reset_index(drop=True)
            dfs.extend([train_df, test_df])
        print('Data has been split')
        return dfs

    n = len(dataframe)
    idx = np.random.RandomState().permutation(n)  # create list with random indexes

    train_length = int(round(n * 0.7, 0))
    if batch_size is not None:
        train_length = train_length - train_length % batch_size
    if is_timeseries:
        return split_timeseries()
    else:
        return split_non_timeseries()


def normalize(train_df, test_df, val_df=None, normalization_method='standard_score', not_norm_cols=None,
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
        normed_train, normed_test, normed_val, denormalize_vals = None, None, None, None

        if normalization_method == 'standard_score':
            mean = train_df.mean()
            std = train_df.std()
            normed_train = standard_score_normalize(train_df, mean, std)
            normed_test = standard_score_normalize(test_df, mean, std)
            if val_df is not None:
                normed_val = standard_score_normalize(val_df, mean, std)
            denormalize_vals = mean.to_frame('mean').join(std.to_frame('std'))

        if normalization_method == 'unity_based':
            min_val = train_df.min()
            max_val = train_df.max()
            normed_train = unity_based_normalize(train_df, min_val, max_val)
            normed_test = unity_based_normalize(test_df, min_val, max_val)
            if val_df is not None:
                normed_val = unity_based_normalize(val_df, min_val, max_val)
            denormalize_vals = min_val.to_frame('min').join(max_val.to_frame('max'))

        if val_df is None:
            return [normed_train, normed_test, denormalize_vals]
        else:
            return [normed_train, normed_test, normed_val, denormalize_vals]

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


def split_and_normalize(dataframe, batch_size=None, use_validation=False, is_timeseries=False,
                        normalization_method='standard_score', not_norm_cols=None,
                        window_normalization=False, normalization_window_len=2500):
    splitted_data = split_data(dataframe, batch_size, use_validation, is_timeseries)
    if not use_validation:
        splitted_data.append(None)

    normalized_data = normalize(train_df=splitted_data[0], test_df=splitted_data[1], val_df=splitted_data[2],
                                normalization_method=normalization_method, not_norm_cols=not_norm_cols,
                                window_normalization=window_normalization,
                                normalization_window_len=normalization_window_len
                                )
    return normalized_data


def denormalize_results(denorm_series, target_column, denorm_values, normalization_method='standard_score'):
    def standard_score_denormalize():
        denormed = denorm_series * denorm_values.loc[target_column, 'std'] + denorm_values.loc[target_column, 'mean']
        return denormed

    def unity_denormalize():
        denormed = denorm_series * (denorm_values.loc[target_column, 'max'] - denorm_values.loc[target_column, 'min']) + \
                   denorm_values.loc[target_column, 'min']
        return denormed

    denormed = None
    normalization_possibilities = ['unity_based', 'standard_score']
    if normalization_method not in normalization_possibilities:
        raise print(f'That is not a recognized normalization, possibilities are: {normalization_possibilities}')
    if normalization_method == 'standard_score':
        denormed = standard_score_denormalize()
    if normalization_method == 'unity_based':
        denormed = unity_denormalize()
    return denormed


def make_predictions(model, x_test, y_test, target_col, test_index, denorm_values=None):

    # Make predictions
    predictions = pd.DataFrame(model.predict(x_test), columns=['prediction'])

    result_df = predictions.join(pd.DataFrame(y_test, columns=['actual']))
    if denorm_values is not None:
        result_df = denormalize_results(result_df, target_col, denorm_values)
    result_df = result_df.set_index(test_index).sort_index()  # set the index from the test data as index,
    # as those contain the timestamps

    # Print the actual scores
    print('XGBR Results')
    print('MSE actual values:', mean_squared_error(result_df.actual, result_df.prediction))
    print('MAE actual values:', mean_absolute_error(result_df.actual, result_df.prediction))
    print('R2 score:', r2_score(result_df.actual, result_df.prediction))
    return result_df