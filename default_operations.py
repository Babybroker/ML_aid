import numpy as np


def split_data(dataframe, batch_size=None, use_validation=False, is_timeseries=False):
    def split_non_timeseries():
        train_df = dataframe.iloc[idx[:train_length]].reset_index(drop=True)
        if use_validation:
            test_length = int(round(n * 0.2, 0))
            train_test_length = test_length + train_length

            test_df = dataframe.iloc[idx[train_length:train_test_length]].reset_index(drop=True)
            val_df = dataframe.iloc[idx[train_test_length:]].reset_index(drop=True)
            return [train_df, test_df, val_df]
        else:
            test_df = dataframe.iloc[idx[train_length:]].reset_index(drop=True)
            return [train_df, test_df]

    def split_timeseries():
        train_df = dataframe[:train_length].reset_index(drop=True)
        if use_validation:
            val_df = dataframe[int(n * 0.7):int(n * 0.9)].reset_index(drop=True)
            test_df = dataframe[int(n * 0.9):].reset_index(drop=True)
            return [train_df, test_df, val_df]
        else:
            test_df = dataframe[train_length:].reset_index(drop=True)
            return [train_df, test_df]

    n = len(dataframe)
    idx = np.random.RandomState(seed=1).permutation(n)  # create list with random indexes

    train_length = int(round(n * 0.7, 0))
    if batch_size is not None:
        train_length = train_length - train_length % batch_size
    if is_timeseries:
        print('Data has been split')
        return split_timeseries()
    else:
        print('Data has been split')
        return split_non_timeseries()


def normalize(train_df, test_df, val_df=None, normalization_method='unity_based', not_norm_cols=None):
    """norm_param1: either the mean or the minimum value of the training set.
     norm_param2 is either the std or the maximum value of the training set"""

    def unity_normalize(dataframe):
        norm_df = dataframe.copy()
        if not_norm_cols:
            norm_cols = np.setdiff1d(norm_df.columns, not_norm_cols)
            norm_df.loc[:, norm_cols] = (norm_df.loc[:, norm_cols] - mean[norm_cols]) / std[norm_cols]
        else:
            norm_df = (norm_df - mean) / std
        return norm_df

    def x_normal_normalize(dataframe):
        norm_df = dataframe.copy()
        if not_norm_cols:
            norm_cols = np.setdiff1d(norm_df.columns, not_norm_cols)
            norm_df.loc[norm_cols] = (norm_df.loc[norm_cols] - min_val[norm_cols]) / (
                    max_val[norm_cols] - min_val[norm_cols])
        else:
            norm_df = (norm_df - min_val) / (max_val - min_val)
        return norm_df

    normed_train, normed_test, normed_val, denormalize_vals = None, None, None, None
    normalization_possibilities = ['unity_based', 'x_normal']
    if normalization_method not in normalization_possibilities:
        raise print(f'That is not a recognized normalization, possibilities are: {normalization_possibilities}')
    if normalization_method == 'unity_based':
        mean = train_df.mean()
        std = train_df.std()
        normed_train = unity_normalize(train_df)
        normed_test = unity_normalize(test_df)
        if val_df is not None:
            normed_val = unity_normalize(val_df)
        denormalize_vals = mean.to_frame('mean').join(std.to_frame('std'))

    if normalization_method == 'x_normal':
        min_val = train_df.min()
        max_val = train_df.max()
        normed_train = x_normal_normalize(train_df)
        normed_test = x_normal_normalize(test_df)
        if val_df is not None:
            normed_val = x_normal_normalize(val_df)
        denormalize_vals = min_val.to_frame('min').join(max_val.to_frame('max'))
    if val_df is None:
        return [normed_train, normed_test, denormalize_vals]
    else:
        return [normed_train, normed_test, denormalize_vals, normed_val]


def split_and_normalize(dataframe, batch_size=None, use_validation=False, is_timeseries=False,
                        normalization_method='unity_based', not_norm_cols=None):
    splitted_data = split_data(dataframe, batch_size, use_validation, is_timeseries)
    if not use_validation:
        splitted_data.append(None)
    normalized_data = normalize(train_df=splitted_data[0],
                                test_df=splitted_data[1],
                                val_df=splitted_data[2],
                                normalization_method=normalization_method,
                                not_norm_cols=not_norm_cols)
    return normalized_data


def denormalize_results(denorm_series, target_column, denorm_values, normalization_method='unity_based'):
    def unity_denormalize():
        denormed = denorm_series * denorm_values.loc[target_column, 'std'] + denorm_values.loc[target_column, 'mean']
        return denormed

    def x_denormalize():
        denormed = denorm_series * (denorm_values.loc[target_column, 'max'] - denorm_values.loc[target_column, 'min']) + \
                   denorm_values.loc[target_column, 'min']
        return denormed

    denormed = None
    normalization_possibilities = ['unity_based', 'x_normal']
    if normalization_method not in normalization_possibilities:
        raise print(f'That is not a recognized normalization, possibilities are: {normalization_possibilities}')
    if normalization_method == 'unity_based':
        denormed = unity_denormalize()
    if normalization_method == 'x_normal':
        denormed = x_denormalize()
    return denormed