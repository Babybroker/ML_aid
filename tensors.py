import tensorflow as tf
import numpy as np


def feed_input(dataframe, target_col, batch_size, shuffle=True):
    y_data = dataframe[target_col]
    x_data = dataframe.drop(columns=target_col)
    dataset = tf.data.Dataset.from_tensor_slices((x_data.values, y_data))
    if shuffle:
        dataset = dataset.shuffle(2000)
    dataset = dataset.batch(batch_size)
    return dataset


def normalize(dataframe, norm_param1, norm_param2, normalization_method='unity_based', not_norm_cols=None):
    """norm_param1: either the mean or the minimum value of the training set.
     norm_param2 is either the std or the maximum value of the training set"""
    def unity_normalize():
        norm_df = dataframe.copy()
        if not_norm_cols:
            norm_cols = np.setdiff1d(norm_df.columns, not_norm_cols)
            norm_df.loc[:, norm_cols] = (norm_df.loc[:, norm_cols] - norm_param1[norm_cols]) / norm_param2[norm_cols]
        else:
            norm_df = (norm_df - norm_param1) / norm_param2
        return norm_df

    def x_normal_normalize():
        norm_df = dataframe.copy()
        if not_norm_cols:
            norm_cols = np.setdiff1d(norm_df.columns, not_norm_cols)
            norm_df.loc[:, norm_cols] = (norm_df.loc[:, norm_cols] - norm_param1[norm_cols]) / (
                    norm_param2[norm_cols] - norm_param1[norm_cols])
        else:
            norm_df = (norm_df - norm_param1) / (norm_param2 - norm_param1)
        return norm_df

    normalization_possibilities = {'unity_based': unity_normalize(),
                                  'x_normal': x_normal_normalize()}
    if normalization_method not in normalization_possibilities:
        raise print(f'That is not a recognized normalization, possibilities are: {normalization_possibilities}')
    normed_df = normalization_possibilities[normalization_method]
    return normed_df


