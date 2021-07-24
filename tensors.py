import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def split_data(dataframe, batch_size, use_validation=False, is_timeseries=False):
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


def create_windows(train_df, test_df, target_col, window_len, shift, batch_size, val_df=None, target_len=1):
    def make_windows(data):
        """Creates windows"""
        amnt_data_points = len(data)
        print('Started making windows')
        print(f'There are {amnt_data_points} datapoints')
        data = data[
            data.index < amnt_data_points - amnt_data_points % window_len + shift]  # cutoff to create equal parts
        x_data = []
        y_data = []
        for i in range(0, len(data) - window_len, shift):
            temp_set = data[i:(i + window_len + shift)].reset_index(drop=True)
            input_set = temp_set[:window_len]
            x_data.append(input_set)
            if target_len != 1:
                y_data.append(temp_set.loc[window_len:window_len + shift + target_len - 1,
                              target_col])  # takes the last values of the temp set
            else:
                y_data.append(
                    [temp_set.loc[window_len + shift - 1, target_col]])  # takes the last values of the temp set
        x_data = np.array(x_data)
        windowed_dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data))
        return windowed_dataset

    train_windows = make_windows(train_df).shuffle(2000).batch(batch_size)
    print('Train set is done')
    test_windows = make_windows(test_df).batch(batch_size)
    print('Test set is done')
    if val_df is not None:
        val_windows = make_windows(val_df).batch(batch_size)
        print('Val set is done')
        return [train_windows, test_windows, val_windows]
    else:
        return [train_windows, test_windows]


def create_tensors(train_df, test_df, target_col, batch_size, val_df=None):
    def feed_input(dataframe, shuffle=False):
        y_data = dataframe[target_col]
        x_data = dataframe.drop(columns=[target_col])
        dataset = tf.data.Dataset.from_tensor_slices((x_data.values, y_data))
        if shuffle:
            dataset = dataset.shuffle(2000)
        dataset = dataset.batch(batch_size)
        return dataset

    training_data = feed_input(train_df, shuffle=True)
    testing_data = feed_input(test_df)
    if val_df is not None:
        val_data = feed_input(val_df)
        return training_data, testing_data, val_data
    else:
        return training_data, testing_data


def denormalize_results(denorm_series, target_column, denorm_values, normalization_method='unity_based'):
    def unity_denormalize():
        denormed = denorm_series * denorm_values.loc[target_column, 'std'] + denorm_values.loc[target_column, 'mean']
        return denormed

    def x_denormalize():
        denormed = denorm_series * (denorm_values.loc[target_column, 'max'] - denorm_values.loc[target_column, 'min']) + \
                   denorm_values.loc[target_column, 'min']
        return denormed

    normalization_possibilities = ['unity_based', 'x_normal']
    if normalization_method not in normalization_possibilities:
        raise print(f'That is not a recognized normalization, possibilities are: {normalization_possibilities}')
    if normalization_method == 'unity_based':
        denormed = unity_denormalize()
    if normalization_method == 'x_normal':
        denormed = x_denormalize()
    return denormed


def prepare_data(dataframe, batch_size, target_column,
                 use_validation=False, normalization_method='unity_based', not_norm_cols=None):
    if use_validation:
        train, test, val = split_data(dataframe, batch_size, use_validation)
        normed_train, normed_test, denormalize_vals, normed_val = normalize(train, test, val,
                                                                            normalization_method, not_norm_cols)
        training_data, testing_data, val_data = create_tensors(normed_train, normed_test,
                                                               val_df=normed_val,
                                                               batch_size=batch_size,
                                                               target_col=target_column
                                                               )
        return training_data, testing_data, val_data, denormalize_vals


def prepare_timeseries_data(dataframe, batch_size, target_column, window_len, shift,
                            target_len=1,
                            use_validation=False, normalization_method='unity_based', not_norm_cols=None):
    print('------------------------')
    print('Data preperation started')
    if use_validation:
        train, test, val = split_data(dataframe, batch_size, is_timeseries=True, use_validation=True)
        normed_train, normed_test, denormalize_vals, normed_val = normalize(train, test, val,
                                                                            normalization_method, not_norm_cols)
        training_data, testing_data, val_data = create_windows(normed_train, normed_test, target_column,
                                                               window_len=window_len,
                                                               shift=shift,
                                                               target_len=target_len,
                                                               batch_size=batch_size,
                                                               val_df=normed_val)

        return training_data, testing_data, val_data, denormalize_vals


def compile_and_fit(model, model_name, train_dataset, val_dataset, monitor, metrics, max_epochs, loss, patience=15):
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            f"best_model_{model_name}.h5", save_best_only=True, monitor=monitor
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor, factor=0.5, patience=patience, min_lr=0.0001
        ),
        tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=patience, verbose=1, mode='min'),
    ]

    model.compile(loss=loss,
                  optimizer=tf.optimizers.Adam(),
                  metrics=[metrics]
                  )

    history = model.fit(train_dataset,
                        epochs=max_epochs,
                        validation_data=val_dataset,
                        callbacks=callbacks,
                        )
    model_final = tf.keras.models.load_model(f"best_model_{model_name}.h5")
    return model_final, history


def plot_loss(history, model_name, loss, metrics, metric_names):
    plt.rcParams.update({'figure.figsize': (13, 7), 'figure.dpi': 240})  # Set figure details
    total_length = 1 + len(metrics)
    fig, axs = plt.subplots(total_length, 1, constrained_layout=True)
    fig.suptitle(model_name, fontsize=16)

    axs[0].plot(history.history[loss])
    axs[0].plot(history.history[f"val_{loss}"])
    axs[0].set_title("model loss")
    axs[0].set_ylabel('loss', fontsize="large")
    axs[0].set_xlabel("epoch", fontsize="large")
    axs[0].legend(["train", "val"], loc="best")
    for i in range(1, total_length):
        metric = metric_names[i - 1]
        axs[i].plot(history.history[metric])
        axs[i].plot(history.history[f"val_{metric}"])
        axs[i].set_title(f"model {metric}")
        axs[i].set_ylabel('accuracy', fontsize="large")
        axs[i].set_xlabel("epoch", fontsize="large")
        axs[i].legend(["train", "val"], loc="best")
    plt.show()
    plt.close()
