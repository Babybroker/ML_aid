"""
The MLaid project is build by Kevin Dankers and released under the licence CC BY-SA. Updates can be downloaded from
the project github https://github.com/Babybroker/ML_aid
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from default_operations import normalize, split_data, denormalize_results


def create_windows(train_df, test_df, target_col, window_len, shift, batch_size,
                   val_df=None, target_len=1, reduce_test_stepsize=False):
    def make_windows(data, reduced_stepsize=False):
        """Creates windows"""
        amnt_data_points = len(data)
        print('Started making windows')
        print(f'There are {amnt_data_points} datapoints')
        total_win_len = window_len + shift + target_len - 1
        data = data[
            data.index < amnt_data_points - amnt_data_points % total_win_len]  # cutoff to create equal parts
        x_data = []
        y_data = []
        if reduced_stepsize:
            step_size = shift + target_len - 1
        else:
            step_size = shift
        for i in range(0, len(data) - total_win_len, step_size):
            temp_set = data[i:(i + total_win_len)].reset_index(drop=True)
            input_set = temp_set[:window_len]
            x_data.append(input_set)
            if target_len == 1:
                y_data.append(
                    [temp_set.loc[window_len + shift - 1, target_col]])  # takes the last values of the temp set
            else:
                y_set = temp_set.loc[window_len + shift - 1:, target_col]
                if len(y_set) != target_len:
                    raise print('The y-dataset is not the expected length')
                y_data.append(y_set.to_list())  # takes the last values of the temp set
        x_data = np.array(x_data)
        y_data = np.array(y_data)
        windowed_dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data))
        return windowed_dataset

    train_windows = make_windows(train_df).shuffle(2000).batch(batch_size)
    print('Train set is done')
    test_windows = make_windows(test_df, reduced_stepsize=reduce_test_stepsize).batch(batch_size)
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


def prepare_data(dataframe, batch_size, target_column,
                 use_validation=False, normalization_method='standard_score', not_norm_cols=None):
    if use_validation:
        train, test, val = split_data(dataframe, batch_size, use_validation)
        normed_train, normed_test, normed_val, denormalize_vals = normalize(train, test, val,
                                                                            normalization_method=normalization_method,
                                                                            not_norm_cols=not_norm_cols
                                                                            )
        print(denormalize_vals)
        training_data, testing_data, val_data = create_tensors(normed_train, normed_test,
                                                               val_df=normed_val,
                                                               batch_size=batch_size,
                                                               target_col=target_column
                                                               )
        return training_data, testing_data, val_data, denormalize_vals


def prepare_timeseries_data(dataframe, target_column, window_len,
                            batch_size=32, shift=1, target_len=1,
                            reduce_test_stepsize=False, use_validation=False,
                            normalization_method='standard_score', not_norm_cols=None,
                            window_normalization=True, normalization_window_len=2500):
    print('------------------------')
    print('Data preperation started')
    if use_validation:
        train, test, val = split_data(dataframe, batch_size, is_timeseries=True, use_validation=True)
        normed_data = normalize(train, test, val, normalization_method, not_norm_cols,
                                window_normalization=window_normalization,
                                normalization_window_len=normalization_window_len
                                )
        windowed_data = create_windows(normed_data[0], normed_data[1], target_column,
                                       window_len=window_len,
                                       shift=shift,
                                       target_len=target_len,
                                       batch_size=batch_size,
                                       val_df=normed_data[2],
                                       reduce_test_stepsize=reduce_test_stepsize
                                       )

    else:
        train, test = split_data(dataframe, batch_size, is_timeseries=True, use_validation=False)
        normed_data = normalize(train, test, normalization_method, not_norm_cols,
                                window_normalization=window_normalization,
                                normalization_window_len=normalization_window_len
                                )
        windowed_data = create_windows(normed_data[0], normed_data[1], target_column,
                                       window_len=window_len,
                                       shift=shift,
                                       target_len=target_len,
                                       batch_size=batch_size,
                                       reduce_test_stepsize=reduce_test_stepsize
                                       )
    denorm_vals = normed_data[-1]
    output = windowed_data
    output.append(denorm_vals)
    return windowed_data


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


def plot_loss(history, model_name, loss, metrics):
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
        metric = metrics[i - 1]
        axs[i].plot(history.history[metric])
        axs[i].plot(history.history[f"val_{metric}"])
        axs[i].set_title(f"model {metric}")
        axs[i].set_ylabel('accuracy', fontsize="large")
        axs[i].set_xlabel("epoch", fontsize="large")
        axs[i].legend(["train", "val"], loc="best")
    plt.show()
    plt.close()


def get_predictions(model, test_data, target_col, denormalize_vals, is_binary=False, is_timeseries=False):
    predictions = model.predict(test_data)

    test_values = np.array([])
    for features, label in test_data.unbatch():
        test_values = np.concatenate([test_values, label.numpy()])

    test_values_denormed = denormalize_results(test_values, target_col, denormalize_vals)
    pred_denormed = denormalize_results(predictions, target_col, denormalize_vals)
    if is_binary:
        result = (pd.DataFrame(round(pred_denormed), columns=['prediction'])
                  .join(pd.DataFrame(test_values_denormed, columns=['actual']))
                  )
    else:
        result = (pd.DataFrame(pred_denormed, columns=['prediction'])
                  .join(pd.DataFrame(test_values_denormed, columns=['actual']))
                  )
    return result
