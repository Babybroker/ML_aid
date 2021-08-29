"""
The MLaid project is build by Kevin Dankers and released under the licence CC BY-SA. Updates can be downloaded from
the project github https://github.com/Babybroker/ML_aid
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from default_operations import normalize, split_data, make_predictions, prepare_windowed_data  # , denormalize_results


def create_windows(xtrain, ytrain, xtest, ytest, target_col, window_len, shift, batch_size,
                   xval=None, yval=None, target_len=1, reduce_test_stepsize=False):
    def make_windows(xdata, ydata, reduced_stepsize=False):
        """Creates windows"""
        amnt_data_points = len(xdata)
        print('Started making windows')
        print(f'There are {amnt_data_points} datapoints')
        total_win_len = window_len + shift + target_len - 1
        cutoff = amnt_data_points - amnt_data_points % total_win_len
        xdata = xdata[0:cutoff]  # cutoff to create equal parts
        ydata = ydata[0:cutoff]
        x_data = []
        y_data = []
        if reduced_stepsize:
            step_size = shift + target_len - 1
        else:
            step_size = shift
        for i in range(0, len(xdata) - total_win_len, step_size):
            x_data.append(xdata[i:(i + window_len)])
            y_data.append(ydata[i + window_len:i+total_win_len])  # takes the last values of the temp set

        windowed_dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data))
        return windowed_dataset

    train_windows = make_windows(xtrain, ytrain).shuffle(2000).batch(batch_size)
    print('Train set is done')
    test_windows = make_windows(xtest, ytest, reduced_stepsize=reduce_test_stepsize).batch(batch_size)
    print('Test set is done')
    if val_df is not None:
        val_windows = make_windows(xval, yval).batch(batch_size)
        print('Val set is done')
        return [train_windows, test_windows, val_windows]
    else:
        return [train_windows, test_windows]


def create_tensors(xtrain, ytrain, xtest, ytest, batch_size, xval=None, yval=None):
    def feed_input(X, y, shuffle=False):

        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        if shuffle:
            dataset = dataset.shuffle(2000)
        dataset = dataset.batch(batch_size)
        return dataset

    training_data = feed_input(xtrain, ytrain, shuffle=True)
    testing_data = feed_input(xtest, ytest)
    if xval is not None:
        val_data = feed_input(xval, yval)
        return training_data, testing_data, val_data
    else:
        return training_data, testing_data


def prepare_data(dataframe, target_column, batch_size, use_validation=False,
                 normalization_method='standard_score', not_norm_cols=None):
    splitted_data = split_data(dataframe, target_col=target_column, batch_size=batch_size,
                               use_validation=use_validation)
    if use_validation:
        xnormed_train, xnormed_test, xnormed_val, scaler = normalize(splitted_data[0], splitted_data[2],
                                                                     splitted_data[4],
                                                                     normalization_method=normalization_method,
                                                                     not_norm_cols=not_norm_cols
                                                                     )
        training_data, testing_data, val_data = create_tensors(xtrain=xnormed_train,
                                                               ytrain=splitted_data[1],
                                                               xtest=xnormed_test,
                                                               ytest=splitted_data[3],
                                                               xval=xnormed_val,
                                                               yval=splitted_data[5],
                                                               batch_size=batch_size,
                                                               )
        return training_data, testing_data, val_data, scaler


def prepare_timeseries_data(dataframe, target_column, window_len,
                            batch_size=32, shift=1, target_len=1,
                            reduce_stepsize=True, use_validation=False,
                            normalization_method='standard_score', not_norm_cols=None,
                            window_normalization=True, normalization_window_len=2500):
    print('------------------------')
    print('Data preperation started')
    prepare_windowed_data(dataframe,
                          target_col=target_column,
                          window_len=window_len,
                          target_len=target_len,
                          shift=shift,
                          batch_size=batch_size,
                          use_validation=use_validation,
                          reduced_stepsize=reduce_stepsize
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


def get_predictions(model, test_data, target_col, index, is_binary=False, is_timeseries=False, time_ahead=1):
    test_values = np.array([0 for i in range(time_ahead)])
    for features, label in test_data.unbatch():
        if time_ahead == 1:
            test_values = np.vstack([test_values, label.numpy()])
        else:
            test_values = np.concatenate([test_values, label.numpy()])
    test_values = np.delete(test_values, 0)

    if is_binary:
        result = (pd.DataFrame(round(pred_denormed), columns=['prediction'])
                  .join(pd.DataFrame(test_values_denormed, columns=['actual']))
                  )
    else:
        result = make_predictions(model=model,
                                  x_test=test_data,
                                  y_test=test_values,
                                  target_col=target_col,
                                  is_tf=True,
                                  index=index

                                  )
    return result
