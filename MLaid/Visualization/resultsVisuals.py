import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from scipy.stats import gaussian_kde
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import MLaid.Visualization.functions as functions


class ResultsVisualizer:
    def __init__(
            self,
            result_df: pd.DataFrame,
            target_col: str,
            predict_col: str,
            folder_name: str = None,
            timestamp: str = None,

    ):
        self.target_col = target_col
        self.predict_col = predict_col
        self.folder_name = folder_name
        self.timestamp = timestamp
        self.result_df = result_df

    def plot_result_pred_heatmap(self, save_visual=False, extra_title_addition=None):
        f, ax = plt.subplots(figsize=(8, 8))

        max_val = max(self.result_df[self.target_col].max(), self.result_df[self.predict_col].max()) * 1.05
        min_val = min(self.result_df[self.target_col].min(), self.result_df[self.predict_col].min()) * 1.05
        ax.set(xlim=(min_val, max_val), ylim=(min_val, max_val))
        ax1 = sns.jointplot(data=self.result_df, x=self.target_col, y=self.predict_col)
        ax1.ax_joint.cla()
        plt.sca(ax1.ax_joint)
        plt.hist2d(self.result_df[self.target_col], self.result_df[self.predict_col], bins=(100, 100), cmap=cm.jet)
        plt.xlabel('Prediction')
        plt.ylabel('Actual value')
        title = f'Prediction vs actual value for the unseen set'
        if extra_title_addition is not None:
            title = title + ' ' + extra_title_addition
        plt.title(title)
        functions.save_func(save_visual=save_visual, timestamp=self.timestamp, folder_name=self.folder_name,
                  filename='result_heatmap')
        plt.show()

    def plot_scatter(self, extra_title_addition=None, custom_xlabel=None, custom_ylabel=None, hue=None,
                     custom_palette=None, remove_outliers=False, save_visual=False, custom_title=None):
        def add_identity(axes, *line_args, **line_kwargs):
            identity, = axes.plot([], [], *line_args, **line_kwargs)

            def callback(axes):
                low_x, high_x = axes.get_xlim()
                low_y, high_y = axes.get_ylim()
                low = max(low_x, low_y)
                high = min(high_x, high_y)
                identity.set_data([low, high], [low, high])

            callback(axes)
            axes.callbacks.connect('xlim_changed', callback)
            axes.callbacks.connect('ylim_changed', callback)
            return axes

        if remove_outliers:
            resultdf = self.result_df.query('z_score < 3')
        mae = round(mean_absolute_error(self.result_df[self.target_col], self.result_df[self.predict_col]), 2)
        r2 = round(r2_score(self.result_df[self.target_col], self.result_df[self.predict_col]), 3)
        rmse = round(np.sqrt(mean_squared_error(self.result_df[self.target_col], self.result_df[self.predict_col])), 2)
        test_mean = round(self.result_df[self.target_col].mean())

        textstr = '\n'.join((
            f"MAE: {mae}",
            f'R2-score: {r2}',
            f'RMSE: {rmse}',
            f'Mean of test: {test_mean}'
        ))

        title = f'Prediction vs actual value for the unseen set' if custom_title is None else custom_title
        if extra_title_addition is not None:
            title = title + ' ' + extra_title_addition
        if custom_palette is None:
            custom_palette = "tab10"

        max_val = max(self.result_df[self.target_col].max(), self.result_df[self.predict_col].max()) * 1.05
        min_val = min(self.result_df[self.target_col].min(), self.result_df[self.predict_col].min()) * 1.05
        f, ax = plt.subplots(figsize=(8, 8))
        f = sns.scatterplot(data=self.result_df, x=self.target_col, y=self.predict_col, s=2, alpha=0.5, hue=hue,
                            palette=custom_palette)

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=7, verticalalignment='top', bbox=props)
        f.set(xlim=(min_val, max_val), ylim=(min_val, max_val))
        ax.set_aspect('equal', adjustable='box')
        add_identity(ax, c='0.35', ls='--', linewidth=1)
        if custom_xlabel is not None:
            ax.set(xlabel=custom_xlabel)
        if custom_ylabel is not None:
            ax.set(ylabel=custom_ylabel)
        plt.title(title)
        functions.save_func(save_visual=save_visual, timestamp=self.timestamp, folder_name=self.folder_name, filename='scatter')
        plt.show()

    def print_scores(self):
        if self.predict_col is None:
            predict_col = [col for col in self.result_df if 'prediction' in col]
            print(f"{predict_col[0].replace('prediction_', '')} Results")

        print('Mean of test data:', self.result_df[self.target_col].mean())
        print('MSE actual values:', mean_squared_error(self.result_df[self.target_col], self.result_df[predict_col]))
        print('MAE actual values:', mean_absolute_error(self.result_df[self.target_col], self.result_df[predict_col]))
        print('R2 score:', r2_score(self.result_df[self.target_col], self.result_df[predict_col]))
        print('\n')

    def plot_time_steps(self, amnt_of_plots, ylabel=None, title=None,
                        timesteps=24, x_col=None, save_visual=False):
        from random import randint, seed
        seed(100)

        if self.predict_col is None:
            predict_col = [col for col in self.result_df if 'prediction' in col]
        fig2, ax2 = plt.subplots(amnt_of_plots, figsize=(9, 12))
        fig2.suptitle(title)
        for i in range(amnt_of_plots):
            n = randint(0, int((len(self.result_df) - timesteps) / timesteps))
            data_slice = self.result_df.iloc[n * timesteps:n * timesteps + timesteps]
            if x_col is None:
                x_vals = data_slice.index
            else:
                x_vals = data_slice[x_col]
            ax2[i].plot(x_vals, data_slice[self.target_col])
            ax2[i].scatter(x_vals, data_slice[self.target_col], edgecolors='k', label='Actual', c='#2ca02c', s=64)
            ax2[i].scatter(x_vals, data_slice[self.predict_col], marker='X', edgecolors='k', label='Prediction',
                           c='#ff7f0e', s=64)
            if ylabel is not None:
                ax2[i].set_ylabel(ylabel)
            if i == 0:
                plt.legend(['Actual', 'Prediction'])
        plt.tight_layout()
        functions.save_func(save_visual=save_visual, timestamp=self.timestamp, folder_name=self.folder_name, filename='timestep')
        plt.show()

    def plot_residuals(self, custom_ylabel=None, extra_title_addition=None, train_set=False,
                       save_visual=False, hue=None):
        """
        Plot the residuals of the model
        :param custom_ylabel: Str
        :param extra_title_addition: Str
        :param train_set: Bool if the residuals are from the train set
        :return:
        """
        residuals = self.result_df[self.predict_col] - self.result_df[self.target_col]
        f, ax = plt.subplots()
        f = sns.scatterplot(data=self.result_df, x=self.target_col, y=residuals, s=2, alpha=0.5, hue=hue)
        ax.axhline(0, ls='--', color='g', linewidth=1)
        ax.set(xlabel='Actual Value')
        if custom_ylabel is not None:
            ax.set(ylabel=custom_ylabel)
        else:
            ax.set(ylabel='Prediction - Actual value')
        if train_set:
            key_word = 'train'
        else:
            key_word = 'unseen test'
        title = f'Residuals of the {key_word} set'
        if extra_title_addition is not None:
            title = title + ' ' + extra_title_addition
        plt.title(title)
        functions.save_func(save_visual=save_visual, timestamp=self.timestamp, folder_name=self.folder_name, filename='residuals')
        plt.show()

    def plot_residual_dist(self, capacity_col=None, save_visual=False):
        """
        Plots the distribution of the residuals
        :param capacity_col: Str
        :return:
        """
        if capacity_col is None:
            residuals = self.result_df[self.predict_col] - self.result_df[self.target_col]
        else:
            residuals = (self.result_df[self.predict_col] - self.result_df[self.target_col]) / self.result_df[
                capacity_col]
        sns.displot(residuals)
        save_func(save_visual=save_visual, timestamp=self.timestamp, folder_name=self.folder_name,
                  filename='residual_distribution')
        plt.show()

    def plot_residuals_over_time(self, target_unit=None, save_visual=False):
        if target_unit is None:
            target_unit = 'EUR/MWh'
        residuals = self.result_df[self.predict_col] - self.result_df[self.target_col]
        f, ax = plt.subplots()
        f = sns.scatterplot(x=self.result_df.index, y=residuals, s=2, alpha=0.5)
        ax.set(ylabel=f'Residuals {target_unit}')
        plt.title('The residuals over time')
        functions.save_func(save_visual=save_visual, timestamp=self.timestamp, folder_name=self.folder_name,
                  filename='residuals_time')
        plt.show()

