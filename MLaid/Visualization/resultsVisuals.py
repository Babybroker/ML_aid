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
        """
        Class used to deal with all the visualizations of the results

        :param result_df: Dataframe containing the results of an ML forecast
        :param target_col: Column name containing the actual values
        :param predict_col: Column name of the forecast values
        :param folder_name: Location to save visuals
        :param timestamp: A timestamp used to label the images
        """
        self.target_col = target_col
        self.predict_col = predict_col
        self.folder_name = folder_name
        self.timestamp = timestamp
        self.result_df = result_df

    def plot_result_pred_heatmap(self, save_visual: bool = False, extra_title_addition: str = None):
        """ Plot a heatmap of the actual values (x-axis) versus the predictions (y-axis) """
        f, ax = plt.subplots(figsize=(8, 8))

        max_val = max(self.result_df[self.target_col].max(), self.result_df[self.predict_col].max()) * 1.05
        min_val = min(self.result_df[self.target_col].min(), self.result_df[self.predict_col].min()) * 1.05
        ax.set(xlim=(min_val, max_val), ylim=(min_val, max_val))
        ax1 = sns.jointplot(data=self.result_df, x=self.target_col, y=self.predict_col)
        ax1.ax_joint.cla()
        plt.sca(ax1.ax_joint)
        plt.hist2d(self.result_df[self.predict_col], self.result_df[self.target_col], bins=(100, 100), cmap=cm.jet)
        plt.xlabel('Actual value')
        plt.ylabel('Prediction')
        title = f'Prediction vs actual value for the unseen set'
        if extra_title_addition is not None:
            title = title + ' ' + extra_title_addition
        plt.title(title)
        functions.save_func(save_visual=save_visual, timestamp=self.timestamp, folder_name=self.folder_name,
                  filename='result_heatmap')
        plt.show()

    def plot_scatter(
            self,
            extra_title_addition: str = None,
            custom_xlabel: str = None,
            custom_ylabel: str = None,
            hue: str = None,
            custom_palette: str = None,
            remove_outliers: bool = False,
            save_visual: bool = False,
            custom_title: str = None
    ):
        """
        Plot a scatter of the actual values (x-axis) versus the predictions (y-axis)

        :param extra_title_addition: Add some custom information to the title of the plot
        :param custom_xlabel: Specify a custom label for the x-axis
        :param custom_ylabel: Specify a custom label for the y-axis
        :param hue: The column name to be used as hue
        :param custom_palette: Specify a seaborn palette
        :param remove_outliers: Remove values in the target column that are 3 std away from the mean
        :param save_visual: Save the visual
        :param custom_title: Specify a custom title
        :return:
        """

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

        resultdf = self.result_df.copy()
        if remove_outliers:
            resultdf['z_score'] = (resultdf[self.target_col] - resultdf[self.target_col].mean()) / resultdf[
                self.target_col].std(ddof=0)
            resultdf = resultdf.query('z_score < 3')
        mae = round(mean_absolute_error(resultdf[self.target_col], resultdf[self.predict_col]), 2)
        r2 = round(r2_score(resultdf[self.target_col], resultdf[self.predict_col]), 3)
        rmse = round(np.sqrt(mean_squared_error(resultdf[self.target_col], resultdf[self.predict_col])), 2)
        test_mean = round(resultdf[self.target_col].mean())

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

        max_val = max(resultdf[self.target_col].max(), resultdf[self.predict_col].max()) * 1.05
        min_val = min(resultdf[self.target_col].min(), resultdf[self.predict_col].min()) * 1.05
        f, ax = plt.subplots(figsize=(8, 8))
        f = sns.scatterplot(data=resultdf, x=self.target_col, y=self.predict_col, s=2, alpha=0.5, hue=hue,
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
        """ Print the default metrics based on the result df """
        if self.predict_col is None:
            predict_col = [col for col in self.result_df if 'prediction' in col]
            print(f"{predict_col[0].replace('prediction_', '')} Results")

        print('Mean of test data:', self.result_df[self.target_col].mean())
        print('MSE actual values:',
              mean_squared_error(self.result_df[self.target_col], self.result_df[self.predict_col]))
        print('MAE actual values:',
              mean_absolute_error(self.result_df[self.target_col], self.result_df[self.predict_col]))
        print('R2 score:', r2_score(self.result_df[self.target_col], self.result_df[self.predict_col]))
        print('\n')

    def plot_time_steps(
            self,
            amnt_of_plots: int,
            custom_ylabel: str = None,
            custom_title: str = None,
            timesteps: int = 24,
            save_visual: bool = False
    ):
        """
        Plot a n-number of time steps

        :param amnt_of_plots: The number of plots to create
        :param custom_ylabel: Specify a custom label for the y-axis
        :param custom_title: Specify a custom title
        :param timesteps: The number of time steps to visualize
        :param save_visual: Save the visual
        :return:
        """
        from random import randint, seed
        seed(100)

        if self.predict_col is None:
            predict_col = [col for col in self.result_df if 'prediction' in col]
        fig2, ax2 = plt.subplots(amnt_of_plots, figsize=(9, 12))
        fig2.suptitle(custom_title)
        for i in range(amnt_of_plots):
            n = randint(0, int((len(self.result_df) - timesteps) / timesteps))
            data_slice = self.result_df.iloc[n * timesteps:n * timesteps + timesteps]
            ax2[i].plot(data_slice.index, data_slice[self.target_col])
            ax2[i].scatter(data_slice.index, data_slice[self.target_col], edgecolors='k', label='Actual', c='#2ca02c',
                           s=64)
            ax2[i].scatter(data_slice.index, data_slice[self.predict_col], marker='X', edgecolors='k',
                           label='Prediction',
                           c='#ff7f0e', s=64)
            if custom_ylabel is not None:
                ax2[i].set_ylabel(custom_ylabel)
            if i == 0:
                plt.legend(['Actual', 'Prediction'])
        plt.tight_layout()
        functions.save_func(save_visual=save_visual, timestamp=self.timestamp, folder_name=self.folder_name, filename='timestep')
        plt.show()

    def plot_residuals(
            self,
            custom_ylabel: str = None,
            extra_title_addition: str = None,
            train_set: bool = False,
            save_visual: bool = False,
            hue: str = None
    ):
        """
        Plot the residuals of the model
        :param hue: The column name to be used as hue
        :param save_visual: Save the visual
        :param custom_ylabel:Specify a custom label for the y-axis
        :param extra_title_addition: Add some custom information to the title of the plot
        :param train_set: If the residuals are from the train set
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

    def plot_residual_dist(self, capacity_col: str = None, save_visual: bool = False):
        """
        Plots the distribution of the residuals. If a capacity_col is specified, the difference of the residuals is
        divided by those values

        :param save_visual: Save the visual
        :param capacity_col: Column used to scale the values by
        :return:
        """
        if capacity_col is None:
            residuals = self.result_df[self.predict_col] - self.result_df[self.target_col]
        else:
            residuals = (self.result_df[self.predict_col] - self.result_df[self.target_col]) / self.result_df[
                capacity_col]
        sns.displot(residuals)
        functions.save_func(save_visual=save_visual, timestamp=self.timestamp, folder_name=self.folder_name,
                  filename='residual_distribution')
        plt.show()

    def plot_residuals_over_time(self, target_unit: str = None, save_visual: bool = False):
        """
        Create a plot of the residual error over time

        :param target_unit: The unit of the residuals, used for plotting the y label
        :param save_visual: Save the visual
        :return:
        """
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

