import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.dates import date2num

import os


class DataPlotting:

    def __init__(self):
        pass

    def plot_data(self, data, x_label, y_label, font_size, dir_name, file_name, label):
        font = {'family': 'DejaVu Sans',
                'weight': 'bold',
                'size': font_size}

        plt.rc('font', **font)

        min_time, max_time, delta, shape = self.information(data, None)

        fig, ax = plt.subplots()

        data.plot(figsize=(20, 10), linewidth=2, layout=(2, 1), fontsize=25, ax=ax)

        if label:
            dir_name = './logs/labelled/plots/' + dir_name

            df_label = self.label_post(data)
            self.plot_label(data, df_label, ax)
            ax.legend()
            fig.autofmt_xdate()
        else:
            dir_name = './logs/unlabelled/plots/' + dir_name

        plt.xlabel(x_label, fontsize=32, fontweight='bold');
        plt.ylabel(y_label, fontsize=32, fontweight='bold');
        plt.suptitle(str(min_time) + ' to ' + str(max_time)
                     + '\nTime delta : ' + str(delta) + '\nShape : ' + str(shape),
                     fontsize=25)

        plt.tight_layout()
        plt.subplots_adjust(top=0.83)

        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        plt.savefig(dir_name + '/' + file_name + '.jpg', dpi=300)
        plt.show()

    def plot_data_comp(self, data, timestamp, data_display, x_label, y_label,
                       legend, title_font_size, fig_size, dir_name, file_name, label):
        font = {'family': 'DejaVu Sans',
                'weight': 'bold',
                'size': legend[0]}
        plt.rc('font', **font)

        fig, axes = plt.subplots(nrows=len(data), ncols=1, sharex=True, sharey=True, linewidth=2)

        time = []
        for n in range(len(data)):
            df = data[n][timestamp[0]:timestamp[1]][data_display]

            min_time, max_time, delta, shape = self.information(df, "Data " + str(n+1) + " : ")
            time.append([min_time, max_time, delta, shape])

            df.plot(figsize=fig_size, linewidth=2, layout=(3, 1), fontsize=25, ax=axes[n])
            if label:
                df_label = self.label_post(df)
                self.plot_label(df, df_label, axes[n])
                axes[n].legend()
                fig.autofmt_xdate()

            axes[n].get_legend().remove()

        handles, labels = plt.gca().get_legend_handles_labels()
        fig.legend(handles, labels, loc='center right', borderaxespad=0.3)
        plt.subplots_adjust(right=legend[1])

        plt.xlabel(x_label, fontsize=32, fontweight='bold');
        fig.text(0.06, 0.53, y_label, ha="center", va="center", rotation=90, fontsize=32)

        if all(v == time[0] for v in time):
            plt.suptitle(str(min_time) + ' to ' + str(max_time)
                         + '\nTime delta : ' + str(delta) + '\nShape : ' + str(shape),
                         fontsize=title_font_size, y=0.95)

        if label:
            dir_name = './logs/labelled/subplots/' + dir_name
        else:
            dir_name = './logs/unlabelled/subplots/' + dir_name

        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        plt.savefig(dir_name + '/' + file_name + '.jpg', dpi=300)
        plt.show()

    @staticmethod
    def label_post(data):
        n = 0
        row_prev = "N/A"
        for index, row in data.iterrows():
            if n == 0:
                df1 = pd.DataFrame([{'start': n, 'end': -1, 'label': row['activity']}])
            elif n == len(data) - 1:
                df1.iloc[-1, df1.columns.get_loc('end')] = n
            else:
                if row['activity'] != row_prev:
                    df1.iloc[-1, df1.columns.get_loc('end')] = n - 1
                    df2 = pd.DataFrame([{'start': n, 'end': -1, 'label': row['activity']}])
                    df1 = df1.append(df2, ignore_index=True)
            row_prev = row['activity']
            n = n + 1
        return df1

    @staticmethod
    def plot_label(data, df_label, ax):
        check = np.zeros(7)
        for index, row in df_label.iterrows():
            if row['label'] == "st":
                if check[0] == 0:
                    ax.axvspan(date2num(data.index[row['start']]), date2num(data.index[row['end']]),
                               label="Still", color=(1, 0, 0), alpha=0.3)
                else:
                    ax.axvspan(date2num(data.index[row['start']]), date2num(data.index[row['end']]),
                               color=(1, 0, 0), alpha=0.3)
                check[0] = 1
            elif row['label'] == "wk":
                if check[1] == 0:
                    ax.axvspan(date2num(data.index[row['start']]), date2num(data.index[row['end']]),
                               label="Walking", color=(0, 1, 0), alpha=0.3)
                else:
                    ax.axvspan(date2num(data.index[row['start']]), date2num(data.index[row['end']]),
                               color=(0, 1, 0), alpha=0.3)
                check[1] = 1
            elif row['label'] == "r":
                if check[2] == 0:
                    ax.axvspan(date2num(data.index[row['start']]), date2num(data.index[row['end']]),
                               label="Run", color=(1, 0.5, 0.3), alpha=0.3)
                else:
                    ax.axvspan(date2num(data.index[row['start']]), date2num(data.index[row['end']]),
                               color=(1, 0.5, 0.3), alpha=0.3)
                check[2] = 1
            elif row['label'] == "wb":
                if check[3] == 0:
                    ax.axvspan(date2num(data.index[row['start']]), date2num(data.index[row['end']]),
                               label="Waiting for bus", color=(1, 1, 0), alpha=0.3)
                else:
                    ax.axvspan(date2num(data.index[row['start']]), date2num(data.index[row['end']]),
                               color=(1, 1, 0), alpha=0.3)
                check[3] = 1
            elif row['label'] == "b":
                if check[4] == 0:
                    ax.axvspan(date2num(data.index[row['start']]), date2num(data.index[row['end']]),
                               label="Bus", color=(0.7, 0.2, 0.7), alpha=0.3)
                else:
                    ax.axvspan(date2num(data.index[row['start']]), date2num(data.index[row['end']]),
                               color=(0.7, 0.2, 0.7), alpha=0.3)
                check[4] = 1
            elif row['label'] == "wsb":
                if check[5] == 0:
                    ax.axvspan(date2num(data.index[row['start']]), date2num(data.index[row['end']]),
                               label="Waiting for subway", color=(0.7, 0.3, 0.1), alpha=0.3)
                else:
                    ax.axvspan(date2num(data.index[row['start']]), date2num(data.index[row['end']]),
                               color=(0.7, 0.3, 0.1), alpha=0.3)
                check[5] = 1
            elif row['label'] == "sb":
                if check[6] == 0:
                    ax.axvspan(date2num(data.index[row['start']]), date2num(data.index[row['end']]),
                               label="Subway", color="crimson", alpha=0.3)
                else:
                    ax.axvspan(date2num(data.index[row['start']]), date2num(data.index[row['end']]),
                               color="crimson", alpha=0.3)
                check[6] = 1

    @staticmethod
    def information(data, name):
        if name != None:
            print(name)
        min_time = data.index.min()
        max_time = data.index.max()
        delta = max_time - min_time
        shape = data.shape
        print('Start :\n', min_time)
        print('End :\n', max_time)
        print('Time delta :\n', delta)
        print('Shape :\n', shape, '\n')

        return min_time, max_time, delta, shape

    @staticmethod
    def plot_result(result_plot, x_label, y_label, font_size, parent_dir, directory, file_name, mse):
        font = {'family': 'DejaVu Sans',
                'weight': 'bold',
                'size': font_size}

        plt.rc('font', **font)

        result_plot.plot(figsize=(20, 10), linewidth=2, fontsize=20)
        plt.xlabel(x_label, fontsize=32, fontweight='bold');
        plt.ylabel(y_label, fontsize=32, fontweight='bold');

        plt.suptitle('MSE = ' + str(mse), fontsize=25)
        dir_name = './logs/result/' + parent_dir + '/' + directory

        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        plt.savefig(dir_name + '/' + file_name + '.jpg', dpi=300)
        plt.show()

    @staticmethod
    def plot_hist(hist_plot, x_label, y_label, font_size):
        font = {'family': 'DejaVu Sans',
                'weight': 'bold',
                'size': font_size}

        plt.rc('font', **font)

        hist_plot.plot(figsize=(10, 5), linewidth=2, fontsize=15)
        plt.xlabel(x_label, fontsize=20, fontweight='bold');
        plt.ylabel(y_label, fontsize=20, fontweight='bold');
        plt.show()

    @staticmethod
    def plot_hist_save(hist_plot, x_label, y_label, font_size, dir_name, file_name, display):
        font = {'family': 'DejaVu Sans',
                'weight': 'bold',
                'size': font_size}

        plt.rc('font', **font)

        fig, ax = plt.subplots()

        hist_plot.plot(figsize=(20, 10), linewidth=2, fontsize=20, ax=ax)
        plt.xlabel(x_label, fontsize=35, fontweight='bold');
        plt.ylabel(y_label, fontsize=35, fontweight='bold');

        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        fig.savefig(dir_name + '/' + file_name + '.jpg', dpi=300)

        if display:
            plt.show(fig)
        else:
            plt.close(fig)

