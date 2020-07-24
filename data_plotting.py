import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.dates import date2num

import os


class DataPlotting:

    def __init__(self):
        pass

    def plot_data(self, data, x_label, y_label, dir_name, file_name, label):
        font = {'family': 'DejaVu Sans',
                'weight': 'bold',
                'size': 15}

        plt.rc('font', **font)

        min_time = data.index.min()
        max_time = data.index.max()
        delta = max_time - min_time
        shape = data.shape
        print("Start :\n", min_time)
        print("End :\n", max_time)
        print('Time delta :\n', delta)
        print('\nShape :\n', shape)

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

    def plot_data_comp(self, data_1, data_2, x_label, y_label, dir_name, file_name, label):
        font = {'family': 'DejaVu Sans',
                'weight': 'bold',
                'size': 11}

        plt.rc('font', **font)

        min_time = data_1.index.min()
        max_time = data_1.index.max()
        delta = max_time - min_time
        print("Start :\n", min_time)
        print("End :\n", max_time)
        print('Time delta :\n', delta)

        fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True, linewidth=2)

        data_1.plot(figsize=(25, 15), linewidth=2, layout=(2, 1), fontsize=25, ax=axes[0])
        data_2.plot(figsize=(25, 15), linewidth=2, layout=(2, 1), fontsize=25, ax=axes[1])

        if label:
            dir_name = './logs/labelled/subplots/' + dir_name

            df_label_1 = self.label_post(data_1)
            self.plot_label(data_1, df_label_1, axes[0])
            axes[0].legend()
            df_label_2 = self.label_post(data_2)
            self.plot_label(data_2, df_label_2, axes[1])
            axes[1].legend()
            fig.autofmt_xdate()
        else:
            dir_name = './logs/unlabelled/subplots/' + dir_name

        plt.xlabel(x_label, fontsize=32);
        fig.text(0.06, 0.53, y_label, ha="center", va="center", rotation=90, fontsize=32)
        plt.suptitle(str(min_time) + ' to ' + str(max_time)
                     + '\nTime delta : ' + str(delta),
                     fontsize=25, y=0.95)

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
