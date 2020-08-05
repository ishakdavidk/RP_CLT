import pandas as pd
import numpy as np
from sklearn import preprocessing
import tensorflow as tf
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '3'

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))

class Data:

    def __init__(self, file_path_1, file_path_2):
        self.file_path_1 = file_path_1
        self.file_path_2 = file_path_2

        self.data_1 = None
        self.data_2 = None
        pass

    def load_data(self, skip_rows, index_col):
        self.data_1 = pd.read_csv(self.file_path_1, sep=",", skipinitialspace=True, skiprows=skip_rows,
                             parse_dates=[index_col], index_col=index_col, low_memory=False)
        self.data_1.index = pd.to_datetime(self.data_1.index, unit='ms', origin='unix')

        self.data_2 = pd.read_csv(self.file_path_2, sep=",", skipinitialspace=True, skiprows=skip_rows,
                             parse_dates=[index_col], index_col=index_col, low_memory=False)
        self.data_2.index = pd.to_datetime(self.data_2.index, unit='ms', origin='unix')

        print("Data 1 : ")
        print(self.file_path_1, "loaded.")
        print(self.data_1.info())

        print("\nData 2 : ")
        print(self.file_path_2, "loaded.")
        print(self.data_2.info())

        return self.data_1, self.data_2

    def total_intensity(self):
        h1 = np.sqrt(self.data_1["mag_x"]**2 + self.data_1["mag_y"]**2)
        f1 = np.sqrt(h1**2 + self.data_1["mag_z"]**2)
        self.data_1.insert(4, 'mag_ti', f1)

        h2 = np.sqrt(self.data_2["mag_x"] ** 2 + self.data_2["mag_y"] ** 2)
        f2 = np.sqrt(h2 ** 2 + self.data_2["mag_z"] ** 2)
        self.data_2.insert(4, 'mag_ti', f2)

        return self.data_1, self.data_2

    def label_mapping(self):
        activity_map1, pos_state_map1 = self.mapping(self.data_1)
        self.data_1.insert(17, 'activity_num', activity_map1)
        self.data_1.insert(19, 'pos_state_num', pos_state_map1)

        activity_map2, pos_state_map2 = self.mapping(self.data_2)
        self.data_2.insert(17, 'activity_num', activity_map2)
        self.data_2.insert(19, 'pos_state_num', pos_state_map2)

        return self.data_1, self.data_2

    @staticmethod
    def mapping(data):
        activity_map = data.activity.map({'st': 0, 'wk': 1, 'r': 2, 'wb': 3, 'b': 4, 'wsb': 5, 'sb': 6})
        pos_state_map = data.pos_state.map({'i': 0, 'o': 1})

        return activity_map, pos_state_map

    def downsampling(self, unit):
        self.data_1 = self.data_1.resample(unit).mean()
        self.data_2 = self.data_2.resample(unit).mean()

        return self.data_1, self.data_2

    def normalise(self, type):
        if type == 1:
            min_max_scaler = preprocessing.MinMaxScaler()
            data_1_norm = min_max_scaler.fit_transform(self.data_1[['mag_ti']])
            data_2_norm = min_max_scaler.fit_transform(self.data_2[['mag_ti']])
        elif type == 2:
            data_1_norm = preprocessing.scale(self.data_1[['mag_ti']])
            data_2_norm = preprocessing.scale(self.data_2[['mag_ti']])
        else:
            print('Please select between the following options:\n1 : MinMaxScaler\n2 : scale')
            return self.data_1, self.data_2

        self.data_1['mag_ti_norm'] = data_1_norm
        self.data_2['mag_ti_norm'] = data_2_norm

        return self.data_1, self.data_2

    @staticmethod
    def window_sliding_GPU(data, window, duplicate):

        print('Data shape : ', data.shape)
        print('\nGenerating training data (GPU)...')

        data = tf.constant(data, dtype=tf.float64)

        for duplicate_itr in range(duplicate):
            for sliding_itr in range(len(data) - (window - 1)):
                if sliding_itr == 0 and duplicate_itr == 0:
                    data_train = tf.reshape(data[0:window], [1,window])
                else:
                    con = tf.reshape(data[sliding_itr:window + sliding_itr], [1,window])
                    data_train = tf.concat([data_train, con], 0)

        print('Training data shape : ', data_train.shape)
        return data_train

    @staticmethod
    def window_sliding_CPU(data, window, duplicate):

        print('Data shape : ', data.shape)
        print('\nGenerating training data (CPU)...')

        for duplicate_itr in range(duplicate):
            for sliding_itr in range(len(data) - (window - 1)):
                if sliding_itr == 0 and duplicate_itr == 0:
                    data_train = [data[0:window]]
                else:
                    data_train = np.append(data_train, [data[sliding_itr:window + sliding_itr]],
                                           axis=0)

        print('Training data shape : ', data_train.shape)
        return data_train

