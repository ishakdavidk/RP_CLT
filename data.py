import pandas as pd
import numpy as np

import os


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

    def downsampling(self):
        self.data_1 = self.data_1.resample('1S').mean()
        self.data_2 = self.data_2.resample('1S').mean()

        return self.data_1, self.data_2

