import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import glob
from sklearn import preprocessing
import tensorflow as tf
import cv2
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '3'

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))

class DataPrep:

    def __init__(self, files_path):
        self.files_path = files_path

        self.data = []
        pass

    def load_data(self, skip_rows, index_col, info):
        files = glob.glob(self.files_path + '/*.txt')

        for file in files:
            df = pd.read_csv(file, sep=",", skipinitialspace=True, skiprows=skip_rows,
                             parse_dates=[index_col], index_col=index_col, low_memory=False)
            df.index = pd.to_datetime(df.index, unit='ms', origin='unix')
            self.data.append(df)

        if len(self.data) == 0:
            print('No data were found in this directory')
            return None
        else:
            if info:
                for n in range(len(self.data)):
                    print('Data ', n + 1, ' : ')
                    print(files[n], 'loaded.')
                    print(self.data[n].info(), '\n')

        return self.data

    def total_intensity(self):
        for n in range(len(self.data)):
            h = np.sqrt(self.data[n]['mag_x'] ** 2 + self.data[n]['mag_y'] ** 2)
            f = np.sqrt(h ** 2 + self.data[n]['mag_z'] ** 2)
            self.data[n].insert(3, 'mag_f', f)

        return self.data

    def label_mapping(self):
        for n in range(len(self.data)):
            activity_map, pos_state_map = self.mapping(self.data[n])
            self.data[n].insert(17, 'activity_num', activity_map)
            self.data[n].insert(19, 'pos_state_num', pos_state_map)

        return self.data

    @staticmethod
    def mapping(data):
        activity_map = data.activity.map({'st': 0, 'wk': 1, 'r': 2, 'wb': 3, 'b': 4, 'wsb': 5, 'sb': 6})
        pos_state_map = data.pos_state.map({'i': 0, 'o': 1})

        return activity_map, pos_state_map

    def downsampling(self, unit):
        for n in range(len(self.data)):
            self.data[n] = self.data[n].resample(unit).mean()

        return self.data

    def normalise(self, norm_type):
        if norm_type == 1:
            min_max_scaler = preprocessing.MinMaxScaler()
            for n in range(len(self.data)):
                data_norm = min_max_scaler.fit_transform(self.data[n][['mag_f']])
                self.data[n]['mag_f_norm'] = data_norm
        elif norm_type == 2:
            for n in range(len(self.data)):
                data_norm = preprocessing.scale(self.data[n][['mag_f']])
                self.data[n]['mag_f_norm'] = data_norm
        else:
            print('Please select between the following options:\n1 : MinMaxScaler\n2 : scale')
            return self.data

        return self.data

    @staticmethod
    def training_matrix_GPU(data, window, duplicate):
        # Generating training matrix data using window sliding technique in GPU

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
    def training_matrix_CPU(data, window, duplicate):
        # Generating training matrix data using window sliding technique in CPU

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

    @staticmethod
    def training_img(data, window, duplicate, dir_name, fill):
        # Generating training image data using window sliding technique and plt plot

        print('Data shape : ', data.shape)
        print('\nGenerating training data...')

        data = tf.constant(data, dtype=tf.float64)

        if fill:
            x = np.arange(0, window, 1)
            final_dir_name = './generated_data/training/fill/' + dir_name
        else:
            final_dir_name = './generated_data/training/line/' + dir_name

        if not os.path.exists(final_dir_name):
            os.makedirs(final_dir_name)

        n = 0
        for duplicate_itr in range(duplicate):
            for sliding_itr in range(len(data) - (window - 1)):
                data_train = data[sliding_itr:window + sliding_itr]

                n = n + 1
                with plt.style.context('grayscale'):
                    plt.xlim(0, window)
                    plt.ylim(0, 1)
                    plt.axis('off')
                    plt.plot(data_train, linewidth=0.1)

                    if fill:
                        plt.fill_between(x, data_train, color='#808080')

                    fig = plt.gcf()
                    fig.set_dpi(100)
                    fig.set_size_inches(0.32, 0.32)
                    file_name = str(n)
                    fig.savefig(final_dir_name + '/' + file_name + '.png')

                    plt.cla()

        plt.close()

        print('Number of generated images : ', n)
        print('Directory : ' + final_dir_name)

    @staticmethod
    def test_img(data, window, dir_name, file_name, fill):
        # Generating test image data using plt plot

        print('Data shape : ', data.shape)
        print('\nGenerating test data...')

        data = tf.constant(data, dtype=tf.float64)

        if fill:
            x = np.arange(0, window, 1)
            final_dir_name = './generated_data/test/fill/' + dir_name
        else:
            final_dir_name = './generated_data/test/line/' + dir_name

        if not os.path.exists(final_dir_name):
            os.makedirs(final_dir_name)

        data_test = data[0:window]
        with plt.style.context('grayscale'):
            plt.xlim(0, window)
            plt.ylim(0, 1)
            plt.axis('off')
            plt.plot(data_test, linewidth=0.1)

            if fill:
                plt.fill_between(x, data_test, color='#808080')

            fig = plt.gcf()
            fig.set_dpi(100)
            fig.set_size_inches(0.32, 0.32)
            fig.savefig(final_dir_name + '/' + file_name + '.png')

            plt.close()

        print('Number of generated images : 1')
        print('Directory : ' + final_dir_name)

    @staticmethod
    def load_images(directory):
        images = []

        for filename in os.listdir(directory):
            img = cv2.imread(os.path.join(directory, filename), 0)
            if img is not None:
                images.append(img)

        images = np.array(images)

        images = np.expand_dims(images, axis=-1)
        images = images.astype("float32") / 255.0

        print('Directory : ' + directory)
        print('Shape : ', images.shape)

        return images