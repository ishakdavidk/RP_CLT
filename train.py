# import the necessary packages
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from tensorflow.python.client import device_lib

from data_plotting import DataPlotting

import numpy as np
import pandas as pd
import os
import time

# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))


class Train:
    def __init__(self, data_train, window):
        self.data_train = data_train
        self.window = window
        pass

    def model1_ae(self, parent_dir, directory, latent_dim=9, epochs=100, batch_size=40):

        ts = int(time.time())
        dir_name = './models/' + parent_dir + '/' + directory + '/' + str(ts)

        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        checkpoint_path = dir_name + '/' + "cp-{epoch:04d}.ckpt"

        input_data = Input(shape=(self.window,))

        encoded = Dense(60, activation='relu')(input_data)
        encoded = Dense(latent_dim, activation='relu')(encoded)
        decoder = Dense(60, activation='relu')(encoded)
        decoder = Dense(self.window, activation='relu')(decoder)

        # create autoencoder model
        autoencoder = Model(inputs=input_data, outputs=decoder, name="autoencoder")

        autoencoder.summary()

        autoencoder.compile(optimizer='adam', loss='mse')
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                         save_weights_only=True,
                                                         verbose=1)
        history = autoencoder.fit(self.data_train, self.data_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        shuffle=False,
                        callbacks=[cp_callback])

        # create encoder model
        encoder = Model(inputs=input_data, outputs=encoded, name="encoder")

        # create decoder model
        encoded_input = Input(shape=(latent_dim,))
        decoder_layer1 = autoencoder.layers[-2]
        decoder_layer2 = autoencoder.layers[-1]
        decoder = Model(inputs=encoded_input, outputs=decoder_layer2(decoder_layer1(encoded_input)),
                        name="decoder")

        return autoencoder, encoder, decoder, history

    @staticmethod
    def model_ae_save(autoencoder, encoder, decoder, history, parent_dir, directory, description):
        dir_name = './models/' + parent_dir + '/' + directory

        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        autoencoder.save(dir_name + '/' + 'autoencoder.h5')
        encoder.save(dir_name + '/' + 'encoder.h5')
        decoder.save(dir_name + '/' + 'decoder.h5')

        if description is not None:
            with open(dir_name + '/' + 'description.txt', 'w') as text_file:
                print(description, file=text_file)

        np.save(dir_name + '/' + 'history.npy', history.history)

        data_plotting = DataPlotting()
        history_dict = history.history
        loss_values = history_dict['loss']
        hist_plot = pd.DataFrame({'MSE Loss': loss_values})
        data_plotting.plot_hist_save(hist_plot, 'Epochs', 'Loss', 20, dir_name, 'loss_history', False)

    @staticmethod
    def model_ae_load(parent_dir, directory):
        dir_name = './models/' + parent_dir + '/' + directory

        autoencoder = load_model(dir_name + '/' + 'autoencoder.h5')
        encoder = load_model(dir_name + '/' + 'encoder.h5')
        decoder = load_model(dir_name + '/' + 'decoder.h5')

        history = np.load(dir_name + '/' + 'history.npy', allow_pickle='TRUE').item()

        autoencoder.summary()

        return autoencoder, encoder, decoder, history
