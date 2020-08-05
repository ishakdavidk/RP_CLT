from data import Data
from train import Train
import numpy as np
from sklearn.metrics import mean_squared_error


# Loading and processing data
file_path_E8508BDFF2AE = './data/v2_dataset/E8508BDFF2AE/CLT (20200716_174206).txt'
file_path_F4428F5EB41D = './data/v2_dataset/F4428F5EB41D/CLT (20200716_174205).txt'
data = Data(file_path_E8508BDFF2AE, file_path_F4428F5EB41D)
data_1, data_2 = data.load_data(2, 'timestamp')

data_1, data_2 = data.total_intensity()
data_1, data_2 = data.downsampling('250ms')
data_1, data_2 = data.normalise(1)

print('\n', data_1.head())
print('\n', data_2.head())
print('\n')

# Generating training data
window = 480
data_train_raw = np.array(data_1["2020-07-16 08:50:00":"2020-07-16 08:55:00"]['mag_ti_norm'])
data_train = data.window_sliding_GPU(data_train_raw, window, 10) # GPU
# data_train = data.window_sliding_CPU(data_train_raw, window, 10) # CPU

# Generating test data
print('\nGenerating test data...')
data_test = data_2["2020-07-16 08:50:00":"2020-07-16 08:52:00"][['mag_ti_norm']].values
data_test = data_test[0:window]
data_test = data_test.reshape(1,len(data_test))
print('Test data shape : ', data_test.shape, '\n')

# Training and testing
latent_dim = 10
epochs=100
batch_size=40

train = Train(data_train, window)
autoencoder, encoder, decoder, history = train.model1_ae('model1_ae/20200716_085000-20200716_085500', 'check_point',
                                                         latent_dim, epochs, batch_size)

latent_vector = encoder.predict(data_test)
reconstructed_imgs = decoder.predict(latent_vector)

print('\nMSE test data (data_2) : ', mean_squared_error(data_test, reconstructed_imgs))

# Test with data from different location
data_test_false = data_2["2020-07-16 08:57:00":"2020-07-16 08:59:00"][['mag_ti_norm']].values
data_test_false = data_test_false[0:window]
data_test_false = data_test_false.reshape(1,len(data_test_false))

print(data_test_false.shape)

latent_vector_2 = encoder.predict(data_test_false)
reconstructed_data_false = decoder.predict(latent_vector_2)
mse_false = mean_squared_error(data_test_false, reconstructed_data_false)

print('MSE test false data (data_2) : ', mse_false)

# description = '%s-60-%s-60-%s\nrelu \nadam \nmse \nbatch_size=%s\nepochs=%s' % (window, latent_dim, window, batch_size, epochs)
# train.model1_ae_savemodel(autoencoder, encoder, decoder, '20200716_085000-20200716_085500', '0p5', description)