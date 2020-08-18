from data_prep import DataPrep
from train import Train
import numpy as np
from sklearn.metrics import mean_squared_error


# Loading and processing data
file_path = './data/v2_CLD/AE_3C_1D/200716-174206_174209_174205'
data_prep = DataPrep(file_path)
data = data_prep.load_data(2, 'timestamp', True)

data = data_prep.total_intensity()
data = data_prep.downsampling('250ms')
data = data_prep.normalise(1)

print('\n', data[0].head())
print('\n', data[1].head())
print('\n', data[2].head())
print('\n')

# Generating training data
window = 480
data_train_raw = np.array(data[0]["2020-07-16 08:50:00":"2020-07-16 08:55:00"]['mag_f_norm'])
data_train = data_prep.window_sliding_GPU(data_train_raw, window, 10) # GPU
# data_train = data_prep.window_sliding_CPU(data_train_raw, window, 10) # CPU

# Generating test data
print('\nGenerating test data...')
data_test_1 = data[1]["2020-07-16 08:50:00":"2020-07-16 08:52:00"][['mag_f_norm']].values
data_test_1 = data_test_1[0:window]
data_test_1 = data_test_1.reshape(1,len(data_test_1))
print('Test data 1 shape : ', data_test_1.shape, '\n')

data_test_2 = data[2]["2020-07-16 08:50:00":"2020-07-16 08:52:00"][['mag_f_norm']].values
data_test_2 = data_test_2[0:window]
data_test_2 = data_test_2.reshape(1,len(data_test_2))
print('Test data 2 shape : ', data_test_2.shape, '\n')

# Training and testing
latent_dim = 10
epochs=100
batch_size=40

train = Train(data_train, window)
autoencoder, encoder, decoder, history = train.model1_ae('model1_ae/20200716_085000-20200716_085500', 'check_point',
                                                         latent_dim, epochs, batch_size)

latent_vector_1 = encoder.predict(data_test_1)
reconstructed_data_1 = decoder.predict(latent_vector_1)
mse_1 = mean_squared_error(data_test_1, reconstructed_data_1)
print('\nMSE test data (s1) : ', mse_1)

latent_vector_2 = encoder.predict(data_test_2)
reconstructed_data_2 = decoder.predict(latent_vector_2)
mse_2 = mean_squared_error(data_test_2, reconstructed_data_2)
print('\nMSE test data (s2) : ', mse_2)

# Test with data from different location
data_test_false = data[2]["2020-07-16 08:57:00":"2020-07-16 08:59:00"][['mag_f_norm']].values
data_test_false = data_test_false[0:window]
data_test_false = data_test_false.reshape(1,len(data_test_false))

print(data_test_false.shape)

latent_vector_false = encoder.predict(data_test_false)
reconstructed_data_false = decoder.predict(latent_vector_false)
mse_false = mean_squared_error(data_test_false, reconstructed_data_false)

print('\nMSE test data (s2) : ', mse_false)

# description = '%s-60-%s-60-%s\nrelu \nadam \nmse \nbatch_size=%s\nepochs=%s' % (window, latent_dim, window, batch_size, epochs)
# train.model1_ae_savemodel(autoencoder, encoder, decoder, '20200716_085000-20200716_085500', '0p5', description)