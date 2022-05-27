import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp
from tensorflow.python.client import device_lib

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
device_lib.list_local_devices()

n_sample = 300

# Create example training data
n_sample_half = int(n_sample/2)
eps_gen = tfp.distributions.Normal(loc=0, scale=0.02)
eps = eps_gen.sample(n_sample_half).numpy();
x = 0.5*np.random.rand(n_sample_half) # Random number in [0, 0.5)
y = x + 0.3*np.sin(2*np.pi*(x+eps)) + 0.3*np.sin(4*np.pi*(x+eps)) + eps
eps = eps_gen.sample(n_sample_half).numpy();
x_neg = 0.5*np.random.rand(n_sample_half) # Random number in [0, 0.5)
y_neg = -(x_neg + 0.3*np.sin(2*np.pi*(x_neg+eps)) + 0.3*np.sin(4*np.pi*(x_neg+eps)) + eps)

x = np.hstack((x, x_neg))
y = np.hstack((y, y_neg))

shuffle_idx = np.arange(0, n_sample, dtype=int)
np.random.shuffle(shuffle_idx)
x = x[shuffle_idx]
y = y[shuffle_idx]

# Set up model
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)]

model = keras.models.Sequential()
model.add(keras.layers.Input(shape=(1, ), name='input'))
for i in (32, 32):
    model.add(keras.layers.Dense(
        i, 
        activation='relu', 
        kernel_regularizer=tf.keras.regularizers.L2(0.001)))
    
n_components = 2
params_size = tfp.layers.MixtureNormal.params_size(n_components, [1, ])
model.add(keras.layers.Dense(params_size, activation=None, name='gaussian_params'))
model.add(tfp.layers.MixtureNormal(
    n_components, [1, ],
    convert_to_tensor_fn=tfp.distributions.Distribution.sample))
model.compile(optimizer=keras.optimizers.Adam(0.005),
              loss=lambda y, model: -model.log_prob(y))
model.fit(x, y, batch_size=32, epochs=200, validation_split=0.3, callbacks=callbacks)
model_gaussian_params = keras.models.Model(
    inputs=model.input, outputs=model.get_layer('gaussian_params').output)

# Plot results sampled from model
plt.figure()
plt.subplot(2, 2, 1)
plt.scatter(x, y)
x_out = np.linspace(-0.25, 0.75, 100)
for _ in range(20):
    y_out = model.predict(x_out) # predict samples from output distribution
    plt.scatter(x_out, y_out, marker='.')
plt.title('Sampled output')
plt.grid(True)

# model_gaussian_params shows the output right before the MDN layer
# First n columns show the weights for the n Gaussians (before softmax) 
# followed by (mean, scaled stddev) pairs for individual Gaussians.
y_params = model_gaussian_params.predict(x_out)

# call method on model object returns a distribution object.
# Distribution object has mean() and stddev(), but only 
# shows for the combined results.
y_distr_out = model(x_out[np.newaxis].T) 

plt.subplot(2, 2, 2)
weights = tf.nn.softmax(y_params[:, :2])
plt.plot(x_out, weights[:, 0], label='Gaussian 1')
plt.plot(x_out, weights[:, 1], label='Gaussian 2')
plt.title('Output Gaussian weights')
plt.grid(True)
plt.legend().set_draggable(True)

plt.subplot(2, 2, 3)
plt.plot(x_out, y_params[:, 2], label='Gaussian 1')
plt.plot(x_out, y_params[:, 4], label='Gaussian 2')
plt.plot(x_out, y_distr_out.mean(), label='Mixture')
plt.title('Output Gaussian mean')
plt.grid(True)
plt.legend().set_draggable(True)

# Can't work out how they scale the stddev component wise. Probably need to 
# dig into the code 
plt.subplot(2, 2, 4)
plt.plot(x_out, y_params[:, 3], label='Gaussian 1 (unknown scaling)')
plt.plot(x_out, y_params[:, 5], label='Gaussian 2 (unknown scaling)')
plt.plot(x_out, y_distr_out.stddev(), label='mixture')
plt.title('Output Gaussian scaled std dev')
plt.grid(True)
plt.legend().set_draggable(True)

plt.show()
