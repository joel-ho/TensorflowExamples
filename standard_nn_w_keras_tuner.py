import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
import keras_tuner as kt

# Constants
RAND_SEED = 888
N_INPUT = 1
N_OUTPUT = 1
N_SAMPLE = 300
VAL_FRAC = 0.2
BATCH_SIZE = 32
N_EPOCHS = 100

# Generate training data
eps_gen = tfp.distributions.Normal(loc=0, scale=0.02)
eps = eps_gen.sample(N_SAMPLE).numpy();

np.random.seed(RAND_SEED)
x = 0.5*np.random.rand(N_SAMPLE) # Random number in [0, 0.5)
y = x + 0.3*np.sin(2*np.pi*(x+eps)) + 0.3*np.sin(4*np.pi*(x+eps)) + eps

# Set up model architecture
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)]

def build_model(hp):
    model = keras.models.Sequential()
    model.add(keras.layers.Input(shape=(N_INPUT, )))
    reg = hp.Float('reg', 1e-5, 1e-1, sampling='log')
    for i_layer in range(hp.Int('n_layers', 2, 5)):
        model.add(
            keras.layers.Dense(
                hp.Choice('n_nodes_{}'.format(i_layer), [16, 32, 64, 128]), 
                activation='relu', 
                kernel_regularizer=tf.keras.regularizers.L2(reg)))
    model.add(keras.layers.Dense(N_OUTPUT, activation='linear'))
    model.compile(
        loss='mean_squared_error', 
        optimizer=keras.optimizers.Adam(
            hp.Float('learn_rate', 1e-4, 1e-2, sampling='log')))
    return model

# Train with tuner

# tuner = kt.Hyperband(
#     build_model,
#     objective='val_loss',
#     max_epochs=N_EPOCHS,
#     project_name='std_nn_hyperband')

tuner = kt.BayesianOptimization( 
    build_model,
    objective='val_loss',
    max_trials=5,
    project_name='std_nn_bayesian_opt')

tuner.search(x, y, 
             epochs=N_EPOCHS, batch_size=BATCH_SIZE, 
             validation_split=VAL_FRAC, callbacks=callbacks)

# Plot results
model = tuner.get_best_models()[0]
x_out = np.linspace(-0.5, 1, 100)
y_out = model.predict(x_out)

plt.figure()
plt.scatter(x, y)
plt.plot(x_out, y_out)
plt.grid(True)