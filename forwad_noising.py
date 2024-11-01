import numpy as np
from util import *

# set the RNG key for Numpy
def RNG_key(key):
    np.random.seed(key)

# add noise to the input as per the given timestamp
def forward_noise(key, x_0, t, sqrt_alpha_bar, one_minus_sqrt_alpha_bar):
    RNG_key(key)
    noise = np.random.normal(size=x_0.shape)
    reshaped_sqrt_alpha_bar_t = np.reshape(np.take(sqrt_alpha_bar, t), (-1, 1, 1, 1))
    reshaped_one_minus_sqrt_alpha_bar_t = np.reshape(np.take(one_minus_sqrt_alpha_bar, t), (-1, 1, 1, 1))
    noisy_image = reshaped_sqrt_alpha_bar_t  * x_0 + reshaped_one_minus_sqrt_alpha_bar_t  * noise
    return noisy_image, noise

# create sample timestamps between 0 & T
def generate_timestamp(key, num, T):
    RNG_key(key)
    return tf.random.uniform(shape=[num], minval=0, maxval=T, dtype=tf.int32)