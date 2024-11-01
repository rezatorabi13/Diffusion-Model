import tensorflow as tf
import numpy as np
from bakward_denoising import *
from forward_noising import generate_timestamp, forward_noise 

def loss_fn(real, generated):
    loss = tf.math.reduce_mean((real - generated) ** 2)
    return loss

