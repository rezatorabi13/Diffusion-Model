from parameters import Param
import tensorflow as tf
import tensorflow_datasets as tfds

def preprocess(x, y):
    processed = tf.image.resize(tf.cast(x, tf.float32) / 127.5 - 1, (32, 32))
    return processed

def loadData(batch):
    # Load the dataset
    train_data = tfds.load('cifar10', as_supervised=True, split="train")

    train_data = train_data.map(preprocess, tf.data.AUTOTUNE)
    train_data = train_data.shuffle(5000).batch(batch).prefetch(tf.data.AUTOTUNE)
    arr = tfds.as_numpy(train_data)
    return arr