import tensorflow as tf


def max_pooling(input, size=2):
    return tf.nn.max_pool(input, ksize=[1, size, size, 1], strides=[1, size, size, 1], padding="SAME")


def avg_pooling(input, size=2):
    return tf.nn.avg_pool(input, ksize=[1, size, size, 1], strides=[1, size, size, 1], padding="SAME")
