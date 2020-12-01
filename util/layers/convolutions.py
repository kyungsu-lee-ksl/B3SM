import tensorflow as tf


def conv_layer(input, input_channel, output_channel, mean=0.0, std=1., bias=0.0, filter_size=3, name=None):

    if name is None:
        name = 'conv_layer'

    with tf.variable_scope(name):

        shape = [filter_size, filter_size, input_channel, output_channel]

        with tf.device('/cpu:0'):
            W = tf.get_variable(name="W", trainable=True, initializer=tf.truncated_normal(shape=shape, mean=mean, stddev=std))
            B = tf.get_variable(name="B", trainable=True, initializer=tf.constant(bias, shape=[output_channel]))

        conv = tf.nn.conv2d(input, W, strides=[1, 1, 1, 1], padding="SAME")
        conv = tf.nn.bias_add(conv, B)

    return conv


def deconv_layer(input, input_channel, output_channel, mean=0.0, std=1., bias=0.0, filter_size=3, stride=2, name=None, batch_size=1):

    if name is None:
        name = 'deconv_layer'

    with tf.variable_scope(name):
        output_shape = [batch_size, int(input.shape[1] * stride), int(input.shape[2] * stride), output_channel]
        shape = [filter_size, filter_size, output_channel, input_channel]


        with tf.device('/cpu:0'):
            W = tf.get_variable(name='W', trainable=True, initializer=tf.truncated_normal(shape=shape, mean=mean, stddev=std))
            B = tf.get_variable(name='B', trainable=True, initializer=tf.constant(bias, shape=[output_channel]))

        conv = tf.nn.conv2d_transpose(input, W, output_shape=output_shape, strides=[1, stride, stride, 1], padding="SAME")
        conv = tf.nn.bias_add(conv, B)

        return conv
