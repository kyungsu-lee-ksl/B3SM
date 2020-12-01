import tensorflow as tf
import numpy as np


def batch_norm(x, n_out, decay=0.99, eps=1e-5, name=None, trainable=True): # n_out=필터의 수

    if name is None:
        name = 'norm'

    with tf.variable_scope(name):
        with tf.device('/cpu:0'):
            beta = tf.get_variable(name='beta', shape=[n_out], initializer=tf.constant_initializer(0.0)
                                   , trainable=trainable)
            gamma = tf.get_variable(name='gamma', shape=[n_out], initializer=tf.random_normal_initializer(1.0, 0.02),
                                    trainable=trainable)

        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments') # 평균과 분산 계산
        ema = tf.train.ExponentialMovingAverage(decay=decay)

    def mean_var_with_update():
        ema_apply_op = ema.apply([batch_mean, batch_var])
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)

    mean, var = tf.cond(tf.constant(True), mean_var_with_update, lambda: (ema.average(batch_mean), ema.average(batch_var)))
    normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, eps)

    return normed


def group_norm(x, batch, G, gamma=None, beta=None, eps=1e-5, name='grouppNorm', trainable=True):

    _, H, W, C = x.shape

    if gamma is None:
        gamma = tf.get_variable(name="%s_gamma" % name, trainable=trainable, initializer=np.ones(shape=[1, 1, 1, C], dtype=np.float32))

    if beta is None:
        beta = tf.get_variable(name="%s_beta" % name, trainable=trainable, initializer=np.zeros(shape=[1, 1, 1, C], dtype=np.float32))

    x = tf.reshape(x, [batch, H, W, C // G, G])

    mean, var = tf.nn.moments(x, axes=[1, 2, 3], keep_dims=True)
    x = (x - mean) / tf.sqrt(var + eps)

    x = tf.reshape(x, [batch, H, W, C])

    return x * gamma + beta

