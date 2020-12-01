import tensorflow as tf
import numpy as np


def usim_layer(layer1, layer2, batch=1):

    _, H, W, C = layer2.shape

    resized1 = tf.image.resize_nearest_neighbor(layer1, size=(2 * H, 2 * W))
    resized2 = tf.image.resize_nearest_neighbor(layer2, size=(2 * H, 2 * W))

    def get_init_values(h, w, flag='LU'):
        init_value = np.zeros(shape=(2 * h, 2 * w), dtype=np.float32)
        if   flag == 'LU': H, W = 0, 0
        elif flag == 'RU': H, W = 0, 1
        elif flag == 'LD': H, W = 1, 0
        elif flag == 'RD': H, W = 1, 1
        else             : return None

        for i in range(H, 2 * h, 2):
            for j in range(W, 2 * w, 2):
                init_value[i, j] = 1.0

        return init_value

    with tf.device('/cpu:0'):
        slsh_init = get_init_values(H, W , 'RU') + get_init_values(H , W , 'LD')
        slsh_init = np.expand_dims(slsh_init, axis=0)
        slsh_init = np.expand_dims(slsh_init, axis=3)

        bslh_init = get_init_values(H , W , 'LU') + get_init_values(H , W , 'RD')
        bslh_init = np.expand_dims(bslh_init, axis=0)
        bslh_init = np.expand_dims(bslh_init, axis=3)

        slsh = tf.get_variable(name='slsh_%d_%d_%d_%d' % (batch, H, W, C), initializer=slsh_init, trainable=False)
        bslh = tf.get_variable(name='bslh_%d_%d_%d_%d' % (batch, H, W, C), initializer=bslh_init, trainable=False)

    layer = resized1 * slsh + resized2 * bslh
    return layer

