import tensorflow as tf

from util.Config import IMAGE_WIDTH, IMAGE_HEIGHT
from util.tf_utils import reveal_boundaries_tensor


def binary_cross_entropy_loss(logit, annotation, batch_size=None):

    label = tf.cast(annotation, dtype=tf.int32)
    label = tf.one_hot(indices=label, depth=2)

    sigmoid = tf.nn.sigmoid(logit)
    sig_p = tf.clip_by_value(sigmoid, 1e-8, 1.0)
    sig_n = tf.clip_by_value(1.0 - sigmoid, 1e-8, 1.0)

    loss = - label * tf.log(sig_p) - (1 - label) * tf.log(sig_n)
    loss = tf.reduce_mean(loss)

    return loss


def l2_loss(logit, annotation, batch_size=None):

    label = tf.cast(annotation, dtype=tf.int32)
    label = tf.one_hot(indices=label, depth=2)

    sigmoid = tf.nn.sigmoid(logit)

    loss = tf.abs(sigmoid - label)
    loss = tf.reduce_mean(loss)

    return loss


def iou_loss(logit, annotation, batch_size=None):
    label = tf.cast(annotation, dtype=tf.int32)
    label = tf.one_hot(indices=label, depth=2)

    sigmoid = tf.nn.sigmoid(logit)

    numerator = 2 * tf.reduce_sum(label * sigmoid)
    denominator = tf.reduce_sum(label) + tf.reduce_sum(sigmoid)
    return 1.0 - numerator / (denominator + 1e-8)



def boundary_loss(logit, annotation, batch_size=None):
    if batch_size is None:
        batch_size = int(annotation.shape[0])
    blabel = tf.expand_dims(annotation, axis=3)
    blabel = reveal_boundaries_tensor(blabel, batch_size=batch_size)

    bsigmoid = reveal_boundaries_tensor(tf.expand_dims(tf.argmax(logit, axis=3), axis=3), batch_size=batch_size)
    bsigmoid = tf.squeeze(tf.split(tf.nn.softmax(logit, axis=3), num_or_size_splits=2, axis=3)[1], axis=-1) *  bsigmoid

    numerator = 2 * tf.reduce_sum(blabel * bsigmoid)
    denominator = tf.reduce_sum(blabel) + tf.reduce_sum(bsigmoid)

    return 1.0 - numerator / (denominator + 1e-8)




if __name__ == '__main__':

    logit = tf.placeholder(dtype=tf.float32, shape=[10, IMAGE_HEIGHT, IMAGE_WIDTH, 2])
    annotations = tf.placeholder(dtype=tf.int32, shape=[10, IMAGE_HEIGHT, IMAGE_WIDTH])

    loss_functions = [
        binary_cross_entropy_loss, l2_loss, iou_loss, boundary_loss
    ]

    [loss_function(logit, annotations, 10) for loss_function in loss_functions]

