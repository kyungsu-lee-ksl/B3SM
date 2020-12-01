import glob
import os
import cv2
import numpy as np
from ksl_util.file.image.image_loader import imshow2
from tqdm import tqdm
import argparse
import tensorflow as tf

from implementations.loss_functions import boundary_loss, iou_loss

from implementations.B3SM import B3SM
from util.Config import IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL, BATCH_SIZE

from implementations.evaluation_metric import iou_value, BIoU

parser = argparse.ArgumentParser(description='Enter valid args.')

parser.add_argument("--image_path", "--ip", default="./data/images/02_testset", metavar="IMAGE_PATH", help='path of images to load and test.')
parser.add_argument("--annotation_path", "--ap", default="./data/annotations/02_testset", metavar="ANNOTATION_PATH", help='path of annotations matched to loaded images')
parser.add_argument("--weight_path", "--wp", default=None, help='path of pre-trained weights.')

parser.add_argument("--height", default=IMAGE_HEIGHT, help="Image height", type=int)
parser.add_argument("--width", default=IMAGE_WIDTH, help="Image width", type=int)
parser.add_argument("--batch_size", default=BATCH_SIZE, help="batch_size", type=int)
parser.add_argument("--learning-rate", default=1e-3, help="learning rate", type=float)
parser.add_argument("--epoch", default=100, help="epochs", type=int)

args = parser.parse_args()



def main():

    path_images         = args.image_path
    path_annotations    = args.annotation_path

    weight_path         = args.weight_path

    names_imgs          = glob.glob('%s/*.png' % path_images)
    names_annotations   = glob.glob('%s/*.png' % path_annotations)

    imgs = np.asarray([cv2.imread(name, cv2.IMREAD_COLOR) for name in tqdm(names_imgs, desc="load images")])[:2]
    gnds = np.asarray([cv2.imread(name, cv2.IMREAD_GRAYSCALE) for name in tqdm(names_annotations, desc="load annotations")])[:2]

    indexes = np.asarray([i for i in range(0, len(imgs))])
    np.random.shuffle(indexes)

    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)

    imgHolder = tf.placeholder(dtype=tf.float32, shape=[args.batch_size, args.height, args.width, IMAGE_CHANNEL])
    gndHolder = tf.placeholder(dtype=tf.int32, shape=[args.batch_size, args.height, args.width])
    boundary_weights = tf.placeholder(dtype=tf.float32, shape=())

    prediction, logits = B3SM(imgHolder, args.batch_size, IMAGE_CHANNEL).structure()

    loss = iou_loss(logits, gndHolder) + boundary_loss(logits, gndHolder, batch_size=args.batch_size) * boundary_weights * 10

    optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
    grads = optimizer.compute_gradients(loss=loss, var_list=tf.trainable_variables(), colocate_gradients_with_ops=True)
    train_ops = optimizer.apply_gradients(grads)

    sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
    saver = tf.train.Saver()
    if args.weight_path is not None:
        saver.restore(sess, tf.train.latest_checkpoint(weight_path))

    average_loss_value = len(imgs)
    for epoch in range(1, args.epoch + 1):

        loss_value = average_loss_value / (float(args.batch_size) * (len(indexes) // args.batch_size + 1))
        average_loss_value = 0
        np.random.shuffle(indexes)

        for index in tqdm(range(0, len(indexes), args.batch_size), desc="epoch(%03d/%03d) loss(%.4f)" % (epoch, args.epoch, loss_value)):
            batch_index = list(indexes[index:index+args.batch_size])
            while len(batch_index) != args.batch_size:
                batch_index.append(np.random.randint(0, len(indexes)-1, 1)[0])

            batch_index = np.asarray(batch_index)
            batch_imgs, batch_gnds = imgs[indexes[batch_index]], gnds[indexes[batch_index]]

            loss_value, prd, _ = sess.run([loss, prediction, train_ops], feed_dict={imgHolder: batch_imgs, gndHolder: batch_gnds, boundary_weights: average_loss_value})
            average_loss_value += loss_value * args.batch_size

            prd = (prd[0] > 0).astype(np.uint8) * 255

            lists = [batch_imgs[0], batch_gnds[0], prd]
            [cv2.imshow('img%d' % window_index, image) for window_index, image in enumerate(lists)]
            [cv2.moveWindow('img%d' % window_index, 300 * window_index, 0) for window_index, image in enumerate(lists)]
            cv2.waitKey(1)







if __name__ == '__main__':

    assert os.path.exists(args.image_path)
    assert os.path.exists(args.annotation_path)

    main()
