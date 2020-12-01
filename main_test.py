import glob
import os
import cv2
import numpy as np
from tqdm import tqdm
import argparse

from implementations.evaluation_metric import iou_value, BIoU

parser = argparse.ArgumentParser(description='Enter valid args.')

parser.add_argument("--image_path", "--ip", default="./data/images/02_testset", metavar="IMAGE_PATH", help='path of images to load and test.')
parser.add_argument("--annotation_path", "--ap", default="./data/annotations/02_testset", metavar="ANNOTATION_PATH", help='path of annotations matched to loaded images')
parser.add_argument("--model_path", "--mp", default="./B3SMStructure", help='path of deep learning model.')
parser.add_argument("--weight_path", "--wp", default="./weights", help='path of pre-trained weights.')


args = parser.parse_args()

import tensorflow as tf



def main():

    path_images         = args.image_path
    path_annotations    = args.annotation_path

    model_path          = args.model_path
    weight_path         = args.weight_path

    names_imgs          = glob.glob('%s/*.png' % path_images)
    names_annotations   = glob.glob('%s/*.png' % path_annotations)

    imgs = [cv2.imread(name, cv2.IMREAD_COLOR) for name in tqdm(names_imgs, desc="load images")]
    gnds = [cv2.imread(name, cv2.IMREAD_GRAYSCALE) for name in tqdm(names_annotations, desc="load annotations")]

    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)
    saver = tf.train.import_meta_graph('%s/%s' % (model_path, 'checkpoint.cpkt-0.meta'))
    saver.restore(sess, tf.train.latest_checkpoint(weight_path))

    base_graph = tf.get_default_graph()
    image_holder = base_graph.get_tensor_by_name('imgHolder:0')
    prediction_holder = tf.argmax(base_graph.get_tensor_by_name('scope/conv_1/BiasAdd:0'), axis=3)

    for index in tqdm(range(0, len(imgs))):
        img, gnd = imgs[index], gnds[index]
        prd = sess.run(prediction_holder, feed_dict={image_holder: [img]})[0]
        prd = ((prd > 0) * 255.).astype(np.uint8)

        iou = iou_value(prd, gnd)
        biou, bprd, bgnd = BIoU(prd, gnd, return_img=True)

        print(iou, biou)

        [cv2.imshow('img%d' % index, image) for index, image in enumerate([img, gnd, prd, bgnd, bprd])]
        [cv2.moveWindow('img%d' % index, 300 * index, 0) for index, image in enumerate([img, gnd, prd, bgnd, bprd])]
        cv2.waitKey(1)



if __name__ == '__main__':

    assert os.path.exists(args.image_path)
    assert os.path.exists(args.annotation_path)
    assert os.path.exists(args.model_path)
    assert os.path.exists(args.weight_path)

    main()
