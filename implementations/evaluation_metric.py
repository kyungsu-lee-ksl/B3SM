import cv2
import numpy as np

def canny(img):
    tmp = cv2.erode(img, kernel=np.ones(shape=(3, 3), dtype=np.float32) * 2, iterations=3)
    tmp = cv2.Canny(tmp, 127, 255)
    tmp = cv2.dilate(tmp, kernel=np.ones(shape=(3, 3), dtype=np.float32) * 2, iterations=2)
    return tmp


def BIoU(prd, gnd, function=canny, return_img=False):

    prd, gnd = function(prd), function(gnd)

    tp, fp, fn, _ = miou(prd, gnd)

    if not return_img:
        return tp / (tp + fp + fn) if tp + fp + fn != 0 else 1.0
    else:
        return tp / (tp + fp + fn) if tp + fp + fn != 0 else 1.0, prd, gnd


def miou(pred, anno):

    tp = np.logical_and(pred, anno)
    tp = np.asarray(tp, 'float64')
    tp = np.sum(tp)

    fp = np.logical_and(np.logical_not(anno), pred)
    fp = np.asarray(fp, 'float64')
    fp = np.sum(fp)

    fn = np.logical_and(np.logical_not(pred), anno)
    fn = np.asarray(fn, 'float64')
    fn = np.sum(fn)

    tn = np.logical_and(np.logical_not(pred), np.logical_not(anno))
    tn = np.asarray(tn, 'float64')
    tn = np.sum(tn)

    return tp, fp, fn, tn


def iou_value(pred, anno):
    tp, fp, fn, tn = miou(pred, anno)
    return tp / (tp + fp + fn) if tp + fp + fn != 0 else 1.0


def biou_value(pred, anno):
    return BIoU(pred, anno, return_img=False)

