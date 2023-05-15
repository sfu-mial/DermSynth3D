import os
import torch
import numpy as np
from sklearn.metrics import jaccard_score

from dermsynth3d.utils.channels import Target
from dermsynth3d.utils.image import load_image


def dice(pred, gt):
    intersection = np.sum(pred & gt)
    union = np.sum(pred | gt)
    if gt.sum() == 0:
        if pred.sum() == 0:
            return 1
        else:
            return 0
    return 2 * intersection / (union + intersection)


def dice_score(target, input):
    # Dice on tensors
    smooth = 1e-8

    iflat = input.flatten()
    tflat = target.flatten()
    intersection = (iflat * tflat).sum()

    if (tflat.sum() == 0) and (iflat.sum() == 0):
        return 1

    return (2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)


def precision(pred, gt):
    intersection = np.sum(pred & gt)

    if gt.sum() == 0:
        if pred.sum() == 0:
            return 1
        else:
            return 0
    return intersection / pred.sum()


def jaccard_index(pred, gt):
    intersection = np.sum(pred & gt)
    union = np.sum(pred | gt)

    if gt.sum() == 0:
        if pred.sum() == 0:
            return 1
        else:
            return 0
    return intersection / union


def recall(pred, gt):
    intersection = np.sum(pred & gt)

    if gt.sum() == 0:
        if pred.sum() == 0:
            return 1
        else:
            return 0
    return intersection / gt.sum()


def argmax_predictions(pred, asint=False):
    pred = np.asarray(pred)

    if len(pred.shape) != 3:
        raise ValueError("Must be HxWxC input")

    if pred.shape[-1] < 2:
        raise ValueError("Cannot handle a single channel. C must be >= 2 ")

    argmax_preds = np.argmax(pred, axis=-1)
    binary_preds = np.zeros(
        shape=(pred.shape[0], pred.shape[1], pred.shape[2]), dtype=np.bool8
    )
    for idx in np.arange(binary_preds.shape[-1]):
        binary_preds[:, :, idx] = argmax_preds == idx

    if asint:
        binary_preds = (binary_preds * 255).astype(np.uint8)

    return binary_preds


def compute_results(ds, dir_predictions, pred_ext):
    results = []
    for image_id in ds.file_ids:
        gt_seg = np.asarray(ds.target(image_id))
        pred_filename = os.path.join(dir_predictions, image_id + pred_ext)
        pred_seg = np.asarray(load_image(pred_filename))
        # pred_seg = np.asarray(ds.prediction(image_id))
        pred_seg = argmax_predictions(pred_seg)

        image_res = {}
        image_res["file_id"] = image_id

        for c_idx in range(gt_seg.shape[-1]):
            pred_channel = pred_seg[:, :, c_idx]
            gt_channel = gt_seg[:, :, c_idx] > 0

            ji = jaccard_index(gt_channel, pred_channel)
            if c_idx == Target.LESION:
                image_res["lesion_ji"] = ji

            if c_idx == Target.SKIN:
                image_res["skin_ji"] = ji

            if c_idx == Target.NONSKIN:
                image_res["nonskin_ji"] = ji

        results.append(image_res)

    return results


def compute_results_segmentation(ds, dir_predictions, pred_ext):
    results = []
    for image_id in ds.file_ids:
        gt_seg = np.asarray(ds.target(image_id))
        pred_filename = os.path.join(dir_predictions, image_id + pred_ext)
        pred_seg = np.asarray(load_image(pred_filename))

        image_res = {}
        image_res["file_id"] = image_id

        pred_channel = (np.asarray(pred_seg)[:, :, 0] / 255) > 0.5
        gt_channel = (np.asarray(gt_seg)[:, :, 0] / 255) > 0

        iou = jaccard_index(gt_channel, pred_channel)
        dice_score = dice(gt_channel, pred_channel)

        image_res["iou"] = iou
        image_res["dice"] = dice_score

        results.append(image_res)

    return results


def conf_mat_cells(ds, dir_predictions, pred_ext):
    tps = []
    fps = []
    tns = []
    fns = []
    for img_id in ds.file_ids:
        gt = np.asarray(ds.target(img_id))
        gt = gt[:, :, 0] > 0
        pred_filename = os.path.join(dir_predictions, img_id + pred_ext)
        pred = np.asarray(load_image(pred_filename))
        skin_pred = argmax_predictions(pred)[:, :, Target.SKIN]

        tp = np.sum((skin_pred == True) & (gt == True))
        fp = np.sum((skin_pred == True) & (gt == False))
        tn = np.sum((skin_pred == False) & (gt == False))
        fn = np.sum((skin_pred == False) & (gt == True))
        assert tp + fp + tn + fn == (gt.shape[0] * gt.shape[1])

        tps.append(tp)
        fps.append(fp)
        tns.append(tn)
        fns.append(fn)

    return {
        "tps": tps,
        "fps": fps,
        "tns": tns,
        "fns": fns,
    }
