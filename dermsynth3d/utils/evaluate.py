import os
import numpy as np
from dermsynth3d.utils.channels import Target
from dermsynth3d.utils.image import load_image
from typing import Tuple


def jaccard_index(
    gt,
    pred,
):
    intersection = np.sum(pred & gt)
    union = np.sum(pred | gt)

    if gt.sum() == 0:
        if pred.sum() == 0:
            return 1
        else:
            return 0

    return intersection / union


def dice(
    target,
    input,
):
    """
    https://github.com/pytorch/pytorch/issues/1249
    """
    smooth = 0.0

    iflat = input.flatten()
    tflat = target.flatten()
    intersection = (iflat * tflat).sum()

    if (tflat.sum() == 0) and (iflat.sum() == 0):
        return 1

    return (2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)


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

            tp, fp, tn, fn = binary_conf_mat(pred_channel, gt_channel)

            ji = jaccard_index(gt_channel, pred_channel)
            if c_idx == Target.LESION:
                key = "lesion"
            elif c_idx == Target.SKIN:
                key = "skin"
            elif c_idx == Target.NONSKIN:
                key = "nonskin"
            else:
                raise ValueError("Error: outside expected range")

            image_res[key + "_ji"] = ji
            image_res[key + "_tp"] = tp
            image_res[key + "_fp"] = fp
            image_res[key + "_tn"] = tn
            image_res[key + "_fn"] = fn

        results.append(image_res)

    return results


def print_results(df, keys=("lesion", "skin", "nonskin")):
    ji_all = {}
    f1_all = {}
    for key in keys:
        tp = df[key + "_tp"].sum()
        fp = df[key + "_fp"].sum()
        tn = df[key + "_tn"].sum()
        fn = df[key + "_fn"].sum()
        ji_all[key] = tp / (tp + fp + fn)
        f1_all[key] = (2 * tp) / ((2 * tp) + fp + fn)

    for key in keys:
        print(
            "{} JI={:.2f} F1={:.2f}".format(
                key,
                ji_all[key],
                f1_all[key],
            )
        )

    for key in keys:
        print(
            "{} JI={:.2f} ({:.2f})".format(
                key,
                df[key + "_ji"].mean(),
                df[key + "_ji"].std(),
            )
        )


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


def binary_conf_mat(pred: np.array, gt: np.array) -> Tuple[int, int, int, int]:
    if pred.dtype != "bool":
        raise TypeError("Expects `pred` to be a boolean array.")

    if gt.dtype != "bool":
        raise TypeError("Expects `gt` is a boolean array.")
    tp = np.sum((pred == True) & (gt == True))
    fp = np.sum((pred == True) & (gt == False))
    tn = np.sum((pred == False) & (gt == False))
    fn = np.sum((pred == False) & (gt == True))

    return tp, fp, tn, fn


def _binary_conf_mat():
    """Tests `binary_conf_mat()`"""
    out = binary_conf_mat(pred=np.asarray([True, False]), gt=np.asarray([True, False]))
    assert out == (1, 0, 1, 0), "Error in conf mat"

    out = binary_conf_mat(pred=np.asarray([False, True]), gt=np.asarray([True, False]))
    assert out == (0, 1, 0, 1), "Error in conf mat"

    out = binary_conf_mat(pred=np.asarray([True, True]), gt=np.asarray([True, True]))
    assert out == (2, 0, 0, 0), "Error in conf mat, out=" + str(out)


_binary_conf_mat()


def make_evaluation_dirs(dir_root: str, dataset_name: str, model_name: str):
    dir_out = os.path.join(dir_root, dataset_name, model_name)

    paths = {}
    paths["prob_segs"] = os.path.join(dir_out, "prob_segs")
    paths["pred_segs"] = os.path.join(dir_out, "pred_segs")
    paths["pred_anatomy"] = os.path.join(dir_out, "pred_anatomy")
    paths["images"] = os.path.join(dir_out, "images")
    paths["targets"] = os.path.join(dir_out, "targets")

    for key in paths:
        if not os.path.isdir(paths[key]):
            os.makedirs(paths[key])

    return paths
