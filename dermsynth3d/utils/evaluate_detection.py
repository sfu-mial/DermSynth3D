import torch
from dermsynth3d.utils.object_detection_metrics.BoundingBox import BoundingBox
from dermsynth3d.utils.object_detection_metrics.BoundingBoxes import BoundingBoxes
from dermsynth3d.utils.object_detection_metrics.utils import (
    BBFormat,
    BBType,
    CoordinatesType,
)
from dermsynth3d.utils.object_detection_metrics.Evaluator import Evaluator


@torch.no_grad()
def evaluate_detection(model, val_dataloader, device):
    model.eval()
    cpu_device = torch.device("cpu")
    centroid_results = []
    iou_results = []

    for _, ids, images, masks, targets in val_dataloader:
        for id, img, mask, target in zip(ids, images, masks, targets):
            gt_boxes = target["boxes"].to(cpu_device)
            if len(gt_boxes) == 0:
                continue

            image = img.to(device)

            output = model([image])[0]

            predicted_boxes = output["boxes"].to(cpu_device)
            predicted_scores = output["scores"].to(cpu_device)

            centroid_result, iou_result = get_results(
                id, predicted_boxes, predicted_scores, gt_boxes
            )

            centroid_results.append(centroid_result)
            iou_results.append(iou_result)

    return centroid_results, iou_results


def get_results(scan_id, predicted_boxes, predicted_scores, gt_boxes):
    centroid_result = sample_results(
        predicted_boxes,
        predicted_scores,
        gt_boxes,
        scan_id,
        decision_func=both_contain_centroids,  # Centroid matching.
        iou_thresh=None,  # None since does not use IoU.
    )

    # We repeat to compute results using the IoU metric.
    # We compute since IoU is traditionally used to determine a "match".
    # These are the same predictions just a different way of determining what
    # a correct "match" is between the predicted and GT.
    iou_result = sample_results(
        predicted_boxes,
        predicted_scores,
        gt_boxes,
        scan_id,
        decision_func=None,  # Defaults to IoU.
        iou_thresh=0.5,  # IoU threshold 0.5 to indicate a match.
    )

    return centroid_result, iou_result


def sample_results(
    predicted_boxes, predicted_scores, gt_boxes, scan_id, decision_func, iou_thresh
):
    # Bounding boxes with a confidence threshold >= 0.5
    bounding_boxes_thresh = convert_to_bounding_boxes(
        scan_id,
        gt_boxes,
        predicted_boxes,
        predicted_scores,
        score_thresh=0.5,
    )

    # All bounding boxes (no threshold on confidence)
    bounding_boxes_nothresh = convert_to_bounding_boxes(
        scan_id,
        gt_boxes,
        predicted_boxes,
        predicted_scores,
        score_thresh=0,
    )

    # Class to compute metrics.
    # Mainly used since average precision
    # of bounding boxes is non-trivial to compute.
    evaluator = Evaluator()

    # Results with predicted bounding boxes' scores >= 0.5
    # The score is needed to make a decision of a detection,
    # so we can compute metrics like recall, precision.
    results_thresh_all = evaluator.GetDetection(
        bounding_boxes_thresh,
        IOUThreshold=iou_thresh,
        decision_func=decision_func,  # Matching centroids.
    )

    thresh_results = summarize_results(
        results_thresh_all,
        scan_id=scan_id,
    )

    # Here we compute the results using
    # all the predicted boxes (i.e., not threshold based on score).
    # We do this to compute average precision since
    # AP is computed over a range of thresholds.
    results_nothresh_all = evaluator.GetDetection(
        bounding_boxes_nothresh,
        IOUThreshold=iou_thresh,
        decision_func=decision_func,
    )

    results_nothresh = summarize_results(
        results_nothresh_all,
        scan_id=scan_id,
    )

    # We form our final results by using the thresholded score results
    # except for AP, which we use from the non-thresholded results.
    results = thresh_results.copy()
    results["ap"] = results_nothresh["ap"]

    return results


def summarize_results(sample_results, scan_id=None):
    TP = sample_results[0]["total TP"]
    FP = sample_results[0]["total FP"]
    FN = sample_results[0]["total positives"] - sample_results[0]["total TP"]
    recall_score = TP / (TP + FN)
    precision_score = TP / (TP + FP)

    result = {
        "scan_id": scan_id,
        "tp": TP,
        "fp": FP,
        "fn": FN,
        "recall": recall_score,
        "precision": precision_score,
        "ap": sample_results[0]["AP"],
        "iou": sample_results[0]["iou"],
    }
    return result


def centroid(box):
    """Assumes `box` is in the format [x,y,x1,y1]"""
    x_center = (box[0] + box[2]) / 2
    y_center = (box[1] + box[3]) / 2
    return x_center, y_center


def source_contains_target_centroid(source_box, target_box):
    target_x, target_y = centroid(target_box)
    contains_center = (
        (target_x > source_box[0])
        & (target_x < source_box[2])
        & (target_y > source_box[1])
        & (target_y < source_box[3])
    )
    return contains_center


def both_contain_centroids(box_a, box_b):
    """Returns `True` if the centroid of `box_a` is within `box_b` and vice versa."""
    return source_contains_target_centroid(
        box_a, box_b
    ) and source_contains_target_centroid(box_b, box_a)


def bounding_box(scan_id, box, score=None, is_gt=True):
    if is_gt:
        bbType = BBType.GroundTruth
    else:
        bbType = BBType.Detected

    bb = BoundingBox(
        imageName=scan_id,
        classId="lesion",
        classConfidence=score,
        x=box[0],
        y=box[1],
        w=box[2],
        h=box[3],
        typeCoordinates=CoordinatesType.Absolute,
        bbType=bbType,
        format=BBFormat.XYX2Y2,
    )

    return bb


def convert_to_bounding_boxes(
    scan_id,
    gt_boxes,
    predicted_boxes,
    predicted_scores,
    score_thresh,
):
    bounding_boxes = BoundingBoxes()

    for gt_box in gt_boxes:
        bb = bounding_box(scan_id=scan_id, box=gt_box, score=None, is_gt=True)
        bounding_boxes.addBoundingBox(bb)

    for pred_score, pred_box in zip(predicted_scores, predicted_boxes):
        if pred_score >= score_thresh:
            bb = bounding_box(
                scan_id=scan_id,
                box=pred_box,
                score=pred_score,
                is_gt=False,
            )
            bounding_boxes.addBoundingBox(bb)

    return bounding_boxes
