import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

from dermsynth3d.utils.utils import get_logger
from dermsynth3d.utils.image import float_img_to_uint8
from dermsynth3d.losses.metrics import dice, precision, recall, jaccard_index
from dermsynth3d.datasets.datasets import ImageDataset


def infer(model, path, logger, test_dataloader, device, real_test_ds, real_paths):
    sigmoid = torch.nn.Sigmoid().to(device)
    # do inference after every best model
    model.eval()
    dir_name = path.split("/")[-2]
    model_name = path.split("/")[-1][:-3]
    os.makedirs(os.path.join(dir_name, "predictions"), exist_ok=True)
    dir_preds = os.path.join(dir_name, "predictions", model_name)
    os.makedirs(dir_preds, exist_ok=True)
    pred_vals = {}
    median_vals = []
    fitz_vals = []
    image_ids = []
    failed = []
    logger.info("Testing===>")
    with torch.no_grad():
        count = 0
        for _, file_ids, pre_imgs, segs in tqdm(test_dataloader):
            pre_imgs = pre_imgs.to(device)
            # Forward pass.
            pred_segs = model(pre_imgs)["out"]
            pred_segs = sigmoid(pred_segs)
            pred_segs_np = np.asarray(pred_segs.cpu().detach().numpy())
            pred_segs_np = np.rollaxis(pred_segs_np, 1, 4)
            pred_segs_np = float_img_to_uint8(pred_segs_np)
            for pred_seg, file_id in zip(pred_segs_np, file_ids):
                pred_seg = Image.fromarray(pred_seg[:, :, 0])
                orig_img = real_test_ds.image(file_id)
                pred_seg = pred_seg.resize(orig_img.size)
                pred_seg.save(os.path.join(dir_preds, file_id + ".png"))
            count += 1

    wound_ds = ImageDataset(
        real_paths[0],
        real_paths[1],
        dir_preds,
        image_extension=",png",
        target_extension=".png",
    )
    results = []
    no_gt = []
    for image_id in wound_ds.file_ids:
        pred_seg = wound_ds.prediction(image_id)
        pred_lesion_seg = (np.asarray(pred_seg)[:, :, 0] / 255) > 0.5
        gt_seg = (np.asarray(wound_ds.target(image_id))[:, :, 0] / 255) > 0
        if gt_seg.sum() == 0:
            no_gt.append(image_id)
        else:
            # js = jaccard_score(gt_seg.flatten(), pred_lesion_seg.flatten())
            di = dice(gt_seg, pred_lesion_seg)
            ji = jaccard_index(gt_seg, pred_lesion_seg)
            pi = precision(gt_seg, pred_lesion_seg)
            ri = recall(gt_seg, pred_lesion_seg)
            results.append(
                {
                    "image_id": image_id,
                    "jaccard_index": ji,
                    "dice_score": di,
                    "precision": pi,
                    "recall": ri,
                }
            )
    results_df = pd.DataFrame(results)
    js = np.sum(results_df["jaccard_index"] > 0)
    ds = results_df["dice_score"].mean()
    prec = results_df["precision"].mean()
    rec = results_df["recall"].mean()
    logger.info(
        f"Jaccard Index: {js} | Precision: {prec} | Recall: {rec} | Dice Score: {ds}"
    )
    logger.info("Testing Done. Predictions Saved at -->", dir_preds)


from dermsynth3d.losses.metrics import compute_results_segmentation


def evaluate(
    seg_model,
    val_dataloader,
    device,
    real_val_ds,
    save_to_disk=False,
    save_dir=None,
    writer=None,
    DA=False,
):
    seg_model.eval()
    sigmoid = torch.nn.Sigmoid().to(device)
    with torch.no_grad():
        count = 0
        # val_loss=0.
        results = []
        val_score = []
        from tqdm import tqdm

        for dataset_types, image_ids, images, segs in tqdm(val_dataloader):
            images = images.to(device)
            segs = segs.to(device)
            if not DA:
                out = seg_model(images)["out"]
            else:
                out = seg_model(images, source=True)
            out = sigmoid(out)
            out_np = out.cpu().detach().numpy().squeeze()
            out_np = np.rollaxis(out_np, 1, -1)
            out_np = float_img_to_uint8(out_np)

            file_id = image_ids[0]
            pred_seg = Image.fromarray(out_np)
            pred_seg = pred_seg.resize(val_dataloader.dataset.image(file_id).size)

            gt_seg = np.asarray(real_val_ds.target(file_id))
            image_res = {}
            image_res["file_id"] = file_id

            if save_to_disk:
                # Batch size of 1 so take the first one.
                os.makedirs(save_dir, exist_ok=True)
                pred_seg.save(os.path.join(save_dir, file_id + ".png"))
            # breakpoint()
            pred_channel = (np.asarray(pred_seg)[:, :] / 255.0) > 0.5
            gt_channel = (np.asarray(gt_seg)[:, :, 0] / 255.0) > 0

            iou = jaccard_index(gt_channel, pred_channel)
            dice_score = dice(gt_channel, pred_channel)

            image_res["iou"] = iou
            image_res["dice"] = dice_score
            results.append(image_res)

        if save_to_disk:
            print("***** Preds saved at----> ", save_dir)

        return pd.DataFrame(results)


def inference_multitask(
    max_imgs,
    model,
    dataloader,
    device,
    save_to_disk=False,
    return_values=True,
    dir_anatomy_preds=None,
    dir_save_images=None,
    dir_save_targets=None,
    dir_save_skin_preds=None,
):
    softmax2d = torch.nn.Softmax2d().to(device)

    if dataloader.batch_size > 1:
        raise ValueError("Batch size must be set to 1.")

    track_file_ids = []
    seg_preds = []
    anatomy_preds = []

    cnt = 0

    with torch.no_grad():
        for _, file_ids, images, _ in tqdm(dataloader):
            if cnt >= max_imgs:
                break

            images = images.to(device)
            out = model(images)
            seg_pred = out["segmenter"]
            seg_pred = softmax2d(seg_pred).cpu().detach().numpy().squeeze()
            seg_pred = np.moveaxis(seg_pred, 0, -1)

            anatomy_pred = out["anatomy"]
            anatomy_pred = softmax2d(anatomy_pred).cpu().detach().numpy().squeeze()

            if return_values:
                track_file_ids.append(file_ids)
                seg_preds.append(seg_pred)
                anatomy_preds.append(anatomy_pred)

            if save_to_disk:
                # Batch size of 1 so take the first one.
                file_id = file_ids[0]
                seg_pred = float_img_to_uint8(seg_pred)
                pred_seg = Image.fromarray(seg_pred)
                orig_img = dataloader.dataset.image(file_id)
                pred_seg = pred_seg.resize(orig_img.size)

                if dir_save_skin_preds is not None:
                    pred_seg.save(
                        os.path.join(
                            dir_save_skin_preds,
                            file_id + dataloader.dataset.target_extension,
                        )
                    )

                if dir_anatomy_preds is not None:
                    np.savez_compressed(
                        os.path.join(dir_anatomy_preds, file_id + ".npz"),
                        anatomy=anatomy_pred,
                    )

                if dir_save_images is not None:
                    orig_img.save(
                        os.path.join(
                            dir_save_images,
                            file_id + dataloader.dataset.image_extension,
                        )
                    )

                if dir_save_targets is not None:
                    target = dataloader.dataset.target(file_id)
                    if type(target) == np.ndarray:
                        target = float_img_to_uint8(target)
                        target = Image.fromarray(target)
                    target.save(
                        os.path.join(
                            dir_save_targets,
                            file_id + dataloader.dataset.target_extension,
                        )
                    )

            cnt += 1

    return {
        "file_ids": track_file_ids,
        "segs": seg_preds,
        "anatomy": anatomy_preds,
    }


def inference_segmentation(
    max_imgs,
    model,
    dataloader,
    device,
    save_to_disk=False,
    return_values=True,
    dir_save_seg_preds=None,
):
    sigmoid = torch.nn.Sigmoid()

    if dataloader.batch_size > 1:
        raise ValueError("Batch size must be set to 1.")

    track_file_ids = []
    seg_preds = []

    cnt = 0

    with torch.no_grad():
        for _, file_ids, images, _ in tqdm(dataloader):
            if cnt >= max_imgs:
                break

            images = images.to(device)
            out = model(images)
            seg_pred = out["out"]
            seg_pred = sigmoid(seg_pred).cpu().detach().numpy().squeeze()
            seg_pred = np.rollaxis(seg_pred, 1, -1)

            if return_values:
                track_file_ids.append(file_ids)
                seg_preds.append(seg_pred)

            if save_to_disk:
                # Batch size of 1 so take the first one.
                file_id = file_ids[0]
                seg_pred = float_img_to_uint8(seg_pred)
                pred_seg = Image.fromarray(seg_pred)
                orig_img = dataloader.dataset.image(file_id)
                pred_seg = pred_seg.resize(orig_img.size)

                if dir_save_seg_preds is not None:
                    pred_seg.save(
                        os.path.join(
                            dir_save_seg_preds,
                            file_id + dataloader.dataset.target_extension,
                        )
                    )

            cnt += 1

    return {
        "file_ids": track_file_ids,
        "segs": seg_preds,
    }
