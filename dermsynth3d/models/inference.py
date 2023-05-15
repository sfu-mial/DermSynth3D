import os
import numpy as np
from PIL import Image
from tqdm.autonotebook import tqdm
import torch
from dermsynth3d.utils.image import float_img_to_uint8
from dermsynth3d.utils.evaluate import argmax_predictions


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
    dir_save_skin_probs=None,
):
    softmax2d = torch.nn.Softmax2d()

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
                prob_seg = Image.fromarray(seg_pred)
                orig_img = dataloader.dataset.image(file_id)
                prob_seg = prob_seg.resize(orig_img.size)

                pred_seg = argmax_predictions(seg_pred, asint=True)
                pred_seg = Image.fromarray(pred_seg)
                pred_seg = pred_seg.resize(orig_img.size)

                if dir_save_skin_probs is not None:
                    prob_seg.save(
                        os.path.join(
                            dir_save_skin_probs,
                            file_id + dataloader.dataset.target_extension,
                        )
                    )

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
