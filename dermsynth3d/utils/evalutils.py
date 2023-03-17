import os


def make_evaluation_dirs(dir_root: str, model_name: str):
    paths = {}
    paths["prob_segs"] = os.path.join(dir_root, model_name, "prob_segs")
    paths["pred_segs"] = os.path.join(dir_root, model_name, "pred_segs")
    paths["pred_anatomy"] = os.path.join(dir_root, model_name, "pred_anatomy")
    paths["images"] = os.path.join(dir_root, model_name, "images")
    paths["targets"] = os.path.join(dir_root, model_name, "targets")

    for key in paths:
        if not os.path.isdir(paths[key]):
            os.makedirs(paths[key])

    return paths
