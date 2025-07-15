from pathlib import Path
from .coco import CocoDetection, make_coco_transforms
import os

def build(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    PATHS = {
        "train": (os.path.join(root, "train2017"), os.path.join(root, "annotations", "instances_train2017.json")),
        "val": (os.path.join(root, "val2017"), os.path.join(root, "annotations", "instances_val2017.json")),
    }
    img_folder, ann_file = PATHS[image_set]
    assert img_folder.exists(), f'Image folder {img_folder} does not exist'
    assert ann_file.exists(), f'Annotation file {ann_file} does not exist'
    dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(image_set), return_masks=args.masks)
    return dataset
