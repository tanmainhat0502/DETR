from pathlib import Path
from .coco import CocoDetection, make_coco_transforms

def build(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided path {root} does not exist'
    PATHS = {
        "train": (root / "train", root / "annotations" / f"{image_set}.json"),
        "val": (root / "val", root / "annotations" / f"{image_set}.json"),
    }
    img_folder, ann_file = PATHS[image_set]
    dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(image_set), return_masks=False)
    return dataset