from pathlib import Path
from .coco import CocoDetection, make_coco_transforms

def build(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    PATHS = {
        "train": (root / "images" / "train", root / "json_folder" / "emotic_train.json"),
        "val": (root / "images" / "val", root / "json_folder" / "emotic_val.json"),
        "test": (root / "images" / "test", root / "json_folder" / "emotic_test.json"),
    }
    img_folder, ann_file = PATHS[image_set]
    assert img_folder.exists(), f'Image folder {img_folder} does not exist'
    assert ann_file.exists(), f'Annotation file {ann_file} does not exist'
    dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(image_set), return_masks=args.masks)
    return dataset
