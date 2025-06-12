import os
import json
import glob
from PIL import Image

def convert_yolo_to_coco(image_dir, label_dir, class_list, output_json):
    categories = [{"id": i, "name": name} for i, name in enumerate(class_list)]
    images, annotations = [], []
    ann_id = 0
    image_id = 0

    for img_file in sorted(glob.glob(os.path.join(image_dir, "*.jpg"))):
        img = Image.open(img_file)
        w, h = img.size
        file_name = os.path.basename(img_file)
        image_entry = {
            "id": image_id,
            "file_name": file_name,
            "width": w,
            "height": h
        }
        images.append(image_entry)

        label_path = os.path.join(label_dir, os.path.splitext(file_name)[0] + ".txt")
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f.readlines():
                    cls, xc, yc, bw, bh = map(float, line.strip().split())
                    x = (xc - bw / 2) * w
                    y = (yc - bh / 2) * h
                    width = bw * w
                    height = bh * h
                    annotations.append({
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": int(cls),
                        "bbox": [x, y, width, height],
                        "area": width * height,
                        "iscrowd": 0
                    })
                    ann_id += 1
        image_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    with open(output_json, "w") as f:
        json.dump(coco, f, indent=2)
    print(f"✅ Saved COCO annotation to {output_json}")

# Example usage:
if __name__ == "__main__":
    convert_yolo_to_coco(
        image_dir="/home/nhattan05022003/coding/Tien_project/DO_AN/detr/data4train/EmoticGender/images/test",
        label_dir="/home/nhattan05022003/coding/Tien_project/DO_AN/detr/data4train/EmoticGender/labels/test",
        class_list=["female", "male", "unknown"],
        output_json="/home/nhattan05022003/coding/Tien_project/DO_AN/detr/data4train/EmoticGender/json_folder/emotic_test.json"
    )
