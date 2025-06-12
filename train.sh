python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py \
--coco_path /kaggle/input/detr-dataset/EmoticGender \
--output_dir /kaggle/working/output \
--batch_size 4 \
--epochs 10 \
--resume /kaggle/working/detr-r101-panoptic-40021d53.pth