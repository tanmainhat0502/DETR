python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py \
    --coco_path /kaggle/input/detr-dataset/EmoticGender \
    --output_dir /kaggle/working/output \
    --batch_size 8 \
    --epochs 10 \
    --resume /kaggle/working/detr-r101-dc5-a2e86def.pth \
    --dataset_file emotic \