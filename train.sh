python main.py \
    --coco_path /root/DETR/data4training \
    --output_dir run \
    --batch_size 4 \
    --epochs 1 \
    --resume /root/DETR/DETR/detr-r50-e632da11.pth \
    --dataset_file emotic \
    --num_classes 3 \
    --wandb_project detr-emotic \
    --wandb_entity tanmnse170507-fpt-university
