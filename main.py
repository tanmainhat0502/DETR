# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import datasets
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from models import build_model
import wandb  # Thêm WandB
import torch.distributed as dist
import atexit
import os
try:
    from kaggle_secrets import UserSecretsClient
except ImportError:
    print("kaggle_secrets not available, relying on manual WandB login")

def cleanup():
    try:
        if wandb.run is not None:
            wandb.finish()
            time.sleep(5)  # giữ kết nối thêm vài giây trước khi thoát
    except Exception as e:
        print(f"Warning: Failed to finish WandB run: {e}")

atexit.register(cleanup)

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--num_classes', default=3, type=int, help='number of classes (excluding background)')
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', default=False, action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='emotic')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', action='store_true', help='enable distributed training')

    # WandB parameters
    parser.add_argument('--wandb_project', default='detr-emotic', type=str, help='WandB project name')
    parser.add_argument('--wandb_name', default=None, type=str,
                    help="Custom name for wandb run")
    return parser


def main(args):
    utils.init_distributed_mode(args)

    if utils.is_main_process():
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name if args.wandb_name else f"run-{int(time.time())}",
            config=vars(args)
        )

    device = torch.device(args.device)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)
    model_without_ddp = model
    model.to(device)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, 
            device_ids=[args.gpu], 
            find_unused_parameters=True
        )
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    try:
        wandb.config.update({"n_parameters": n_parameters})
    except Exception as e:
        print(f"Warning: Failed to log n_parameters to WandB: {e}")

    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    # Log dataset size lên WandB
    if not args.distributed or utils.is_main_process():
        try:
            wandb.config.update({
                "train_samples": len(dataset_train),
                "val_samples": len(dataset_val),
            })
        except:
            pass

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                                  collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    if args.dataset_file == 'emotic':
        from datasets import emotic
        dataset_train = emotic.build(image_set='train', args=args)
        dataset_val = emotic.build(image_set='val', args=args)
    base_ds = get_coco_api_from_dataset(dataset_val)

    output_dir = Path(args.output_dir)

    # Tải checkpoint
    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'], strict=False)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        state_dict = checkpoint['model']
        
        # Khởi tạo class_embed
        num_classes = args.num_classes
        hidden_dim = model_without_ddp.class_embed.weight.size(1)
        new_class_embed = torch.nn.Linear(hidden_dim, num_classes + 1)
        new_class_embed.to(device)
        model_without_ddp.class_embed = new_class_embed
        
        # Loại bỏ class_embed
        state_dict.pop('class_embed.weight', None)
        state_dict.pop('class_embed.bias', None)
        
        # Tải state_dict
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(state_dict, strict=False)
        print(f"Missing keys: {missing_keys}")
        print(f"Unexpected keys: {unexpected_keys}")
        
        # In danh sách
        print("Model parameters:")
        for idx, (name, param) in enumerate(model_without_ddp.named_parameters()):
            print(f"Index {idx}: {name}")
        
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    if args.eval:
        test_stats, coco_evaluator = evaluate(
            model, criterion, postprocessors,
            data_loader_val, 
            base_ds, 
            device, 
            args.output_dir
        )
        if args.output_dir and (not args.distributed or utils.is_main_process()):
            try:
                utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
                # Log kết quả
                if coco_evaluator is not None:
                    stats = coco_evaluator.coco_eval["bbox"].stats
                    wandb.log({
                        "eval/mAP@50:95": stats[0],
                        "eval/mAP@50": stats[1],
                        'eval/AP75': stats[2],
                        'eval/APs': stats[3],
                        'eval/APm': stats[4],
                        'eval/APl': stats[5],
                    })
            except Exception as e:
                print(f"Warning: Failed to log eval to results to WandB: {e}")
        return

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, 
            optimizer, 
            device, 
            epoch,
            args.clip_max_norm)
        lr_scheduler.step()
        torch.cuda.empty_cache()
        
        if args.output_dir and (not args.distributed or utils.is_main_process()):
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 100 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)
                # Log checkpoint
                try:
                    wandb.save(str(checkpoint_path))
                except:
                    pass

        test_stats, coco_evaluator = evaluate(
            model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir
        )

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            # Log eval stats lên WandB
            try:
                if coco_evaluator is not None:
                    stats = coco_evaluator.coco_eval["bbox"].stats
                    wandb.log({
                        "epoch": epoch,
                        "train/loss": train_stats.get("loss", 0.0),
                        "train/loss_ce": train_stats.get("loss_ce", 0.0),
                        "train/loss_bbox": train_stats.get("loss_bbox", 0.0),
                        "train/loss_giou": train_stats.get("loss_giou", 0.0),
                        "eval/AP": stats[0],
                        "eval/AP50": stats[1],
                        "eval/AP75": stats[2],
                        "eval/APs": stats[3],
                        "eval/APm": stats[4],
                        "eval/APl": stats[5],
                    })
            except Exception as e:
                print(f"Warning: Failed to log eval stats to WandB: {e}")

            # for evaluation logs
            if coco_evaluator is not None:
                (output_dir / 'eval').mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ['latest.pth']
                    if epoch % 50 == 0:
                        filenames.append(f'{epoch:03d}.pth')
                    for name in filenames:
                        torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                   output_dir / "eval" / name)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    try:
        wandb.log({"training_time_seconds": int(total_time)})
    except Exception as e:
        print(f"Warning: Failed to log training time to WandB: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    n_gpu = torch.cuda.device_count()
    if n_gpu > 1:
        args.distributed = True
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in range(n_gpu))
    else:
        args.distributed = False
        args.gpu = 0

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
