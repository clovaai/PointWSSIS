"""
PointWSSIS (https://arxiv.org/abs/2303.15062)
Copyright (c) 2023-present NAVER Cloud Corp.
Apache-2.0
"""

import os
import warnings
warnings.filterwarnings(action="ignore")

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import random
import argparse
import numpy as np
import time
import itertools
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils import data
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from network import UNet_ResNet101
from datasets import dataset_factory
from utils import AverageMeter, optim_factory, save_checkpoint, tensorboard_report
from utils.loss import DiceLoss
from utils.distributed import gather

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pycocotools.mask as mask_util

torch.backends.cudnn.benchmark = True

NUM_CLASSES = {'coco': 80, 'BDD100K': 8}

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

def get_argparser():
    parser = argparse.ArgumentParser()

    # Path option
    parser.add_argument("--data_root", type=str, default="/mnt/tmp", help="path to Dataset")
    parser.add_argument("--workspace", type=str, default="results", help="workspace path")
    parser.add_argument("--exp_name", type=str, default="segmentation", help="experiment name")
    parser.add_argument("--dataset", type=str, default='coco', choices=["coco", "BDD100K"])

    # Deeplab Options
    parser.add_argument("--num_classes", type=int, default=80, help="80 for coco, 8 for BDD100K")
    parser.add_argument("--box_scale", type=float, default=2.0)
    
    parser.add_argument("--weak_pth", nargs='+', type=str)
    parser.add_argument("--gt_json", type=str)
    parser.add_argument("--eval_pth", type=str)
    parser.add_argument("--eval_json", type=str, default="datasets/coco/instances_train2017_refine_test_1K.json")
    
    # Train Options
    parser.add_argument("--ddp", action="store_true", default=False)
    parser.add_argument("--amp", action="store_true", default=False)
    parser.add_argument("--test_only", action="store_true", default=False)
    parser.add_argument("--train_iters", type=int, default=0, help="total training iteration")
    parser.add_argument("--warm_iters", type=int, default=0, help="warm-up iterations")

    parser.add_argument("--optim", type=str, default="adam", choices=["sgd", "adam"])
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate (default: 1e-4)")
    parser.add_argument("--lr_policy", type=str, default="cosine", help="learning rate scheduler policy")
    parser.add_argument("--weight_decay", type=float, default=0, help="weight decay")

    parser.add_argument("--batch_size", type=int, default=16, help="batch size (default: 16)")
    parser.add_argument("--train_size", type=str, default="256", help="input size for training (support multi-scale sizes)")
    parser.add_argument("--val_size", type=str, default="256", help="input size for validation (support multi-scale sizes)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--restore", default='none', type=str, help="restore from checkpoint")
    parser.add_argument("--ckpt_dir", default="ckpts", type=str, help="save dir")
    
    parser.add_argument("--aug_min_scale", type=float, default=0.5, help="min scale for random scale augmentation")
    parser.add_argument("--aug_max_scale", type=float, default=1.5, help="max scale for random scale augmentation")
    parser.add_argument("--aug_color", type=float, default=0.0, help="prob for random color jiterring augmentation")

    parser.add_argument("--random_seed", type=int, default=3407, help="random seed (default: 2)")
    parser.add_argument("--print_interval", type=int, default=100, help="print interval of loss (default: 10)")
    parser.add_argument("--val_interval", type=int, default=1000, help="epoch interval for eval (default: 100)")

    # DDP Options
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--local_rank", type=int, default=int(os.getenv("LOCAL_RANK", "0")))

    return parser


def print_func(string, rank):
    if rank == 0:
        print(string)


def train():
    # ==========   Train Loop   ==========#
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    avg_loss = AverageMeter(args.ddp)
    data_time = AverageMeter(args.ddp)
    avg_time = AverageMeter(args.ddp)

    # switch to train mode
    model.train()
    start = time.time()
    end = time.time()
    epoch = 0
    cur_itrs = 0

    # best score dict initialize
    best_AP = 0

    while cur_itrs < args.train_iters:
        try:
            data_end = time.time()
            dat = next(data_iter)
            data_time.update(time.time() - data_end)
        except:
            epoch += 1
            data_iter = iter(train_loader)
            dat = next(data_iter)
            avg_loss.reset()
            data_time.reset()
            avg_time.reset()

            if args.local_rank == 0:
                save_path = f"{args.ckpt_dir}/last.pt"
                save_checkpoint(save_path, model, epoch, cur_itrs, best_AP, args.ddp)

        cur_itrs += 1

        images = dat['img'].to(device, non_blocking=True)
        labels = dat['target'].to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=args.amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        avg_time.update(time.time()-end)
        avg_loss.update(loss.item(), images.size(0))

        if cur_itrs % args.print_interval == 0:
            data_time.synch(device)
            avg_time.synch(device)
            avg_loss.synch(device)

            if args.local_rank == 0:
                print(
                    "Progress: [{0}][{1}/{2}] ({3:.1f}%, {4:.1f} min) | "
                    "Data Time: {5:.1f} ms | "
                    "Epoch Time: {6:.1f} ms | "
                    "Left: {7:.1f} min | "
                    "LR: {8:.8f} | "
                    "Loss: {avg_loss.avg:.6}".format(
                        epoch, cur_itrs, args.train_iters, cur_itrs / args.train_iters * 100, (end - start) / 60,
                        data_time.avg * 1000, avg_time.avg * 1000, (args.train_iters - cur_itrs) * avg_time.avg / 60,
                        optimizer.param_groups[0]["lr"], avg_loss=avg_loss)
                )

                report_dict = dict()
                report_dict["optim/progress"] = float(cur_itrs / args.train_iters)
                report_dict["optim/epoch"] = epoch
                report_dict["optim/learning_rate"] = optimizer.param_groups[-1]["lr"]
                report_dict["optim/data_time"] = data_time.avg * 1000
                report_dict["optim/epoch_time"] = avg_time.avg * 1000
                report_dict["optim/left"] = ((args.train_iters - cur_itrs) * avg_time.avg / 60)
                report_dict["train/loss"] = avg_loss.avg
                
                tensorboard_report(writer, report_dict, cur_itrs)

        if (cur_itrs) % args.val_interval == 0:
            print_func("validation...", rank=args.local_rank)
            model.eval()

            if args.ddp:
                torch.distributed.barrier()

            val_score = validate()

            if args.local_rank == 0:
                report_dict = dict()

                for k, v in val_score.items():
                    if type(v) is not dict:
                        report_dict[k] = float(v)

                tensorboard_report(writer, report_dict, cur_itrs)

                if val_score["val/AP"] >= best_AP:
                    best_AP = val_score["val/AP"]
                    save_path = f"{args.ckpt_dir}/best_AP.pt"
                    save_checkpoint(save_path, model, epoch, cur_itrs, best_AP, args.ddp)
                    print("Best COCO AP Model saved: %.8f" % (best_AP))

            model.train()

        end = time.time()


def validate():
    """Do validation and return specified samples"""
    score = {}
    _annotations = []

    if args.local_rank == 0:
        progress_bar = tqdm(total=len(valid_loader))

    with torch.no_grad():
        for inputs, meta in valid_loader:
            if args.local_rank == 0:
                progress_bar.update(1)
                
            for n in range(len(inputs)):
                meta[n]['image_id'] = int(meta[n]['image_id'])
                meta[n]['category_id'] = int(meta[n]['category_id'])
                meta[n]['bbox'] = list(map(int, meta[n]['bbox']))
                meta[n]['crop_bbox'] = list(map(int, meta[n]['crop_bbox']))
                meta[n]['ori_size'] = list(map(int, meta[n]['ori_size']))
                meta[n]['segmentation'] = meta[n]['segmentation'][0].numpy()

                if meta[n]['ori_size'] == [0, 0]:
                    refined_mask = meta[n]['segmentation']
                else:
                    out = model(inputs[n].to(device))
                    out = torch.softmax(out, 1)
                    out = F.interpolate(out, size=meta[n]['ori_size'], mode="bilinear", align_corners=False)
                    out = out.argmax(1)[0].detach().cpu().numpy()
                    out = (out > 0).astype(np.uint8)

                    refined_mask = np.zeros_like(meta[n]['segmentation'])
                    refined_mask[meta[n]['crop_bbox'][1]:meta[n]['crop_bbox'][3]+1, 
                                 meta[n]['crop_bbox'][0]:meta[n]['crop_bbox'][2]+1] = out
                    
                    if refined_mask.sum() > 0:
                        y_coord, x_coord = refined_mask.nonzero()
                        ymin, xmin = int(y_coord.min()), int(x_coord.min())
                        ymax, xmax = int(y_coord.max()), int(x_coord.max())
                        meta[n]['bbox'] = [xmin, ymin, xmax-xmin, ymax-ymin]
                    else:
                        refined_mask = meta[n]['segmentation']

                refined_mask_rle = mask_util.encode(np.asfortranarray(refined_mask))
                refined_mask_rle['counts'] = refined_mask_rle['counts'].decode('ascii') 
                
                meta[n]['segmentation'] = refined_mask_rle
                meta[n]['area'] = int(np.sum(refined_mask))
                meta[n]['iscrowd'] = False
                meta[n]['id'] = 0
                meta[n]['score'] = 1.0

                del meta[n]['ori_size']
                del meta[n]['crop_bbox']
                _annotations.append(meta[n])

    annotations = gather(_annotations, dst=0)
    annotations = list(itertools.chain(*annotations))

    if args.local_rank == 0:
        progress_bar.close()
        
        for inst_id, anno in enumerate(annotations):
            anno['id'] = inst_id+1
                
        pred_annotations = COCO()
        pred_annotations.dataset['categories'] = valid_dst.gt_annotations.dataset['categories']
        pred_annotations.dataset['annotations'] = annotations
        pred_annotations.dataset['images'] = valid_dst.gt_annotations.dataset['images']
        pred_annotations.createIndex()

        print('\nEvaluating Refined Masks:')
        coco_eval = COCOeval(valid_dst.gt_annotations, pred_annotations, 'segm')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        score['val/AP'] = coco_eval.stats[0]
        score['val/AP50'] = coco_eval.stats[1]
        score['val/AP75'] = coco_eval.stats[2]
        score['val/AP_S'] = coco_eval.stats[3]
        score['val/AP_M'] = coco_eval.stats[4]
        score['val/AP_L'] = coco_eval.stats[5]
        score['val/AR100'] = coco_eval.stats[7]

    return score


if __name__ == "__main__":

    args = get_argparser().parse_args()

    # Setup random seed
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    session_dir = os.path.join(args.workspace, args.exp_name)
    args.ckpt_dir = os.path.join(session_dir, "ckpt")

    if args.local_rank == 0:
        os.makedirs(session_dir, exist_ok=True, mode=0o777)
        os.makedirs(args.ckpt_dir, exist_ok=True, mode=0o777)
        writer = SummaryWriter(log_dir=session_dir)

    args.gpu = args.local_rank
    torch.cuda.set_device(args.gpu)
    n_gpus = torch.cuda.device_count()
    device = torch.device(f"cuda:{args.gpu}")

    args.ddp = True if n_gpus > 1 else False

    # Init dirstributed system
    if args.ddp:
        torch.distributed.init_process_group(backend="nccl", rank=args.local_rank, world_size=n_gpus)
        args.world_size = torch.distributed.get_world_size()

    batch_per_gpu_train = args.batch_size // n_gpus
    
    # str -> integer list
    args.train_size = list(map(int, args.train_size.split(","))) if len(args.train_size.split(",")) > 1 else int(args.train_size)
    args.val_size = list(map(int, args.val_size.split(","))) if len(args.val_size.split(",")) > 1 else int(args.val_size)
    
    # Setup dataloader
    train_dst, valid_dst = dataset_factory(args)

    train_loader = data.DataLoader(
        train_dst,
        batch_size=batch_per_gpu_train,
        num_workers=args.workers,
        sampler=DistributedSampler(train_dst, num_replicas=n_gpus, rank=args.local_rank) if args.ddp else None, 
        shuffle=False if args.ddp else True,
        pin_memory=True,
        drop_last=True,
    )

    valid_loader = data.DataLoader(
        valid_dst,
        batch_size=1,
        num_workers=0,
        sampler=DistributedSampler(valid_dst, num_replicas=n_gpus, rank=args.local_rank) if args.ddp else None, 
        pin_memory=True,
        drop_last=False,
    )

    args.num_classes = NUM_CLASSES[args.dataset]
    print_func(f"LOG | Dataset | {train_dst} | {valid_dst} ", args.local_rank)
    print_func("LOG | Num_Classes: %d, Train Iters: %d, Warm Iters: %d"
        % (args.num_classes, args.train_iters, args.warm_iters), args.local_rank)

    # Set up model
    # input channel
        #   RGB image (3-channel)
        #   prior mask (1-channel)
        #   center map (num_classes+1 channel)
    
    model = UNet_ResNet101(
        output_channel=2,
        input_channel=3+1+args.num_classes+1,
    )
    model.to(device)

    optimizer, scheduler = optim_factory(
        args,
        model.parameters(),
        args.train_iters,
        args.warm_iters,
    )

    # Set up criterion
    criterion = DiceLoss()
    
    # Restore
    if os.path.exists(args.restore):
        checkpoint = torch.load(args.restore, map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint["model_state"], strict=True)

        print_func("LOG: Model restored from %s" % args.restore, rank=args.local_rank)
        del checkpoint  # free memory
        # model.head_init()
    else:
        print_func("LOG: Training without restoring weight: %s" % args.restore, rank=args.local_rank)

    # DDP
    if args.ddp:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(
            model,
            device_ids=[args.gpu],
            output_device=args.gpu,
        )
        torch.distributed.barrier()

    print_func("LOG: start training...", rank=args.local_rank)
    print_func(args, rank=args.local_rank)

    train()
    