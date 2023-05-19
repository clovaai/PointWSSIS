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

import argparse
import numpy as np
import torch
import json
import itertools
from tqdm import tqdm

import torch.utils.data
import torch.utils.data.distributed
from torch.utils import data
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

import pycocotools.mask as mask_util
from torch.nn.functional import interpolate

from network import UNet_ResNet101
from datasets import ValidSet
from utils.distributed import gather

torch.backends.cudnn.benchmark = True

NUM_CLASSES = {'coco': 80, 'BDD100K': 8}

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

def get_argparser():
    parser = argparse.ArgumentParser()

    # Path option
    parser.add_argument("--data_root", type=str, default="/mnt/tmp", help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='coco', choices=["coco", "BDD100K"])
    parser.add_argument("--weak_pth", type=str, default="higher_hrnet48", help="model name")
    parser.add_argument("--weak_json", type=str, default=None, help="model name")
    parser.add_argument("--strong_json", type=str, default=None, help="model name")
    parser.add_argument("--save_path", type=str, default=None, help="model name")

    # Deeplab Options
    parser.add_argument("--num_classes", type=int, default=80, help="80 for coco, 8 for BDD100K")
    parser.add_argument("--size", type=int, default=256, help='input tensor size')
    parser.add_argument("--size_cut", type=int, default=0)
    parser.add_argument("--box_scale", type=float, default=2.0)

    # Train Options
    parser.add_argument("--ckpt", default='none', type=str, help="restore from checkpoint")
    
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--local_rank", type=int, default=int(os.getenv("LOCAL_RANK", "0")))

    return parser

def inference():
    """Do validation and return specified samples"""
    model.eval()
    _annotations = []
    
    with torch.no_grad():
        for inputs, meta in tqdm(valid_loader):
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
                    out = model(inputs[n].to(device, non_blocking=True))
                    out = torch.softmax(out, 1)
                    out = interpolate(out, size=meta[n]['ori_size'], mode="bilinear", align_corners=False)
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
                del meta[n]['ori_size']
                del meta[n]['crop_bbox']
                _annotations.append(meta[n])
                
    annotations = gather(_annotations, dst=0)
    annotations = list(itertools.chain(*annotations))

    return annotations


if __name__ == "__main__":

    args = get_argparser().parse_args()
    
    args.gpu = args.local_rank
    torch.cuda.set_device(args.gpu)
    n_gpus = torch.cuda.device_count()
    device = torch.device(f"cuda:{args.gpu}")

    args.ddp = True if n_gpus > 1 else False

    # Init dirstributed system
    if args.ddp:
        torch.distributed.init_process_group(backend="nccl", rank=args.local_rank, world_size=n_gpus)
        args.world_size = torch.distributed.get_world_size()
        
    args.num_classes = NUM_CLASSES[args.dataset]

    valid_dst = ValidSet(
        root=args.data_root, 
        dataset=args.dataset,
        num_classes=args.num_classes,
        weak_pth=args.weak_pth,
        gt_json=args.strong_json,
        val_size=args.size, 
        box_scale=args.box_scale
    )
    if args.local_rank == 0:
        print("======================")
        print(args)
        print("======================")
        print(valid_dst)
        print("======================")
    
    valid_loader = data.DataLoader(
        valid_dst,
        batch_size=1,
        num_workers=0,
        sampler=DistributedSampler(valid_dst, num_replicas=n_gpus, rank=args.local_rank) if args.ddp else None, 
        pin_memory=False,
        drop_last=False,
    )
    
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
    
    checkpoint = torch.load(args.ckpt, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["model_state"], strict=True)
    if args.local_rank == 0:
        print("LOG: Model restored from %s" % args.ckpt)

    if args.ddp:
        model = DDP(
            model,
            device_ids=[args.gpu],
            output_device=args.gpu,
        )
        
    pseudo_annotations = inference()
    
    if args.local_rank == 0:
        strong_labels = valid_dst.gt_annotations.dataset
        print("strong_labels : ", len(strong_labels['images']), len(strong_labels['annotations']))

        with open(args.weak_json) as f:
            weak_json = json.load(f)
            print("weak_json : ", len(weak_json['images']), len(weak_json['annotations']))

        refined_weak_results = {
            'info': weak_json['info'],
            'licenses': weak_json['licenses'],
            'categories': weak_json['categories'],
            'images': weak_json['images'] + strong_labels['images'],
            'annotations': strong_labels['annotations'] + pseudo_annotations,
            }

        for inst_id, anno in enumerate(refined_weak_results["annotations"]):
            anno['id'] = inst_id+1

        with open(args.save_path, 'w') as f:
            json.dump(refined_weak_results, f)

        print("\n=============================================")
        print(f" strong_labels       : img {len(strong_labels['images'])} , anno {len(strong_labels['annotations'])}")
        print(f" weak_labels         : img {len(weak_json['images'])} , anno {len(weak_json['annotations'])}")
        print(f" refined_weak_labels : img {len(weak_json['images'])} , anno {len(refined_weak_results['annotations'])-len(strong_labels['annotations'])}")
        print(f" merge_labels        : img {len(refined_weak_results['images'])} , anno {len(refined_weak_results['annotations'])}")
        print("=============================================\n")
