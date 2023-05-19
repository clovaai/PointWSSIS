"""
PointWSSIS (https://arxiv.org/abs/2303.15062)
Copyright (c) 2023-present NAVER Cloud Corp.
Apache-2.0
"""

import torch
from .scheduler import CosineAnnealingWarmUpRestarts
from .utils import AverageMeter, save_checkpoint, tensorboard_report, get_coco_info

def optim_factory(
    args, params, train_iters, warm_iters, resume_iter=-1, lr_scale=1.0,
):
    lr = args.lr * lr_scale
    
    if "cosine" in args.lr_policy:
        min_lr = lr * 0.01
        max_lr = lr
        lr = min_lr

    # optimizer
    if args.optim == "adam":
        optimizer = torch.optim.Adam(params, lr)

    elif args.optim == "sgd":
        optimizer = torch.optim.SGD(
            params,
            lr=lr,
            weight_decay=args.weight_decay,
            momentum=0.9,
            nesterov=True,
        )

    else:
        raise "[ERROR] please check again args.optim"

    # lr scheduler
    if args.lr_policy == "cosine":
        scheduler = CosineAnnealingWarmUpRestarts(
            optimizer,
            T_0=train_iters,
            T_mult=1,
            gamma=1.0,
            T_up=warm_iters,
            eta_max=max_lr,
            last_epoch=resume_iter,
        )
    else:
        scheduler = None

    return optimizer, scheduler


