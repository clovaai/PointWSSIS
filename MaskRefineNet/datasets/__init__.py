"""
PointWSSIS (https://arxiv.org/abs/2303.15062)
Copyright (c) 2023-present NAVER Cloud Corp.
Apache-2.0
"""

from utils import ext_transforms as et
from .dataset import TrainSet, ValidSet

def dataset_factory(args):
    """Dataset And Augmentation"""
    # augmentation list for training

    if type(args.train_size) == list:
        train_ext = [
            et.ExtResize(args.train_size),
        ]
    else:
        train_ext = [
            et.ExtRandomScaleResize(args.train_size, (args.aug_min_scale, args.aug_max_scale), interpolation="nearest"),
            et.ExtCenterCrop(args.train_size),
        ]
    
    train_ext += [
        et.ExtColorJitter(brightness=0.5, contrast=0.25, saturation=0.25, p=args.aug_color),
        et.ExtRandomHorizontalFlip(),
        et.ExtToTensor(target_type="uint8"),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    train_transform = et.ExtCompose(train_ext)

    train_dst = TrainSet(
        root=args.data_root, 
        dataset=args.dataset,
        num_classes=args.num_classes,
        weak_pth=args.weak_pth,
        gt_json=args.gt_json,
        transform=train_transform, 
        box_scale=args.box_scale
    )

    valid_dst = ValidSet(
        root=args.data_root, 
        dataset=args.dataset,
        num_classes=args.num_classes,
        weak_pth=args.eval_pth,
        gt_json=args.eval_json,
        val_size=args.val_size, 
        box_scale=args.box_scale,
        filtering=True,
    )

    return train_dst, valid_dst
