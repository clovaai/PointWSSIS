"""
PointWSSIS (https://arxiv.org/abs/2303.15062)
Copyright (c) 2023-present NAVER Cloud Corp.
Apache-2.0
"""

import os
import math
import torch.utils.data as data
import torch
import numpy as np
import random
import json
from PIL import Image

import torch.nn.functional as F
import torchvision.transforms.functional as TF
from pycocotools.coco import COCO
import pycocotools.mask as mask_util

from datasets.utils import center_map_gen, gaussian, COCO_CATEGORIES, BDD100K_CATEGORIES


class TrainSet(data.Dataset):
    def __init__(self, root, dataset, num_classes, weak_pth=None, 
                 gt_json=None, transform=None, size_cut=0, box_scale=2.0):

        self.dataset = dataset
        self.num_classes = num_classes
        self.weak_pth = weak_pth
        self.gt_json = gt_json
        self.transform = transform
        self.size_cut = size_cut
        self.box_scale = box_scale
        
        self.sigma = 6
        self.g = gaussian(self.sigma)

        if self.dataset == 'coco':
            self.cls_ids = [c['id'] for c in COCO_CATEGORIES]
            self.img_dir = os.path.join(root, "coco", "train2017", "%012d.jpg")
        elif self.dataset ==  'BDD100K':
            self.cls_ids = [c['id'] for c in BDD100K_CATEGORIES]
            self.img_dir = os.path.join(root, "BDD100K", "train", "%s")
        else:
            raise "[ERROR] Invalid dataset: support only coco and BDD100K "

        self.img_info = {}
        self.gt_labels_per_img = {}
        self.pred_labels_per_img = {}
        
        with open(os.path.join(root, dataset, "annotations", self.gt_json)) as f:
            gt_annotations = json.load(f)
            #gt_image_ids = [p['id'] for p in gt_annotations['images']]
        
        for img in gt_annotations['images']:
            self.img_info[img['id']] = img
    
        for anno in gt_annotations['annotations']:
            if anno['image_id'] not in self.gt_labels_per_img:
                self.gt_labels_per_img[anno['image_id']] = []

            self.gt_labels_per_img[anno['image_id']].append(anno)

        for wpth in self.weak_pth:
            weak_labels = torch.load(wpth, map_location='cpu')
            #weak_labels = [w for w in weak_labels if w['image_id'] in gt_image_ids]

            for anno in weak_labels:
                if anno['image_id'] not in self.pred_labels_per_img:
                    self.pred_labels_per_img[anno['image_id']] = []
                self.pred_labels_per_img[anno['image_id']] += anno['instances']

        img_ids = self.pred_labels_per_img.keys()
        self.matched_labels = self.label_matching(img_ids)
        
        np.random.shuffle(self.matched_labels)
        del weak_labels
        
    def __getitem__(self, index):
        scale = random.uniform(1.0, self.box_scale)
        
        l = self.matched_labels[index]
        if self.dataset == 'coco':
            img_path = self.img_dir % l['img_id']
        elif self.dataset == 'BDD100K':
            img_path = self.img_dir % l['file_name']
        else:
            raise "[ERROR] Invalid img path in datasets.coco.py"

        img = np.uint8( Image.open(img_path).convert('RGB') )
        point = l['point']
        pred = self.pred_labels_per_img[l['img_id']][l['pred_idx']]
        gt = self.gt_labels_per_img[l['img_id']][l['gt_idx']]

        pred_mask = mask_util.decode(pred['segmentation'])
        h, w = pred_mask.shape

        if type(gt['segmentation']) == list:
            gt_mask = mask_util.decode(mask_util.merge(mask_util.frPyObjects(gt['segmentation'], h, w)))
        elif type(gt['segmentation']['counts']) == list:
            gt_mask = mask_util.decode(mask_util.frPyObjects(gt['segmentation'], h, w))
        else:
            gt_mask = np.zeros((h, w), dtype=np.uint8)

        # loosely crop
        y_coord, x_coord = pred_mask.nonzero()
        ymin, xmin = int(y_coord.min()), int(x_coord.min())
        ymax, xmax = int(y_coord.max()), int(x_coord.max())

        box_cx, box_cy = (xmin+xmax)/2, (ymin+ymax)/2
        box_w, box_h = (xmax-xmin), (ymax-ymin)

        xmin = int( max(0, box_cx - box_w / 2 * scale) )
        ymin = int( max(0, box_cy - box_h / 2 * scale) )
        xmax = int( min(w-1, box_cx + box_w / 2 * scale) )
        ymax = int( min(h-1, box_cy + box_h / 2 * scale) )

        if point[1] > xmax:
            xmax = int( min(w-1, point[1] + box_w / 2 * scale) )
        if point[1] < xmin:
            xmin = int( max(0, point[1] - box_w / 2 * scale) )
        if point[0] > ymax:
            ymax = int( min(h-1, point[0] + box_h / 2 * scale) )
        if point[0] < ymin:
            ymin = int( max(0, point[0] - box_h / 2 * scale) )

        cropped_img = img[ymin:ymax, xmin:xmax]
        pred_cropped_mask = pred_mask[ymin:ymax, xmin:xmax] * l['category']
        gt_cropped_mask = gt_mask[ymin:ymax, xmin:xmax] * l['category']
        
        #############
        cropped_img = Image.fromarray(cropped_img)
        pred_cropped_mask = Image.fromarray(pred_cropped_mask)
        gt_cropped_mask = Image.fromarray(gt_cropped_mask)
                       
        if self.transform is not None:
            [cropped_img], [pred_cropped_mask, gt_cropped_mask] = self.transform([cropped_img], [pred_cropped_mask, gt_cropped_mask])

        pred_cropped_mask = pred_cropped_mask.long()
        gt_cropped_mask = gt_cropped_mask.long()
        t_cls = gt_cropped_mask.max()
        th, tw = gt_cropped_mask.shape
        
        if t_cls == 0:
            center_map = torch.zeros((self.num_classes+1, th, tw), dtype=torch.float32)
                
        else:
            y_coord, x_coord = gt_cropped_mask.nonzero(as_tuple=True)
            cy, cx = y_coord.float().mean(), x_coord.float().mean()
            
            center_map = np.zeros((self.num_classes+1, th, tw), dtype=np.float32)
            center_map = center_map_gen(center_map, int(cx), int(cy), t_cls, self.sigma, self.g)
            center_map = torch.from_numpy(center_map)

        # make binary mask
        gt_cropped_mask = torch.where(gt_cropped_mask > 0, 1, 0)
        pred_cropped_mask = torch.where(pred_cropped_mask > 0, 1, 0)
        
        cropped_img = torch.cat([cropped_img, center_map, pred_cropped_mask.unsqueeze(0)])

        return {"img": cropped_img, "target": gt_cropped_mask}
                       

    def __len__(self):
        return len(self.matched_labels)


    def __repr__(self):
        format_string = f"weak ({self.weak_pth}) | gt ({self.gt_json}) | # data {len(self.matched_labels)}"
        return format_string
    

    def label_matching(self, img_ids):
        matched_labels = []
        
        for img_id in img_ids:
            if img_id in self.gt_labels_per_img and img_id in self.pred_labels_per_img:
                info = self.img_info[img_id]

                gt_labels = self.gt_labels_per_img[img_id]
                pred_labels = self.pred_labels_per_img[img_id]

                gt_point_per_cls = {}

                for j, l in enumerate(gt_labels):
                    if type(l['segmentation']) == list:
                        m = mask_util.decode(mask_util.merge(mask_util.frPyObjects(l['segmentation'], info['height'], info['width'])))
                    elif type(l['segmentation']['counts']) == list:
                        m = mask_util.decode(mask_util.frPyObjects(l['segmentation'], info['height'], info['width']))
                    # elif 'counts' in l['segmentation']:
                    #     m = mask_util.decode(l['segmentation'])
                    else:
                        print("[TYPE ERROR] in label_matching")
                        continue
                    cate = self.cls_ids.index(l['category_id'])

                    y_coord, x_coord = m.nonzero()
                    cy, cx = y_coord.mean(), x_coord.mean()

                    if cate not in gt_point_per_cls:
                        gt_point_per_cls[cate] = []

                    gt_point_per_cls[cate].append({"point":(cy, cx), "idx":j})

                for j, l in enumerate(pred_labels):
                    cate = l['category_id']
                    mask_h, mask_w = l['segmentation']['size']

                    cy = (l['bbox'][0] / 10000 * mask_h)
                    cx = (l['bbox'][1] / 10000 * mask_w)

                    match_point = {"point":(0,0), "dist":999999, "idx":-1}

                    if cate in gt_point_per_cls:
                        for gt in gt_point_per_cls[cate]:
                            dist = math.sqrt( (cy-gt['point'][0])**2 + (cx-gt['point'][1])**2 )
                            if dist < match_point["dist"]:
                                match_point["dist"] = dist
                                match_point["point"] = gt['point']
                                match_point["idx"] = gt['idx']

                        if dist != 999999:
                            matched_labels.append(
                                {
                                    "pred_idx":j, 
                                    "gt_idx":match_point['idx'],
                                    "point":match_point['point'],
                                    "img_id":img_id,
                                    "category":cate,
                                    "file_name":info["file_name"],
                                }
                            )
        
        return matched_labels



class ValidSet(data.Dataset):
    def __init__(self, root, dataset, num_classes, weak_pth=None, 
                 gt_json=None, transform=None, size_cut=0, val_size=256, 
                 box_scale=2.0, filtering=False):

        self.dataset = dataset
        self.num_classes = num_classes
        self.weak_pth = weak_pth
        self.gt_json = gt_json
        self.transform = transform
        self.size_cut = size_cut
        self.val_size = val_size
        self.box_scale = box_scale
        
        self.sigma = 6
        self.g = gaussian(self.sigma)

        if self.dataset == 'coco':
            self.img_dir = os.path.join(root, "coco", "train2017", "%012d.jpg")
            self.cls_ids = [c['id'] for c in COCO_CATEGORIES]
            
        elif self.dataset ==  'BDD100K':
            self.img_dir = os.path.join(root, "BDD100K", "%s", "%s")
            self.cls_ids = [c['id'] for c in BDD100K_CATEGORIES]
        else:
            raise "[ERROR] Invalid dataset: support only coco and BDD100K "

        self.gt_annotations = COCO(gt_json)
            
        gt_image_ids = [p['id'] for p in self.gt_annotations.dataset['images']]
        
        self.id_to_fname = {}
        for p in self.gt_annotations.dataset['images']:
            self.id_to_fname[p['id']] = p['file_name']

        self.weak_labels = torch.load(weak_pth, map_location='cpu')
        if filtering:
            self.weak_labels = [w for w in self.weak_labels if w['image_id'] in gt_image_ids]

    def __getitem__(self, index):

        pred = self.weak_labels[index]
        
        if self.dataset == 'coco':
            img_path = self.img_dir % pred['image_id']
        elif self.datsaet == 'BDD100K':
            img_path = self.img_dir % ("train", self.id_to_fname[pred['image_id']])
            if not os.path.exists(img_path):
                img_path = self.img_dir % ("val", self.id_to_fname[pred['image_id']])
        else:
            raise "[ERROR] Invalid img path in datasets.coco.py"

        x = Image.open(img_path).convert('RGB')
        x = TF.to_tensor(x)
        x = TF.normalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        meta = []
        inputs = []
        
        for inst in pred['instances']:
            category_id = inst['category_id']
            rle = inst['segmentation']
            mask = mask_util.decode(rle)
            mask_h, mask_w = mask.shape
            
            y_coord, x_coord = mask.nonzero()
            ymin, xmin = int(y_coord.min()), int(x_coord.min())
            ymax, xmax = int(y_coord.max()), int(x_coord.max())
            bbox = (xmin, ymin, xmax-xmin, ymax-ymin)
            
            box_cx, box_cy = (xmin+xmax)/2, (ymin+ymax)/2
            box_w, box_h = (xmax-xmin), (ymax-ymin)

            crop_xmin = int( max(0, box_cx - box_w / 2 * self.box_scale) )
            crop_ymin = int( max(0, box_cy - box_h / 2 * self.box_scale) )
            crop_xmax = int( min(mask_w-1, box_cx + box_w / 2 * self.box_scale) )
            crop_ymax = int( min(mask_h-1, box_cy + box_h / 2 * self.box_scale) )
            
            gt_cy = (inst['bbox'][0] / 10000 * mask_h)
            gt_cx = (inst['bbox'][1] / 10000 * mask_w)

            if gt_cx > crop_xmax:
                crop_xmax = int( min(mask_w-1, gt_cx + box_w / 2 * self.box_scale) )
            if gt_cx < crop_xmin:
                crop_xmin = int( max(0, gt_cx - box_w / 2 * self.box_scale) )
            if gt_cy > crop_ymax:
                crop_ymax = int( min(mask_h-1, gt_cy + box_h / 2 * self.box_scale) )
            if gt_cy < crop_ymin:
                crop_ymin = int( max(0, gt_cy - box_h / 2 * self.box_scale) )

            if (crop_ymax-crop_ymin+1) > self.size_cut and (crop_xmax-crop_xmin+1) > self.size_cut and mask.sum() > 0:
                cropped_x = x[:, crop_ymin:crop_ymax+1, crop_xmin:crop_xmax+1].clone()
                cropped_mask = torch.from_numpy(mask[crop_ymin:crop_ymax+1, crop_xmin:crop_xmax+1]) * (category_id + 1)
                cropped_cls = cropped_mask.max()
                
                _, ori_h, ori_w = cropped_x.shape

                cropped_mask = torch.where(cropped_mask > 0, 1, 0)

                cropped_x, cropped_mask = self.tensor_resize(cropped_x, cropped_mask, self.val_size)
                
                _, crop_h, crop_w = cropped_mask.shape

                gt_cx = int( (gt_cx-crop_xmin) / ori_w * crop_w)
                gt_cy = int( (gt_cy-crop_ymin) / ori_h * crop_h)

                center_map = np.zeros((self.num_classes+1, crop_h, crop_w), dtype=np.float32)
                center_map = center_map_gen(center_map, gt_cx, gt_cy, cropped_cls, self.sigma, self.g)
                center_map = torch.from_numpy(center_map)

                cropped_x = torch.cat([cropped_x, center_map, cropped_mask])
                    
                inputs.append(cropped_x)
            else:
                inputs.append(0)
                ori_h, ori_w = 0, 0
                
            meta.append(
                {
                    'image_id': inst['image_id'], 
                    'category_id': self.cls_ids[inst['category_id']], 
                    'bbox': bbox,
                    'crop_bbox': (crop_xmin, crop_ymin, crop_xmax, crop_ymax),
                    'segmentation': mask,
                    'ori_size': (ori_h, ori_w),
                }
            )
        return inputs, meta

    def tensor_resize(self, img, mask, size):

        if type(size) == list:
            h, w = size
        else:
            h, w = size, size

        img = F.interpolate(img[None, ...], (h, w), mode="bilinear", align_corners=False).squeeze(0)
        mask = F.interpolate(mask[None, None, ...].float(), (h, w), mode="nearest").squeeze(0)
        
        return img, mask

    def __len__(self):
        return len(self.weak_labels)

    def __repr__(self):
        format_string = f" weak ({self.weak_pth}): # {len(self.weak_labels)} \n gt ({self.gt_json}): # {len(self.gt_annotations.dataset['images'])}"
        return format_string
    
