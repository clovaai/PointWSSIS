import os
import argparse
import json
import torch
import pycocotools.mask as mask_util
from tqdm import tqdm
from torch.multiprocessing import Process

COCO_CATEGORIES = [
    {"color": [220, 20, 60], "isthing": 1, "id": 1, "name": "person"},
    {"color": [119, 11, 32], "isthing": 1, "id": 2, "name": "bicycle"},
    {"color": [0, 0, 142], "isthing": 1, "id": 3, "name": "car"},
    {"color": [0, 0, 230], "isthing": 1, "id": 4, "name": "motorcycle"},
    {"color": [106, 0, 228], "isthing": 1, "id": 5, "name": "airplane"},
    {"color": [0, 60, 100], "isthing": 1, "id": 6, "name": "bus"},
    {"color": [0, 80, 100], "isthing": 1, "id": 7, "name": "train"},
    {"color": [0, 0, 70], "isthing": 1, "id": 8, "name": "truck"},
    {"color": [0, 0, 192], "isthing": 1, "id": 9, "name": "boat"},
    {"color": [250, 170, 30], "isthing": 1, "id": 10, "name": "traffic light"},
    {"color": [100, 170, 30], "isthing": 1, "id": 11, "name": "fire hydrant"},
    {"color": [220, 220, 0], "isthing": 1, "id": 13, "name": "stop sign"},
    {"color": [175, 116, 175], "isthing": 1, "id": 14, "name": "parking meter"},
    {"color": [250, 0, 30], "isthing": 1, "id": 15, "name": "bench"},
    {"color": [165, 42, 42], "isthing": 1, "id": 16, "name": "bird"},
    {"color": [255, 77, 255], "isthing": 1, "id": 17, "name": "cat"},
    {"color": [0, 226, 252], "isthing": 1, "id": 18, "name": "dog"},
    {"color": [182, 182, 255], "isthing": 1, "id": 19, "name": "horse"},
    {"color": [0, 82, 0], "isthing": 1, "id": 20, "name": "sheep"},
    {"color": [120, 166, 157], "isthing": 1, "id": 21, "name": "cow"},
    {"color": [110, 76, 0], "isthing": 1, "id": 22, "name": "elephant"},
    {"color": [174, 57, 255], "isthing": 1, "id": 23, "name": "bear"},
    {"color": [199, 100, 0], "isthing": 1, "id": 24, "name": "zebra"},
    {"color": [72, 0, 118], "isthing": 1, "id": 25, "name": "giraffe"},
    {"color": [255, 179, 240], "isthing": 1, "id": 27, "name": "backpack"},
    {"color": [0, 125, 92], "isthing": 1, "id": 28, "name": "umbrella"},
    {"color": [209, 0, 151], "isthing": 1, "id": 31, "name": "handbag"},
    {"color": [188, 208, 182], "isthing": 1, "id": 32, "name": "tie"},
    {"color": [0, 220, 176], "isthing": 1, "id": 33, "name": "suitcase"},
    {"color": [255, 99, 164], "isthing": 1, "id": 34, "name": "frisbee"},
    {"color": [92, 0, 73], "isthing": 1, "id": 35, "name": "skis"},
    {"color": [133, 129, 255], "isthing": 1, "id": 36, "name": "snowboard"},
    {"color": [78, 180, 255], "isthing": 1, "id": 37, "name": "sports ball"},
    {"color": [0, 228, 0], "isthing": 1, "id": 38, "name": "kite"},
    {"color": [174, 255, 243], "isthing": 1, "id": 39, "name": "baseball bat"},
    {"color": [45, 89, 255], "isthing": 1, "id": 40, "name": "baseball glove"},
    {"color": [134, 134, 103], "isthing": 1, "id": 41, "name": "skateboard"},
    {"color": [145, 148, 174], "isthing": 1, "id": 42, "name": "surfboard"},
    {"color": [255, 208, 186], "isthing": 1, "id": 43, "name": "tennis racket"},
    {"color": [197, 226, 255], "isthing": 1, "id": 44, "name": "bottle"},
    {"color": [171, 134, 1], "isthing": 1, "id": 46, "name": "wine glass"},
    {"color": [109, 63, 54], "isthing": 1, "id": 47, "name": "cup"},
    {"color": [207, 138, 255], "isthing": 1, "id": 48, "name": "fork"},
    {"color": [151, 0, 95], "isthing": 1, "id": 49, "name": "knife"},
    {"color": [9, 80, 61], "isthing": 1, "id": 50, "name": "spoon"},
    {"color": [84, 105, 51], "isthing": 1, "id": 51, "name": "bowl"},
    {"color": [74, 65, 105], "isthing": 1, "id": 52, "name": "banana"},
    {"color": [166, 196, 102], "isthing": 1, "id": 53, "name": "apple"},
    {"color": [208, 195, 210], "isthing": 1, "id": 54, "name": "sandwich"},
    {"color": [255, 109, 65], "isthing": 1, "id": 55, "name": "orange"},
    {"color": [0, 143, 149], "isthing": 1, "id": 56, "name": "broccoli"},
    {"color": [179, 0, 194], "isthing": 1, "id": 57, "name": "carrot"},
    {"color": [209, 99, 106], "isthing": 1, "id": 58, "name": "hot dog"},
    {"color": [5, 121, 0], "isthing": 1, "id": 59, "name": "pizza"},
    {"color": [227, 255, 205], "isthing": 1, "id": 60, "name": "donut"},
    {"color": [147, 186, 208], "isthing": 1, "id": 61, "name": "cake"},
    {"color": [153, 69, 1], "isthing": 1, "id": 62, "name": "chair"},
    {"color": [3, 95, 161], "isthing": 1, "id": 63, "name": "couch"},
    {"color": [163, 255, 0], "isthing": 1, "id": 64, "name": "potted plant"},
    {"color": [119, 0, 170], "isthing": 1, "id": 65, "name": "bed"},
    {"color": [0, 182, 199], "isthing": 1, "id": 67, "name": "dining table"},
    {"color": [0, 165, 120], "isthing": 1, "id": 70, "name": "toilet"},
    {"color": [183, 130, 88], "isthing": 1, "id": 72, "name": "tv"},
    {"color": [95, 32, 0], "isthing": 1, "id": 73, "name": "laptop"},
    {"color": [130, 114, 135], "isthing": 1, "id": 74, "name": "mouse"},
    {"color": [110, 129, 133], "isthing": 1, "id": 75, "name": "remote"},
    {"color": [166, 74, 118], "isthing": 1, "id": 76, "name": "keyboard"},
    {"color": [219, 142, 185], "isthing": 1, "id": 77, "name": "cell phone"},
    {"color": [79, 210, 114], "isthing": 1, "id": 78, "name": "microwave"},
    {"color": [178, 90, 62], "isthing": 1, "id": 79, "name": "oven"},
    {"color": [65, 70, 15], "isthing": 1, "id": 80, "name": "toaster"},
    {"color": [127, 167, 115], "isthing": 1, "id": 81, "name": "sink"},
    {"color": [59, 105, 106], "isthing": 1, "id": 82, "name": "refrigerator"},
    {"color": [142, 108, 45], "isthing": 1, "id": 84, "name": "book"},
    {"color": [196, 172, 0], "isthing": 1, "id": 85, "name": "clock"},
    {"color": [95, 54, 80], "isthing": 1, "id": 86, "name": "vase"},
    {"color": [128, 76, 255], "isthing": 1, "id": 87, "name": "scissors"},
    {"color": [201, 57, 1], "isthing": 1, "id": 88, "name": "teddy bear"},
    {"color": [246, 0, 122], "isthing": 1, "id": 89, "name": "hair drier"},
    {"color": [191, 162, 208], "isthing": 1, "id": 90, "name": "toothbrush"},
]
thing_ids = [k["id"] for k in COCO_CATEGORIES]


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Detectron2 Demo")
    parser.add_argument("--save_path", default="/home/user/DB/coco/annotations", help="path to config file",)
    parser.add_argument("--full_json", default="/home/user/DB/coco/annotations/instances_train2017.json", help="path to config file",)
    parser.add_argument("--strong_json", default="/home/user/DB", help="path to config file",)
    parser.add_argument("--weak_pth", default="/home/user/DB", help="path to config file",)
    parser.add_argument("--workers", default=4, type=int, help="the number of workers for multi-processing",)
    args = parser.parse_args()

    with open(args.full_json) as json_file:
        full_labels = json.load(json_file)

    with open(args.strong_json) as json_file:
        strong_labels = json.load(json_file)

    weak_labels = torch.load(args.weak_pth, map_location='cpu')

    save_json = {}
    save_json["info"] = full_labels["info"]
    save_json["licenses"] = full_labels["licenses"]
    save_json["categories"] = full_labels["categories"]
    save_json["images"] = full_labels["images"]
    save_json["annotations"] = strong_labels["annotations"]

    '''
    strong image : ['license', 'file_name', 'coco_url', 'height', 'width', 'date_captured', 'flickr_url', 'id']
    strong annotation : ['segmentation', 'area', 'iscrowd', 'image_id', 'bbox', 'category_id', 'id']

    weak annotation : ['image_id', 'category_id', 'bbox', 'score', 'segmentation']
    '''

    strong_images = [s['id'] for s in strong_labels["images"]]
    print("strong_images : ", len(strong_images))

    weak_images = [pred['image_id'] for pred in weak_labels]
    print("weak_images : ", len(weak_images))

    merge_images = list(set(strong_images+weak_images))
    print("merge_images : %d vs %d" %(len(strong_images)+len(weak_images), len(merge_images)))

    def run(args, pid):
        anno = []

        for n, pred in enumerate(tqdm(weak_labels)):
            if n % args.workers != pid:
                continue

            for inst in pred['instances']:
                rle = inst['segmentation']
                mask = mask_util.decode(rle)

                if mask.sum() == 0:
                    continue

                y_coord, x_coord = mask.nonzero()
                ymin, xmin = y_coord.min(), x_coord.min()
                ymax, xmax = y_coord.max(), x_coord.max()
                bbox = [int(xmin), int(ymin), int(xmax-xmin), int(ymax-ymin)]

                anno_dict = {}
                anno_dict['segmentation'] = inst['segmentation']
                anno_dict['area'] = int(mask.sum())
                anno_dict['iscrowd'] = 0
                anno_dict['image_id'] = int(pred['image_id'])
                anno_dict['bbox'] = bbox
                anno_dict['category_id'] = thing_ids[inst['category_id']]
                anno_dict['id'] = 0

                anno.append(anno_dict)

        # temporarily save annotations on disk
        with open(f"{os.path.basename(args.save_path)}_id{pid}.json", 'w') as f:
            json.dump(anno, f)


    processes = []
    for i in range(args.workers):
        proc = Process(target=run, args=(args, i))
        processes.append(proc)
        proc.start()
    for proc in processes:
        proc.join()

    # aggregate all annotations from temporarily saved files and remove them
    for pid in range(args.workers):
        with open(f"{os.path.basename(args.save_path)}_id{pid}.json", 'r') as f:
            save_json["annotations"] += json.load(f)
        os.system(f"rm -f {os.path.basename(args.save_path)}_id{pid}.json")
        
    for inst_id, anno in enumerate(save_json["annotations"]):
            anno['id'] = inst_id+1

    with open(args.save_path, 'w') as f:
        json.dump(save_json, f)
        