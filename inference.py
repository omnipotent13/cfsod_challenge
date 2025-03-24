import ast
from argparse import ArgumentParser

from mmengine.logging import print_log

from mmdet.apis import DetInferencer
from mmdet.evaluation import get_classes

import argparse
import os
import os.path as osp
import warnings
from copy import deepcopy

from mmengine import ConfigDict
from mmengine.config import Config, DictAction
from mmengine.runner import Runner

from mmdet.engine.hooks.utils import trigger_visualization_hook
from mmdet.evaluation import DumpDetResults
from mmdet.registry import RUNNERS
from mmdet.utils import setup_cache_size_limit_of_dynamo
from tqdm import tqdm
import json

import sys
sys.path.append('./')


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        'model',
        type=str,
        help='Config or checkpoint .pth file or the model name '
        'and alias defined in metafile. The model configuration '
        'file will try to read from .pth if the parameter is '
        'a .pth weights file.')
    parser.add_argument(
        'root_path',
        type=str,
        help='path of images.')
    parser.add_argument(
        'json_path',
        type=str,
        help='path of dataset json, used only for images name.')
    parser.add_argument(
        'cat_path',
        type=str,
        help='path of catagory json, obtained by qwen.')
    parser.add_argument(
        'output_path',
        type=str,
        help='path of the output json.')
    parser.add_argument('--weights', default=None, help='Checkpoint file')
    parser.add_argument(
        '--out-dir',
        type=str,
        default='outputs',
        help='Output directory of images or prediction results.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='none',
        choices=['coco', 'voc', 'citys', 'random', 'none'],
        help='Color palette used for visualization')
    parser.add_argument(
        '--custom-entities',
        '-c',
        action='store_true',
        help='Whether to customize entity names? '
        'If so, the input text should be '
        '"cls_name1 . cls_name2 . cls_name3 ." format')
    parser.add_argument(
        '--chunked-size',
        '-s',
        type=int,
        default=-1,
        help='If the number of categories is very large, '
        'you can specify this parameter to truncate multiple predictions.')

    call_args = vars(parser.parse_args())

    if call_args['model'].endswith('.pth'):
        print_log('The model is a weight file, automatically '
                  'assign the model to --weights')
        call_args['weights'] = call_args['model']
        call_args['model'] = None

    init_kws = ['model', 'weights', 'device', 'palette']
    init_args = {}
    for init_kw in init_kws:
        init_args[init_kw] = call_args.pop(init_kw)

    return init_args, call_args


def main():
    init_args, call_args = parse_args()
    
    root_path = call_args['root_path']
    json_path = call_args['json_path']
    cat_path = call_args['cat_path']
    output_path = call_args['output_path']
    weights_path = init_args['weights']

    score_threshold = 0.3
    if 'dataset2' in root_path:
        score_threshold = 0.1
    os.makedirs(output_path, exist_ok=True)
    cat_maps = {}
    cat_id_maps = {}
    name_maps = {}
    images = []
    with open(cat_path, 'r') as f:
        cat_data = json.load(f)
    with open(json_path, 'r') as f:
        json_data = json.load(f)
        categories_dict = cat_data["categories"]
        cat_list = []
        cat_idx = 0
        for cat in categories_dict:
            cat_list.append(cat["name"])
            cat_maps[cat["name"]] = cat["id"]
            cat_id_maps[cat_idx] = cat["id"]
            cat_idx = cat_idx + 1
        img_list = json_data["images"]
        for img in img_list:
            # name_maps[img["id"]] = img["file_name"]
            name_maps[img["file_name"]] = img["id"]
            images.append(img["file_name"])
    todo_str = ""
    for cat in cat_list:
        todo_str = todo_str + cat + ". "
        
    todo_str = todo_str[:-2]
    text_prompt = todo_str

    save_jsons = []
    call_args['texts'] = text_prompt

    inferencer = DetInferencer(**init_args)

    chunked_size = call_args.pop('chunked_size')
    inferencer.model.test_cfg.chunked_size = chunked_size

    save_jsons = []

    img_idx = 0
    print ('inference text prompt:       ',text_prompt)
    print ('inference score threshold:       ',score_threshold)

    for img in tqdm(images):

        img_name = img
        img_path = os.path.join(root_path, img)
        image_path = img_path
        call_args['inputs'] = image_path

        result = inferencer(**call_args)

        labels = result['predictions'][0]['labels']
        scores = result['predictions'][0]['scores']
        boxes = result['predictions'][0]['bboxes']

        score_threshold = score_threshold
        todo_scores = []
        for i in range(len(scores)):
            if scores[i] > score_threshold:
                todo_scores.append(scores[i])
        todo_labels = []
        todo_boxes = []
        for i in range(len(labels)):
            if scores[i] > score_threshold:
                todo_labels.append(labels[i])
                todo_boxes.append(boxes[i])
        
        for li in range(len(todo_scores)):
            # todo_label = todo_labels[li]
            todo_label = cat_id_maps[todo_labels[li]]
            todo_imgid = name_maps[img_name]
            _box = todo_boxes[li]
            x0, y0, x1, y1 = _box
            _box = [x0, y0, x1 - x0, y1 - y0]
            _score = todo_scores[li]

            save_jsons.append({"image_id": todo_imgid, "category_id": todo_label, "bbox": _box, "score": _score})
    
        img_idx = img_idx + 1
        print ('process:  ', img_idx, len(images))

    # 提取数据集名称
    if 'dataset1' in root_path:
        dataset_name = 'dataset1'
    elif 'dataset2' in root_path:
        dataset_name = 'dataset2'
    elif 'dataset3' in root_path:
        dataset_name = 'dataset3'
    else:
        dataset_name = 'unknown'

    # 提取shot信息
    if '1shot' in weights_path:
        shot_info = '_1shot'
    elif '5shot' in weights_path:
        shot_info = '_5shot'
    elif '10shot' in weights_path:
        shot_info = '_10shot'
    else:
        shot_info = ''

    # 对dataset2特殊处理，它不需要shot信息
    if dataset_name == 'dataset2':
        output_filename = f'{dataset_name}.json'
    elif dataset_name == 'unknown':
        output_filename = 'coco_instances_results.json'
    else:
        output_filename = f'{dataset_name}{shot_info}.json'

    # 保存JSON文件
    output_file_path = os.path.join(output_path, output_filename)
    with open(output_file_path, 'w') as f:
        json.dump(save_jsons, f)
        

if __name__ == '__main__':
    main()