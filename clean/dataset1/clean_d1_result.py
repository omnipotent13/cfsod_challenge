import os
import json
import numpy as np


test_cat_path = 'clean/dataset1/test_image_categories.json'
test_anno_path = './dataset/dataset1/annotations/test.json'
base_dir = "output/dataset1"

json_names = os.listdir(base_dir)
for json_name in json_names:
    if os.path.isdir(os.path.join(base_dir, json_name)):
        continue
    todo_json_path = os.path.join(base_dir, json_name)
    clean_json_name = json_name.split('.')[0] + ".json"
    save_json_path = os.path.join(base_dir, clean_json_name)
    if not os.path.exists(todo_json_path):
        continue

    with open(todo_json_path, 'r') as f:
        todo_json = json.load(f)
    with open(test_cat_path, 'r') as f:
        test_cat = json.load(f)["images"]
    with open(test_anno_path, 'r') as f:
        test_anno = json.load(f)

    anno_maps = {test_anno["images"][i]["file_name"]: test_anno["images"][i]["id"] for i in range(len(test_anno["images"]))}
    # test_img_ids = [x['id'] for x in test_anno["images"]]
    img_name_id_maps = {x['file_name']: x['id'] for x in test_anno["images"]}
    img_id_name_maps = {x['id']: x['file_name'] for x in test_anno["images"]}
    test_anno_maps = {x['name']: x['id'] for x in test_anno["categories"]}

    cat_test_imgs = []
    cat_img_maps = {x['id']: [] for x in test_anno["categories"]}
    for item in test_cat.keys():
        cat_list = test_cat[item]
        cat_test_imgs += cat_list
        for cat in cat_list:
            if cat in img_name_id_maps:
                cat_img_maps[img_name_id_maps[cat]] = test_anno_maps[item]


    save_json = []


    mod_nums = 0
    for item in todo_json:
        img_id = item['image_id']
        cat_id = item['category_id']
        target_id = cat_img_maps[img_id]
        if cat_id != target_id:
            mod_nums += 1
            cat_id = target_id
        _info = {}
        _info['image_id'] = img_id
        _info['category_id'] = cat_id
        _info['bbox'] = item['bbox']
        _info['score'] = item['score']
        
        save_json.append(_info)

    with open(save_json_path, 'w') as f:
        json.dump(save_json, f)



