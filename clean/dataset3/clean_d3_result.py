import os
import json

# 文件路径
TEST_CAT_PATH = './clean/dataset3/test_image_categories.json'
TEST_ANNO_PATH = './dataset/dataset3/annotations/test.json'
BASE_DIR = './output/dataset3'

# 读取 JSON 文件
def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def process_annotations():
    test_cat = load_json(TEST_CAT_PATH)["images"]
    test_anno = load_json(TEST_ANNO_PATH)
    
    # 创建文件名到 ID 的映射
    img_name_id_map = {img['file_name']: img['id'] for img in test_anno["images"]}
    
    # 类别名称到类别 ID 的映射
    category_name_to_id = {cat['name']: cat['id'] for cat in test_anno["categories"]}
    
    # 每张图片对应的**有效类别 ID**
    img_valid_categories = {}
    for category, image_list in test_cat.items():
        cat_id = category_name_to_id.get(category)  # 获取类别 ID
        if cat_id is None:
            continue
        for img_name in image_list:
            img_id = img_name_id_map.get(img_name)
            if img_id is not None:
                img_valid_categories.setdefault(img_id, set()).add(cat_id)

    # 遍历输出目录
    for json_name in os.listdir(BASE_DIR):
        if os.path.isdir(os.path.join(BASE_DIR, json_name)):
            continue
        todo_json_path = os.path.join(BASE_DIR, json_name)
        new_json_name = json_name.split('.')[0] + '_clean.json'
        save_json_path = os.path.join(BASE_DIR, new_json_name)
        
        if not os.path.exists(todo_json_path):
            continue
        
        todo_json = load_json(todo_json_path)
        
        save_json = []
        removed_count = 0
        
        for item in todo_json:
            img_id = item['image_id']
            cat_id = item['category_id']
            
            # 检查类别 ID 是否属于该图片
            if img_id in img_valid_categories and cat_id in img_valid_categories[img_id]:
                save_json.append(item)
            else:
                removed_count += 1
        
        print(f"{removed_count} bboxes removed in {json_name}")
        
        with open(save_json_path, 'w') as f:
            json.dump(save_json, f, indent=4)

if __name__ == "__main__":
    process_annotations()