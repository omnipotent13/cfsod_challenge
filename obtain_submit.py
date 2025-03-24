import os
import shutil

base_dir = './output'
submit_dir = './submit'
if os.path.exists(submit_dir):
    os.remove(submit_dir)
os.makedirs(submit_dir, exist_ok=True)

dataset_names = os.listdir(base_dir)
for dataset_name in dataset_names:
    dataset_path = os.path.join(base_dir, dataset_name)
    json_files = os.listdir(dataset_path)
    for json_file in json_files:
        json_path = os.path.join(dataset_path, json_file)
        shutil.copy(json_path, os.path.join(submit_dir, json_file))





