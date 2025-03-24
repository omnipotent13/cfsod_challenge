import os
import json
import numpy as np

def nms(boxes, scores, threshold):
    """
    非极大值抑制的纯 Python 实现

    :param boxes: 边界框数组，形状为 (N, 4)，每一行表示一个边界框 [x1, y1, x2, y2]
    :param scores: 每个边界框对应的得分，形状为 (N,)
    :param threshold: 交并比（IoU）阈值，用于判断两个边界框是否重叠
    :return: 保留的边界框的索引
    """
    if len(boxes) == 0:
        return []
    # 边界框的坐标
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # 计算每个边界框的面积
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # 按得分从高到低排序
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        # 计算当前得分最高的边界框与其他边界框的交集区域的坐标
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        # 计算交集区域的宽度和高度
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        # 计算交集面积
        inter = w * h
        # 计算并集面积
        union = areas[i] + areas[order[1:]] - inter
        # 计算交并比（IoU）
        iou = inter / union

        # 保留 IoU 小于阈值的边界框的索引
        inds = np.where(iou <= threshold)[0]
        order = order[inds + 1]

    return keep

todo_json_path = './output/dataset2/dataset2.json'
save_json_paths = ['./output/dataset2/dataset2_1shot.json', './output/dataset2/dataset2_5shot.json', './output/dataset2/dataset2_10shot.json']

with open(todo_json_path, 'r') as f:
    todo_json = json.load(f)

# 按图像ID分组
pred_img_map = {}
for item in todo_json:
    img_id = item['image_id']
    if img_id not in pred_img_map:
        pred_img_map[img_id] = []
    pred_img_map[img_id].append(item)

save_json = []

for img_id, predictions in pred_img_map.items():
    # 过滤面积大于10000的边界框
    filtered_predictions = [p for p in predictions if p['bbox'][2] * p['bbox'][3] <= 10000]
    
    if filtered_predictions:
        pred_boxes = [p['bbox'] for p in filtered_predictions]
        bboxes = [[x[0], x[1], x[0]+x[2], x[1]+x[3]] for x in pred_boxes]
        scores = [p['score'] for p in filtered_predictions]
        
        keep = nms(np.array(bboxes), np.array(scores), 0.5)
        
        for k in keep:
            save_json.append(filtered_predictions[k])

for save_json_path in save_json_paths:
    with open(save_json_path, 'w') as f:
        json.dump(save_json, f)
