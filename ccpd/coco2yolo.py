


import os
import json
from tqdm import tqdm
import argparse
import threading

parser = argparse.ArgumentParser()
parser.add_argument('--json_path', default='./ccpd/ccpd_annotations_test.json', type=str, help="input: coco format(json)")
parser.add_argument('--save_path', default='./ccpd/labels/test', type=str, help="specify where to save the output dir of labels")
args = parser.parse_args()

def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = box[0] + box[2] / 2.0
    y = box[1] + box[3] / 2.0
    w = box[2]
    h = box[3]

    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

def process_image(data, img, ana_txt_save_path, id_map):
    filename = img["file_name"]
    img_width = img["width"]
    img_height = img["height"]
    img_id = img["id"]
    head, tail = os.path.splitext(filename)
    ana_txt_name = head + ".txt"  # 对应的txt名字，与jpg一致
    with open(os.path.join(ana_txt_save_path, ana_txt_name), 'w') as f_txt:
        for ann in data['annotations']:
            if ann['image_id'] == img_id:
                box = convert((img_width, img_height), ann["bbox"])
                f_txt.write("%s %s %s %s %s\n" % (id_map[ann["category_id"]], box[0], box[1], box[2], box[3]))

if __name__ == '__main__':
    json_file = args.json_path
    ana_txt_save_path = args.save_path

    if not os.path.exists(ana_txt_save_path):
        os.makedirs(ana_txt_save_path)

    data = json.load(open(json_file, 'r'))

    id_map = {}
    with open(os.path.join(ana_txt_save_path, 'classes.txt'), 'w') as f:
        for i, category in enumerate(data['categories']):
            f.write(f"{category['name']}\n")
            id_map[category['id']] = i

    threads = []
    for img in tqdm(data['images']):
        t = threading.Thread(target=process_image, args=(data, img, ana_txt_save_path, id_map))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()
