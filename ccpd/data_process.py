import os
import shutil

import cv2
import random
import os.path


def annotation_from_name(file_name):
    # 图片的名称就是标签，由"_"字符分隔
    # # 0236-16_34-222&408_398&520-377&473_222&520_243&455_398&408-0_0_28_14_33_32_29-69-74.jpg
    file_name = file_name[:-4]
    # 0236-16_34-222&408_398&520-377&473_222&520_243&455_398&408-0_0_28_14_33_32_29-69-74
    name_split = file_name.split('-')
    location = name_split[2]
    # 边界框坐标 222&408_398&520
    location = location.split('_')
    # 222&408
    left_top = location[0].split('&')
    # 398&520
    right_bottom = location[1].split('&')
    x1 = int(left_top[0])
    y1 = int(left_top[1])
    x2 = int(right_bottom[0])
    y2 = int(right_bottom[1])
    # 四个顶点位置  377&473_222&520_243&455_398&408
    x1y1 = name_split[3].split('_')[0].split('&')
    x2y2 = name_split[3].split('_')[1].split('&')
    x3y3 = name_split[3].split('_')[2].split('&')
    x4y4 = name_split[3].split('_')[3].split('&')
    # 边界框两个顶点以及四个顶点位置 0-11
    return (
    x1, y1, x2, y2, int(x1y1[0]), int(x1y1[1]), int(x2y2[0]), int(x2y2[1]), int(x3y3[0]), int(x3y3[1]), int(x4y4[0]),
    int(x4y4[1]))


def generate_data_list(ccpd_path=r'D:\traffic_detection\ccpd'):
    # D:\data\CCPD\
    # Base	200	正常车牌
    # Blur	5	由于相机抖动造成的模糊
    # Challenge	10	其他的比较有挑战性的车牌
    # DB		20	光线暗或者比较亮
    # FN		20	距离摄像头相当的远或者相当近
    # Rotate	10	水平倾斜20-25°，垂直倾斜-10-10°
    # Tilt	10	水平倾斜15-45°，垂直倾斜15-45°
    # Weather	10	在雨天，雪天，或者雾天
    # NP		5	没有车牌的新车
    # 输入的数据集
    train_dir = r'D:\traffic_detection\ccpd\train'
    val_dir = r'D:\traffic_detection\ccpd\val'
    test_dir = r'D:\traffic_detection\ccpd\test'

    image_roots = [os.path.join(ccpd_path, 'ccpd_base'),
                   os.path.join(ccpd_path, 'ccpd_blur'),
                   os.path.join(ccpd_path, 'ccpd_challenge'),
                   os.path.join(ccpd_path, 'ccpd_db'),
                   os.path.join(ccpd_path, 'ccpd_fn'),
                   os.path.join(ccpd_path ,'ccpd_np'),
                   os.path.join(ccpd_path, 'ccpd_rotate'),
                   os.path.join(ccpd_path, 'ccpd_tilt'),
                   os.path.join(ccpd_path, 'ccpd_weather')]
    # 输出
    train_list_file_path = r'D:\traffic_detection\ccpd\data_list_CCPD_train.txt'
    test_list_file_path = r'D:\traffic_detection\ccpd\data_list_CCPD_test.txt'
    val_list_file_path=r'D:\traffic_detection\ccpd\data_list_CCPD_val.txt'
    if not os.path.exists(os.path.dirname(train_list_file_path)):
        os.makedirs(os.path.dirname(train_list_file_path))
    fout_train = open(train_list_file_path, 'w')
    fout_val = open(val_list_file_path,'w')
    fout_test = open(test_list_file_path, 'w')
    # train_proportion:标注数据中用于train的比例 -取值 0到1之间
    train_proportion = 0.7
    val_proportion = 0.1
    # 训练图片的序数
    train_counter = 0
    test_counter = 0
    val_counter=0
    for root in image_roots:
        print(root)
        # 图片的名称列表
        file_name_list = [name for name in os.listdir(root) if name.endswith('.jpg')]
        # shuffle 重新随机图片列表
        random.shuffle(file_name_list)
        # 训练集 和测试集 数据
        file_name_list_train = file_name_list[:int(len(file_name_list) * train_proportion)]
        file_name_list_val = file_name_list[int(len(file_name_list) * train_proportion):int(len(file_name_list)*(train_proportion+val_proportion))]
        file_name_list_test = file_name_list[int(len(file_name_list)*(train_proportion+val_proportion)):]

        for file_name in file_name_list_train:
            location_annotation = annotation_from_name(file_name)
            line = os.path.join(root, file_name) + ',1,1,' + str(location_annotation[0]) + ',' + str(
                location_annotation[1]) + ',' + str(location_annotation[2]) + ',' + str(location_annotation[3]) \
                   + ',' + str(location_annotation[4]) + ',' + str(location_annotation[5]) + ',' + str(
                location_annotation[6]) + ',' + str(location_annotation[7]) + ',' + str(location_annotation[8]) \
                   + ',' + str(location_annotation[9]) + ',' + str(location_annotation[10]) + ',' + str(
                location_annotation[11])

            shutil.copy(os.path.join(root, file_name), os.path.join(train_dir, file_name))

            fout_train.write(line + '\n')
            train_counter += 1
            print(train_counter)

        for file_name in file_name_list_val:
            location_annotation = annotation_from_name(file_name)
            line = os.path.join(root, file_name) + ',1,1,' + str(location_annotation[0]) + ',' + str(
                location_annotation[1]) + ',' + str(location_annotation[2]) + ',' + str(location_annotation[3]) \
                   + ',' + str(location_annotation[4]) + ',' + str(location_annotation[5]) + ',' + str(
                location_annotation[6]) + ',' + str(location_annotation[7]) + ',' + str(location_annotation[8]) \
                   + ',' + str(location_annotation[9]) + ',' + str(location_annotation[10]) + ',' + str(
                location_annotation[11])

            shutil.copy(os.path.join(root, file_name), os.path.join(val_dir, file_name))

            fout_val.write(line + '\n')
            val_counter += 1
            print(val_counter)

        for file_name in file_name_list_test:
            location_annotation = annotation_from_name(file_name)
            print("test", file_name)
            line = os.path.join(root, file_name) + ',1,1,' + str(location_annotation[0]) + ',' + str(
                location_annotation[1]) + ',' + str(location_annotation[2]) + ',' + str(location_annotation[3]) \
                   + ',' + str(location_annotation[4]) + ',' + str(location_annotation[5]) + ',' + str(
                location_annotation[6]) + ',' + str(location_annotation[7]) + ',' + str(location_annotation[8]) \
                   + ',' + str(location_annotation[9]) + ',' + str(location_annotation[10]) + ',' + str(
                location_annotation[11])
            shutil.copy(os.path.join(root, file_name), os.path.join(test_dir, file_name))

            fout_test.write(line + '\n')
            test_counter += 1
            print(test_counter)

    fout_train.close()
    fout_test.close()
    fout_val.close()


if __name__ == '__main__':
    # 返回值 [image absolute path],[pos/neg flag],[num of bboxes],[x1],[y1],[width1],[height1],[x2],[y2],[width2],[height2]......
    generate_data_list(ccpd_path=r'D:\traffic_detection\ccpd')
