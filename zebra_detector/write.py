import os

def write_dataset_txt(dataset_path, output_txt_path):
    '''
    dataset_path: 包含图片文件的文件夹的路径。
    output_txt_path: 输出文本文件的路径，用于保存图片文件的相对或绝对路径。
    '''
    # 初始化一个空列表，用于存储文件路径
    image_paths = []
    
    # 遍历文件夹中的所有文件
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            # 检查文件是否为JPEG图片
            if file.lower().endswith(".jpg"):
                # 将图片文件的路径添加到列表中
                # 使用绝对路径或相对路径根据需要修改
                image_paths.append(os.path.join(root, file))
    
    # 将所有图片文件的路径写入指定的文本文件中
    with open(output_txt_path, 'w') as f:
        for path in image_paths:
            f.write(path + "\n")

    print(f"写入完成: {output_txt_path}")

if __name__ == '__main__':
    # 指定训练集和测试集图片所在的文件夹路径
    train_images_path = r"/hy-tmp/zebra_detector/dataset/train"
    test_images_path = r"/hy-tmp/zebra_detector/dataset/val"
    
    # 指定输出文本文件的路径
    train_txt_path = "/hy-tmp/zebra_detector/dataset/train.txt"
    test_txt_path = "/hy-tmp/zebra_detector/dataset/val.txt"
    
    # 生成train.txt和test.txt
    write_dataset_txt(train_images_path, train_txt_path)
    write_dataset_txt(test_images_path, test_txt_path)
