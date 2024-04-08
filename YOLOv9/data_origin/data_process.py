import os
import random
import shutil

image_path=r'D:\yolov7\data_origin\images'
label_path=r'D:\yolov7\data_origin\labels'
new_file_path=r'D:\yolov7\mydata'
def split_data(image_path,label_path,new_file_path,train_rate,val_rate,test_rate):
    '''
    image_path:原文件图片位置
    label_path：原标签位置
    new_file_path：数据分隔后的新位置 这里是data 放入train val test三个文件夹中
    train_rate:训练集占比
    val_rate:验证集占比
    test_rate:测试集占比
    '''
    # 读取数据后，将两种文件通过zip函数绑定
    each_class_image = []
    each_class_label = []
    for image in os.listdir(image_path):
        each_class_image.append(image)
    for label in os.listdir(label_path):
        each_class_label.append(label)
    total=len(each_class_image)
    data=list(zip(each_class_image,each_class_label))
    #打乱顺序，再将两个列表分开
    random.shuffle(data)
    each_class_image,each_class_label=zip(*data)
    #三个列表储存一下图片和标注文件的元素
    train_images=each_class_image[0:int(train_rate*total)]
    val_images = each_class_image[int(train_rate * total):int((train_rate+val_rate)*total)]
    test_images = each_class_image[int((train_rate+val_rate) * total):]

    train_labels=each_class_label[0:int(train_rate*total)]
    val_labels = each_class_label[int(train_rate * total):int((train_rate+val_rate)*total)]
    test_labels = each_class_label[int((train_rate+val_rate) * total):]
    #写入新文件夹
    for image in train_images:
        #print(image)
        old_path = os.path.join(image_path,image)
        new_path1 = os.path.join(new_file_path,'train','images')
        if not os.path.exists(new_path1):
            os.makedirs(new_path1)
        new_path=os.path.join(new_path1,image)
        shutil.copy(old_path, new_path)

    for label in train_labels:
        #print(label)
        old_path = os.path.join(label_path,label)
        new_path1 = os.path.join(new_file_path,'train','labels')
        if not os.path.exists(new_path1):
            os.makedirs(new_path1)
        new_path = os.path.join(new_path1 , label)
        shutil.copy(old_path, new_path)

    for image in val_images:
        old_path = os.path.join(image_path,image)
        new_path1 = os.path.join(new_file_path,'val','images')
        if not os.path.exists(new_path1):
            os.makedirs(new_path1)
        new_path = os.path.join(new_path1,image)
        shutil.copy(old_path, new_path)

    for label in val_labels:
        old_path = os.path.join(label_path,label)
        new_path1 = os.path.join(new_file_path,'val','labels')
        if not os.path.exists(new_path1):
            os.makedirs(new_path1)
        new_path = os.path.join(new_path1 , label)
        shutil.copy(old_path, new_path)

    for image in test_images:
        old_path = os.path.join(image_path,image)
        new_path1 = os.path.join(new_file_path ,'test' , 'images')
        if not os.path.exists(new_path1):
            os.makedirs(new_path1)
        new_path =  os.path.join(new_path1,image)
        shutil.copy(old_path, new_path)

    for label in test_labels:
        old_path = os.path.join(label_path,label)
        new_path1 = os.path.join(new_file_path,'test','labels')
        if not os.path.exists(new_path1):
            os.makedirs(new_path1)
        new_path = os.path.join(new_path1 , label)
        shutil.copy(old_path, new_path)

if __name__ == '__main__':
    split_data(image_path,label_path,new_file_path,train_rate=0.7,val_rate=0.1,test_rate=0.2)
