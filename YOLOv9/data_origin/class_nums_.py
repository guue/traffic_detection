import os
_txt=r'YOLOv9/dataset_car/train/labels'
_txt1=r'YOLOv9/dataset_car/val/labels'
_txt2=r'YOLOv9/dataset_car/test/labels'
class_=[] #存放所有类别的数组
print("-----------------------训练集---------------------")
for txt in os.listdir(_txt):
    txt_src = _txt +'/'+ str(txt)
    with open(txt_src,'r') as f:
        for line in f:
            colums=line.strip().split()
            class_.append(int(colums[0]))
for i in range(0,4):
    t=class_.count(i)

    print(f'第{i}类总数{t}')

class_val=[] #存放所有类别的数组
print("----------------------验证集------------------------")
for txt in os.listdir(_txt1):
    txt1_src = _txt1 +'/'+ str(txt)
    with open(txt1_src,'r') as f:
        for line in f:
            colums=line.strip().split()
            if int(colums[0])==2:
                print(txt1_src)
            class_val.append(int(colums[0]))
for i in range(0,4):

    t=class_val.count(i)

    print(f'第{i}类总数{t}')

class_test=[] #存放所有类别的数组
print("-----------------------测试集--------------------")
for txt in os.listdir(_txt2):
    txt2_src = _txt2 +'/'+ str(txt)
    with open(txt2_src,'r') as f:
        for line in f:
            colums=line.strip().split()
            class_test.append(int(colums[0]))
for i in range(0,4):
    t=class_test.count(i)

    print(f'第{i}类总数{t}')