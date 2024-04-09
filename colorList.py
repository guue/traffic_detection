import numpy as np
import collections
#定义字典存放颜色分量上下限
#例如：{颜色: [min分量, max分量]}
def getColorList():
    dict = collections.defaultdict(list)
    #红色1
    lower_red = np.array([0, 43, 46])
    upper_red = np.array([10, 255, 255])
    color_list = []
    color_list.append(lower_red)
    color_list.append(upper_red)
    dict['red1']=color_list
    #红色2
    lower_red = np.array([156, 43, 46])
    upper_red = np.array([180, 255, 255])
    color_list = []
    color_list.append(lower_red)
    color_list.append(upper_red)
    dict['red2']=color_list
    #黄色
    lower_yellow = np.array([26, 43, 46])
    upper_yellow = np.array([34, 255, 255])
    color_list = []
    color_list.append(lower_yellow)
    color_list.append(upper_yellow)
    dict['yellow'] = color_list
    #绿色
    lower_green = np.array([78,43,46])
    upper_green = np.array([99,255,255])
    color_list = []
    color_list.append(lower_green)
    color_list.append(upper_green)
    dict['green'] = color_list
    return dict