# 取的是隔一帧的距离   后续可以增加cint参数  取多几帧距离

import math


def Estimated_speed(outputs, output, id, fps, cint, flag=0):
    """
    @param: outputs: 前一帧的位置信息 带id 二维的
            output： 当前帧位置信息  无id信息
            cint: 每n帧检测一次
    """
    SpeedOver = False
    prev_IDs = []  # 之前的ids
    work_IDs = []  # 有效的ids
    work_locations = output  # 当前帧数据：中心点x坐标、中心点y坐标、目标序号、车辆类别、车辆像素宽度
    work_prev_locations = []  # 上一帧数据，数据格式相同
    for i in range(len(outputs)):
        prev_IDs.append(outputs[i][4])  # 获得前一帧中跟踪到车辆的ID

    for m, n in enumerate(prev_IDs):  # 进行筛选，找到在两帧图像中均被检测到的有效车辆ID，存入work_IDs中
        if id == n:
            work_IDs.append(m)
            work_prev_locations = outputs[m]  # 将当前帧有效检测车辆的信息存入work_locations中

    if len(work_IDs) > 0:
        locations = [0, 0]
        prev_locations = [0, 0]  # 存放中心点坐标
        bbox_prev = work_prev_locations[0:4]
        bbox = work_locations[0:4]
        # p1 左上角坐标 p2 右下角坐标
        p1, p2 = (int(bbox_prev[0]), int(bbox_prev[1])), (int(bbox_prev[2]), int(bbox_prev[3]))

        w1 = bbox_prev[2] - bbox_prev[0]
        h1 = bbox_prev[3] - bbox_prev[1]
        w2 = bbox[2] - bbox[0]
        h2 = bbox[3] - bbox[1]

        # 取两次宽度高度平均值 作为最终宽度高度
        width = (w1 + w2) / 2
        height = (h1 + h2) / 2

        prev_locations[0] = (p2[0] - p1[0]) / 2 + p1[0]
        prev_locations[1] = (p2[1] - p1[1]) / 2 + p1[1]

        x1, x2 = (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3]))

        locations[0] = (x2[0] - x1[0]) / 2 + x1[0]
        locations[1] = (x2[1] - x1[1]) / 2 + x1[1]

        d = math.sqrt((locations[0] - prev_locations[0]) ** 2 + (locations[1] - prev_locations[1]) ** 2)

        # flag 1 代表车流是横着的 height是车的长度 长度选取1.8 flag 0 车流垂直 width为车身宽度 真实值选取2
        if w1 / h1 >= 2 or w2 / h2 >= 2 or flag == 1:
            dpix = 1.8 / height
        else:
            dpix = 2 / width * 3  # 像素真实距离比 车宽2 *3为修正值

        speed = d * dpix * 3.6 * fps / cint

        if speed > 40:
            SpeedOver = True

        return speed, SpeedOver
    return " ", SpeedOver
