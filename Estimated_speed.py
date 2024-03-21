import math
def Estimated_speed(outputs, output, id, fps, width):
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
        locations = [0,0]
        prev_locations = [0,0]  # 存放中心点坐标
        bbox_prev = work_prev_locations[0:4]
        bbox = work_locations[0:4]
        p1, p2 = (int(bbox_prev[0]), int(bbox_prev[1])), (int(bbox_prev[2]), int(bbox_prev[3]))
        print(p1)
        prev_locations[0] = (p2[0] - p1[0]) / 2 + p1[0]
        prev_locations[1] = (p2[1] - p1[1]) / 2 + p1[1]
        x1, x2 = (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3]))
        locations[0] = (x2[0] - x1[0]) / 2 + x1[0]
        locations[1] = (x2[1] - x1[1]) / 2 + x1[1]
        d=math.sqrt((locations[0] - prev_locations[0]) ** 2 + (locations[1] - prev_locations[1]) ** 2)

        # speed = (math.sqrt((work_locations[0] - work_prev_locations[0]) ** 2 +  # 计算有效检测车辆的速度，采用线性的从像素距离到真实空间距离的映射
        #                    (work_locations[1] - work_prev_locations[1]) ** 2) *  # 当视频拍摄视角并不垂直于车辆移动轨迹时，测算出来的速度将比实际速度低
        #          width * fps / 5 * 3.6 * 2)
        speed = ((d /width ) *2.5*3.6* fps  )  #像素速度

        if speed > 60:
            SpeedOver = True

        print(speed)
        # speed = 11.3
        speed = str(round(speed, 1)) + "km/h"
        return speed,SpeedOver
    return " ",SpeedOver