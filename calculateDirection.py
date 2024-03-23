def calculateDirection(outputs, output,id):
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

        x11, y11, x12, y12 = work_prev_locations[0:4]
        x21, y21, x22, y22 = work_locations[0:4]

        w1 = x12 - x11
        h1 = y12 - y11
        w2 = x22 - x21
        h2 = y22 - y21

        cx1, cy1 = x11 + w1 / 2, y11 + h1 / 2
        cx2, cy2 = x21 + w2 / 2, y21 + h2 / 2
        dx = cx2 - cx1
        dy = cy2 - cy1
        if dy > 0 and 3 * abs(dy) >= abs(dx):
            return 'DOWN'
        if dy < 0 and 1.5 * abs(dy) >= abs(dx):
            return 'UP'
        if dx > 0 and abs(dx) >= abs(dy):
            return 'RIGHT'
        if dx < 0 and abs(dx) >= abs(dy):
            return "LEFT"

        return 'UP'
    else:
        return ''
