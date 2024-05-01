import math

import numpy as np


from velocity_cal.distance import *

K, RT = camera_parameters(excel_path)






def calculate_average_speed(car_id, fps, car_locations_history):
    # 确保车辆位置历史中有足够的数据
    if car_id not in car_locations_history or len(car_locations_history[car_id]) < 2:
        return None, False
    # 获取车辆的位置和尺寸历史
    locations_sizes = car_locations_history[car_id]
    total_distance = 0.0
    for i in range(1, len(locations_sizes)):
        prev_location = locations_sizes[i - 1][1]
        current_location = locations_sizes[i][1]
        p1 = np.array([prev_location[0], prev_location[1], 1])
        p2 = np.array([current_location[0], current_location[1], 1])
        # print(current_location)
        """
        location_sizes[i][1] 为中心点坐标
        """
        # real_distance_prev = pixel_coordinate_transform(prev_location[1])
        # real_distance_curr = pixel_coordinate_transform(current_location[1])
        total_distance += comp_distance(RT=RT,K=K,p1=p1,p2=p2)

    # 总时间
    total_time_seconds = (locations_sizes[-1][0] - locations_sizes[0][0]) / fps
    print(f"id:{car_id},box:{current_location},distance:{total_distance},time:{total_time_seconds}")


    # 平均速度
    average_speed_kmh = (total_distance / total_time_seconds) * 3.6


    # 检测是否超速
    SpeedOver = average_speed_kmh > 72  # 假定超过60km/h为超速
    return average_speed_kmh, SpeedOver



