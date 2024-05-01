import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

excel_path = r'velocity_cal/camera_parameters.xlsx'




# 计算旋转矩阵
def rotate_Matrix(*, p0x, p0y, K):
    """
    计算从图像坐标系到世界坐标系的旋转矩阵。
    
    参数:
    - p0x: 消失点在图像中的x坐标。
    - p0y: 消失点在图像中的y坐标。
    - K: 相机的内参矩阵。
    
    返回:
    - a: 俯仰角。
    - b: 偏航角。
    - R: 计算得到的旋转矩阵。
    """
    K_1 = np.linalg.inv(K)
    P0 = np.array([p0x, p0y, 1])
    r3 = np.dot(K_1, P0) / np.linalg.norm(np.dot(K_1, P0))
    a = -math.asin(r3[1])
    b = math.atan(r3[0] / r3[2])
    R = np.array([
        [math.cos(b), math.sin(a) * math.sin(b), math.cos(a) * math.sin(b)],
        [0, math.cos(a), -math.sin(a)],
        [-math.sin(b), math.cos(b) * math.sin(a), math.cos(a) * math.cos(b)]
    ])
    return a, b, R


def camera_parameters(excel_path):
    """
    从Excel文件加载相机的内参和外参矩阵。
    参数:
    excel_path (str): 包含相机参数的Excel文件路径。
    返回:
    tuple: 包含外参矩阵和内参矩阵的元组。
    """
    # Load Intrinsics matrix of Camera
    df_intrinsic = pd.read_excel(excel_path, sheet_name='内参矩阵', header=None)
    df_extrinsic = pd.read_excel(excel_path, sheet_name='外参矩阵', header=None)

    return df_intrinsic.values, df_extrinsic.values


# 计算外参矩阵
def RT_Matrix(*, R, h):
    """
    根据旋转矩阵和摄像机高度构造外参矩阵。
    
    参数:
    - R: 旋转矩阵。
    - h: 摄像机相对于基准平面的高度。
    
    返回:
    - Mrt: 完整的外参矩阵。
    """
    T = np.array([0, h, 0])
    temp = np.concatenate((R, T.reshape(-1, 1)), axis=1)
    Mrt = np.concatenate((temp, np.array([[0, 0, 0, 1]])), axis=0)
    return Mrt


# 计算真实世界坐标
def comp_real_position(*, Mrt, K, h, pos):
    """
    根据外参矩阵、内参矩阵和点的像素坐标计算其在三维空间中的真实世界坐标。
    
    参数:
    - Mrt: 外参矩阵。
    - K: 内参矩阵。
    - h: 摄像机的高度。
    - pos: 点的像素坐标（齐次坐标形式）。
    
    返回:
    - res: 点在三维空间中的坐标。
    """
    # 在K矩阵的右侧加上一列全0，形成一个新的3x4矩阵K_E。
    K_E = np.concatenate((K, np.array([0, 0, 0]).reshape(-1, 1)), axis=1)
    # 将增强后的内参矩阵K_E与外参矩阵Mrt相乘，得到一个3x4矩阵M2。
    M2 = np.dot(K_E, Mrt)
    # 从M2矩阵中提取各个列向量，用于构建三维到二维点映射的方程。
    # r1, r2, r3是转换矩阵的前三列，r4是平移向量。
    r1 = M2[:, 0]
    r2 = M2[:, 1]
    r3 = M2[:, 2]
    r4 = M2[:, 3]
    M3 = np.concatenate((r1.reshape(-1, 1), pos.reshape(-1, 1), r3.reshape(-1, 1)), axis=1)
    r5 = -1 * h * r2 - r4
    res = np.dot(np.linalg.inv(M3), r5)
    return res


def comp_distance(*, RT, K, p1, p2, h=4000):
    p1 = comp_real_position(Mrt=RT, K=K, h=h, pos=p1)
    p2 = comp_real_position(Mrt=RT, K=K, h=h, pos=p2)
    AB = np.linalg.norm(p2 - p1) / 1000
    return AB





def evaluate_accuracy(point_pairs, real_distances,K,RT):
    """
    评估精度，计算估计距离和实际距离之间的误差，并进行可视化

    参数:
    - point_pairs: 点对列表，每个元素为((x1, y1), (x2, y2))形式
    - real_distances: 真实距离列表，与点对列表一一对应
    """
    estimated_distances = []
    errors = []
    for (p1, p2), real_dist in zip(point_pairs, real_distances):
        p1 = np.array([p1[0], p1[1], 1])
        p2 = np.array([p2[0], p2[1], 1])
        estimated_dist = comp_distance(K=K,RT=RT,p1=p1,p2=p2)
        error = abs(estimated_dist - real_dist)
        errors.append(error)
        estimated_distances.append(estimated_dist)
        print(f"点对 {p1} 和 {p2} 的估计距离为 {estimated_dist:.3f} 米, 真实距离为 {real_dist} 米, 误差为 {error:.3f} 米")

    plt.figure(figsize=(10, 5))
    indices = np.arange(len(point_pairs))
    plt.plot(indices, real_distances, 'o-', label='Real Distance', markersize=8, color='blue', linewidth=2)
    plt.plot(indices, estimated_distances, 'x-', label='Estimated Distance', markersize=8, color='red', linewidth=2)
    plt.plot(indices, errors, 's-', label='Error (Est - Real)', markersize=8, color='green', linewidth=2)
    plt.xlabel('Point Pair Index')
    plt.ylabel('Distance (m) / Error (m)')
    plt.title('Comparison of Estimated Distances and Errors')
    plt.legend()
    average_error = np.mean(errors)

    plt.annotate(f'Average Error: {average_error:.3f} m', xy=(0.5, 0.5), xycoords='axes fraction',
                 horizontalalignment='center', verticalalignment='center', fontsize=12, color='purple')

    plt.grid(True)
    plt.tight_layout()
    plt.savefig('velocity_cal/distance_error.png')
    plt.show()


    print(f"平均误差: {average_error:.3f} 米")






if __name__ == '__main__':


    K, RT = camera_parameters(excel_path)
    print(K)
    print(RT)
    point_pairs = [((419, 687), (435, 613)),((630, 689),(576, 469)),((306, 621),(335, 621)), ((474, 578),(548, 578)),((447, 541), (463, 466)),((624, 501),(633, 501)),((557, 379),(560, 392))]
    real_distances = [6.0, 30,0.44,1.8,15.0,0.3,6]
    evaluate_accuracy(point_pairs, real_distances,K=K,RT=RT)
