import numpy as np
import cv2


def cross_product(a, b):
    """
    计算两个 3x1 列向量的叉乘结果。

    功能说明：
        叉乘是三维空间中用于求解两个向量垂直方向（法向量）的运算。
        在双目几何计算中，它常用于确定旋转矩阵的正交坐标轴方向。

    参数:
        a (np.ndarray): 第一个 3x1 列向量
        b (np.ndarray): 第二个 3x1 列向量

    返回:
        np.ndarray: 结果为 3x1 列向量的叉乘结果
    """
    # 检查输入是否为 3x1 的列向量
    if a.shape != (3, 1) or b.shape != (3, 1):
        raise ValueError("输入向量必须为3x1列向量")
    # np.cross 默认返回一维数组，这里 reshape 为列向量格式
    return np.cross(a.reshape(3), b.reshape(3)).reshape(3, 1)


def is_rotation_matrix(R):
    """
    检查给定矩阵是否为有效的旋转矩阵。

    功能说明：
        旋转矩阵必须满足两个条件：
        1. 正交性：R^T * R = I
        2. 行列式 det(R) = 1
        若任一条件不满足，则说明矩阵不是合法旋转矩阵。

    参数:
        R (np.ndarray): 待检测矩阵 (3x3)

    返回:
        bool: 若 R 为有效旋转矩阵返回 True，否则返回 False
    """
    # 维度检查
    if R.shape != (3, 3):
        return False
    # 验证正交性与行列式条件
    return np.allclose(R.T @ R, np.eye(3), atol=1e-6) and np.isclose(np.linalg.det(R), 1.0, atol=1e-6)


def compute_rotation_matrix(T):
    """
    根据平移向量 T，计算将其旋转至 X 轴负方向的旋转矩阵。

    功能说明：
        利用罗德里格斯公式 (Rodrigues Formula) 计算旋转矩阵。
        目标是将当前平移向量 T 对齐到 X 轴负方向 (-1, 0, 0)。

    参数:
        T (np.ndarray): 平移向量，形状为 (3, 1)

    返回:
        np.ndarray: 旋转矩阵 (3x3)
    """
    # 将列向量转换为一维数组
    T_vec = T.flatten()

    # 定义目标方向：X 轴负方向
    target = np.array([-1, 0, 0], dtype=np.float64)

    # 计算旋转轴（T 与目标向量的叉积）
    axis = np.cross(T_vec, target)

    # 计算旋转角度（两向量夹角）
    angle = np.arccos(np.dot(T_vec, target) / (np.linalg.norm(T_vec) * np.linalg.norm(target)))

    # 若叉积结果接近零，说明两向量平行，直接返回单位阵
    if np.linalg.norm(axis) < 1e-8:
        return np.eye(3)

    # 单位化旋转轴
    axis /= np.linalg.norm(axis)

    # Rodrigues 公式：将旋转向量 (轴 * 角度) 转换为旋转矩阵
    R, _ = cv2.Rodrigues(axis * angle)
    return R


def update_intrinsic_matrix(K, cx, cy):
    """
    更新相机内参矩阵中的主点坐标 (cx, cy)。

    功能说明：
        在图像缩放或视图调整时，光心位置可能发生偏移。
        此函数通过修改内参矩阵中的第三列元素更新光心坐标。

    参数:
        K (np.ndarray): 原始相机内参矩阵 (3x3)
        cx (float): 新的光心 x 坐标
        cy (float): 新的光心 y 坐标

    返回:
        np.ndarray: 更新后的相机内参矩阵副本
    """
    # 创建副本，避免修改原矩阵
    K_new = K.copy()
    # 修改光心位置参数
    K_new[0, 2], K_new[1, 2] = cx, cy
    return K_new


def calculate_bounding_box(all_coords):
    """
    计算输入点集的二维边界框 (Bounding Box)。

    功能说明：
        用于计算校正后图像的可视区域范围。
        给定多个点集，返回整体最小包围矩形的上下左右边界。

    参数:
        all_coords (list): 坐标点列表，例如 [[(x1, y1), (x2, y2)], [(x3, y3), (x4, y4)]]

    返回:
        tuple: (x_min, x_max, y_min, y_max)
    """
    # 将所有坐标点展开成二维数组
    all_coords = np.array([p for coords in all_coords for p in coords])
    # 分别计算 x, y 的最小与最大值
    x_min, y_min = all_coords.min(axis=0)
    x_max, y_max = all_coords.max(axis=0)
    return x_min, x_max, y_min, y_max


def process_camera_coordinates(image, cameraMatrix, R, newCameraMatrix):
    """
    将输入图像的角点从原始相机坐标系投影到新的相机坐标系下。

    功能说明：
        通过旋转矩阵 R 和新相机矩阵 newCameraMatrix，
        计算原始图像四个角点 (0,0)、(w,0)、(0,h)、(w,h)
        在新的相机成像平面中的投影坐标。
        该函数用于确定新视图的成像边界。

    参数:
        image (np.ndarray): 输入图像
        cameraMatrix (np.ndarray): 原相机内参矩阵
        R (np.ndarray): 旋转矩阵 (3x3)
        newCameraMatrix (np.ndarray): 新相机内参矩阵

    返回:
        list: 投影后角点的像素坐标 [(x', y'), ...]
    """
    # 获取图像宽高
    h, w = image.shape[:2]

    # 定义原图像的四个角点
    corners = np.array([[0, 0], [w, 0], [0, h], [w, h]], dtype=np.float64)

    # 计算原相机矩阵的逆矩阵（将像素坐标映射到相机坐标系）
    cameraMatrixInv = np.linalg.inv(cameraMatrix)

    projected = []
    for x, y in corners:
        # 将像素坐标表示为齐次坐标形式
        p = np.array([[x], [y], [1.0]])

        # 将像素坐标转为相机坐标，再应用旋转变换
        cam_coord = R @ (cameraMatrixInv @ p)

        # 再投影回新相机平面
        proj = newCameraMatrix @ cam_coord

        # 齐次归一化得到最终像素坐标
        projected.append((proj[0, 0] / proj[2, 0], proj[1, 0] / proj[2, 0]))

    return projected
