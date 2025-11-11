import numpy as np
import cv2


def cross_product(a, b):
    """
    计算两个 3x1 列向量的叉乘结果。

    功能说明：
        叉乘是三维空间中用于求解两个向量垂直方向（法向量）的运算。
        在双目几何计算中，它常用于确定旋转矩阵的正交坐标轴方向。

    参数：
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
    all_coords = np.array(all_coords, dtype=np.float64)

    # 自动修正：若是一维（例如 [x1,y1,x2,y2,...]），则 reshape 成 Nx2
    if all_coords.ndim == 1:
        if len(all_coords) % 2 != 0:
            raise ValueError(f"输入坐标长度 {len(all_coords)} 不是偶数，无法reshape为(N,2)。")
        all_coords = all_coords.reshape(-1, 2)

    # 若是多组坐标 [[(x,y)...], [(x,y)...]]，展平
    if all_coords.ndim == 3:
        all_coords = all_coords.reshape(-1, 2)

    # 最终检查形状
    if all_coords.ndim != 2 or all_coords.shape[1] != 2:
        raise ValueError(f"输入坐标形状错误: {all_coords.shape}, 应为 (N, 2)")

    x_min, y_min = np.min(all_coords, axis=0)
    x_max, y_max = np.max(all_coords, axis=0)
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
        np.ndarray: 投影后角点的像素坐标，形状为 (N, 2)
                    [[x1', y1'], [x2', y2'], ...]
    """
    h, w = image.shape[:2]

    # 定义原图像四个角点
    corners = np.array([[0, 0],
                        [w, 0],
                        [0, h],
                        [w, h]], dtype=np.float64)

    # 计算原相机矩阵的逆矩阵
    cameraMatrixInv = np.linalg.inv(cameraMatrix)

    projected = []
    for x, y in corners:
        p = np.array([[x], [y], [1.0]])
        cam_coord = R @ (cameraMatrixInv @ p)
        proj = newCameraMatrix @ cam_coord
        x_proj = proj[0, 0] / proj[2, 0]
        y_proj = proj[1, 0] / proj[2, 0]
        projected.append([x_proj, y_proj])

    # ⚠️ 转为 NumPy 数组，确保形状为 (N,2)
    projected = np.array(projected, dtype=np.float64)
    return projected

def unified_intrinsics_rectify(
    left_path: str,
    right_path: str,
    cameraMatrix1: np.ndarray,
    cameraMatrix2: np.ndarray,
    R1: np.ndarray,
    R2: np.ndarray,
    res_scale: float = 1.0,
    fov_scale: float = 1.0
):
    """
    生成自适应完整视场投影，使左右图像分辨率相等且 fx, fy 完全一致。

    功能说明：
        - 自动根据左右相机的参数，生成一个新的统一相机矩阵。
        - fx, fy 取两台相机内参中 fx1, fx2, fy1, fy2 的平均值。
        - 图像分辨率由视场和分辨率因子自动决定(以光轴与图像坐标系交点为中心缩放)：
            - res_scale：
                - 分辨率缩放因子:
                - 仅作用在 fx, fy 上，
                - 由自适应机制导致分辨率同步变化；
                - 图片内容实际不变
            - fov_scale：
                - 视场缩放因子：
                - 作用在图像尺寸上
                - 传感器像素大小不变，尺寸由中心向内收缩
                - 视场变小，图片内容放大

    参数:
        left_path (str): 左图路径
        right_path (str): 右图路径
        cameraMatrix1 (np.ndarray): 左相机内参矩阵
        cameraMatrix2 (np.ndarray): 右相机内参矩阵
        R1, R2 (np.ndarray): 左右旋转矩阵 (3x3)
        process_camera_coordinates (function): 角点投影函数
        calculate_bounding_box (function): 边界框计算函数
        update_intrinsic_matrix (function): 内参更新函数
        res_scale (float): 分辨率调整因子（仅作用在 fx, fy 上）
        fov_scale (float): 视场调整因子（仅作用在图像尺寸和光心确定上）

    返回:
        tuple:
            newCameraMatrix (np.ndarray): 更新后的统一内参矩阵
            new_size (tuple): 新图像尺寸 (width, height)
            (coords_left, coords_right): 投影后的角点坐标
    """

    # -------------------------------
    # 1️⃣ 计算平均焦距 fx, fy
    # -------------------------------
    fx_avg = (cameraMatrix1[0, 0] + cameraMatrix2[0, 0]) / 2
    fy_avg = (cameraMatrix1[1, 1] + cameraMatrix2[1, 1]) / 2
    fx_fy_mean = (fx_avg + fy_avg) / 2  # 四个值的平均

    # -------------------------------
    # 2️⃣ 构造新的相机矩阵
    # -------------------------------
    newCameraMatrix = np.eye(3, dtype=np.float64)
    newCameraMatrix[0, 0] = fx_fy_mean * res_scale
    newCameraMatrix[1, 1] = fx_fy_mean * res_scale

    # -------------------------------
    # 3️⃣ 读取左右图像
    # -------------------------------
    img_left = cv2.imread(left_path)
    img_right = cv2.imread(right_path)

    if img_left is None or img_right is None:
        raise FileNotFoundError("❌ 图像加载失败，请检查输入路径！")

    # -------------------------------
    # 4️⃣ 投影角点到新相机坐标系
    # -------------------------------
    coords_left = process_camera_coordinates(img_left, cameraMatrix1, R1, newCameraMatrix)
    coords_right = process_camera_coordinates(img_right, cameraMatrix2, R2, newCameraMatrix)

    # -------------------------------
    # 5️⃣ 计算完整视场边界
    # -------------------------------
    x_min, x_max, y_min, y_max = calculate_bounding_box([coords_left, coords_right])

    # -------------------------------
    # 6️⃣ 计算新的光心位置(以图像坐标系中心缩放fov_scale)
    # -------------------------------
    new_cx = (x_max - x_min) * fov_scale/2 - (x_max + x_min)/2
    new_cy = (y_max - y_min) / 2 * fov_scale
    newCameraMatrix = update_intrinsic_matrix(newCameraMatrix, new_cx, new_cy)

    print("✅ 更新后的统一内参矩阵:\n", newCameraMatrix)

    # -------------------------------
    # 7️⃣ 计算新图像尺寸（自适应机制不受res_scale影响，受fov_scale影响）
    # -------------------------------
    width = int((x_max - x_min) * fov_scale)
    height = int((y_max - y_min) * fov_scale)
    new_size = (width, height)

    print(f"✅ 新图像分辨率: {new_size} (res_scale={res_scale}, fov_scale={fov_scale})")

    return (newCameraMatrix, newCameraMatrix), new_size, (coords_left, coords_right)

def unified_dual_intrinsics_rectify(
    left_path: str,
    right_path: str,
    cameraMatrix1: np.ndarray,
    cameraMatrix2: np.ndarray,
    R1: np.ndarray,
    R2: np.ndarray,
    res_scale: float = 1.0,
    fov_scale: float = 1.0
):
    """
    生成自适应完整视场投影（左右光心独立、分辨率统一）。

    功能说明：
        - 自动根据左右相机的参数生成两个独立的新相机矩阵（左右光心独立计算）。
        - fx, fy 完全相同，取两相机平均值。
        - 分辨率保证左右图一致（取左右各自FOV范围的最大宽高）。
        - 光心位置计算规则：
            * 左图光心 cx_left = (x1_max - x1_min) * fov_scale/2 - (x1_max + x1_min)/2
            * 右图光心 cx_right = (x2_max - x2_min) * fov_scale/2 - (x2_max + x2_min)/2
            * 两图光心的 cy 相同 = 新图像高度的一半（保证水平对齐）

    参数:
        left_path (str): 左图路径
        right_path (str): 右图路径
        cameraMatrix1 (np.ndarray): 左相机内参矩阵
        cameraMatrix2 (np.ndarray): 右相机内参矩阵
        R1, R2 (np.ndarray): 左右旋转矩阵 (3x3)
        res_scale (float): 分辨率缩放因子（仅作用在 fx, fy 上）
        fov_scale (float): 视场缩放因子（仅作用在图像尺寸和光心确定上）

    返回:
        tuple:
            (newCameraMatrix_left, newCameraMatrix_right): 左右新相机矩阵
            new_size (tuple): 统一新图像尺寸 (width, height)
            (coords_left, coords_right): 投影后的角点坐标
    """

    # -------------------------------
    # 1️⃣ 计算平均焦距 fx, fy
    # -------------------------------
    fx_avg = (cameraMatrix1[0, 0] + cameraMatrix2[0, 0]) / 2
    fy_avg = (cameraMatrix1[1, 1] + cameraMatrix2[1, 1]) / 2
    fx_fy_mean = (fx_avg + fy_avg) / 2

    # -------------------------------
    # 2️⃣ 构造新的基础相机矩阵（左右共用 fx, fy）
    # -------------------------------
    baseK = np.eye(3, dtype=np.float64)
    baseK[0, 0] = fx_fy_mean * res_scale
    baseK[1, 1] = fx_fy_mean * res_scale

    # -------------------------------
    # 3️⃣ 读取左右图像
    # -------------------------------
    img_left = cv2.imread(left_path)
    img_right = cv2.imread(right_path)
    if img_left is None or img_right is None:
        raise FileNotFoundError("❌ 图像加载失败，请检查输入路径！")

    # -------------------------------
    # 4️⃣ 投影角点到新相机坐标系
    # -------------------------------
    coords_left = process_camera_coordinates(img_left, cameraMatrix1, R1, baseK)
    coords_right = process_camera_coordinates(img_right, cameraMatrix2, R2, baseK)

    # -------------------------------
    # 5️⃣ 计算左右各自边界
    # -------------------------------
    x1_min, x1_max, y1_min, y1_max = calculate_bounding_box(coords_left)
    x2_min, x2_max, y2_min, y2_max = calculate_bounding_box(coords_right)

    # -------------------------------
    # 6️⃣ 计算统一的图像分辨率（取最大宽高）
    # -------------------------------
    max_width = max(x1_max - x1_min, x2_max - x2_min)
    max_height = max(y1_max - y1_min, y2_max - y2_min)

    width = int(max_width * fov_scale)
    height = int(max_height * fov_scale)
    new_size = (width, height)

    # -------------------------------
    # 7️⃣ 分别计算左右光心位置
    # -------------------------------
    cx_left = (x1_max - x1_min) * fov_scale/2 - (x1_max + x1_min)/2
    cx_right = (x2_max - x2_min) * fov_scale/2 - (x2_max + x2_min)/2
    cy = height / 2  # 保证水平对齐

    # -------------------------------
    # 8️⃣ 构造左右相机新内参矩阵
    # -------------------------------
    newCameraMatrix_left = baseK.copy()
    newCameraMatrix_left[0, 2] = cx_left
    newCameraMatrix_left[1, 2] = cy

    newCameraMatrix_right = baseK.copy()
    newCameraMatrix_right[0, 2] = cx_right
    newCameraMatrix_right[1, 2] = cy

    # -------------------------------
    # 9️⃣ 输出调试信息
    # -------------------------------
    print("✅ 左相机新内参矩阵:\n", newCameraMatrix_left)
    print("✅ 右相机新内参矩阵:\n", newCameraMatrix_right)
    print(f"✅ 统一新图像尺寸: {new_size} (res_scale={res_scale}, fov_scale={fov_scale})")
    print(f"✅ 左右光心坐标: cx_left={cx_left:.2f}, cx_right={cx_right:.2f}, cy={cy:.2f}")

    return (newCameraMatrix_left, newCameraMatrix_right), new_size, (coords_left, coords_right)

