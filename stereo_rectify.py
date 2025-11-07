import cv2
import numpy as np
from stereo_math import (
    is_rotation_matrix,
    compute_rotation_matrix,
    calculate_bounding_box,
    process_camera_coordinates,
    update_intrinsic_matrix
)


def rectification(R, T, delta_angle, R1, R2):
    """执行旋转，进行双目图像的旋转矩阵校正"""
    # 计算旋转矩阵（基于平移向量T）
    R_rect = compute_rotation_matrix(T)
    # 更新R1和R2的旋转矩阵
    R1 = R_rect @ R @ R1
    R2 = R_rect @ R2

    # 调整旋转角度，通过delta_angle调整
    R_adjust = np.array([
        [1, 0, 0],  # X轴不变
        [0, np.cos(delta_angle), -np.sin(delta_angle)],  # Y轴旋转
        [0, np.sin(delta_angle), np.cos(delta_angle)]  # Z轴旋转
    ])

    # 应用调整后的旋转矩阵
    R1 = R_adjust @ R1
    R2 = R_adjust @ R2

    # 重新计算最终的旋转矩阵和T
    R = R2 @ R @ R1.T  # 旋转矩阵
    T = R2 @ T  # 平移向量

    # 打印旋转矩阵和T的相关信息
    print("校正后旋转矩阵:\n", R)
    print("校正后平移向量:\n", T)
    print("T 向量模长:", np.linalg.norm(T))
    print("旋转矩阵合法性:", is_rotation_matrix(R))

    return R, T, R1, R2  # 返回新的旋转矩阵和T


def rectify_images(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2,
                   R, T, R1, R2, left_path, right_path,
                   corrected_left_path, corrected_right_path,
                   delta_angle, image_magnification):
    """执行完整的双目校正流程"""
    # 首先进行旋转矩阵的校正
    R, T, R1, R2 = rectification(R, T, delta_angle, R1, R2)

    # 计算新的相机内参矩阵（根据左右相机内参的平均值进行缩放）
    newCameraMatrix = cameraMatrix1.copy()
    newCameraMatrix[0, 0] = (cameraMatrix1[0, 0] + cameraMatrix2[0, 0]) / 2 * image_magnification
    newCameraMatrix[1, 1] = (cameraMatrix1[1, 1] + cameraMatrix2[1, 1]) / 2 * image_magnification

    # 加载左右图像
    img_left = cv2.imread(left_path)
    img_right = cv2.imread(right_path)

    # 如果图像加载失败，则抛出错误
    if img_left is None or img_right is None:
        raise FileNotFoundError("图像加载失败，请检查路径")

    # 处理图像的坐标，校正相机坐标
    coords_left = process_camera_coordinates(img_left, cameraMatrix1, R1, newCameraMatrix)
    coords_right = process_camera_coordinates(img_right, cameraMatrix2, R2, newCameraMatrix)

    # 计算左右图像的裁剪边界
    x_min, x_max, y_min, y_max = calculate_bounding_box([coords_left, coords_right])

    # 更新新的内参矩阵
    new_cx = (x_max - x_min) / 2 * image_magnification
    new_cy = (y_max - y_min) / 2 * image_magnification
    newCameraMatrix = update_intrinsic_matrix(newCameraMatrix, new_cx, new_cy)

    print("更新后的内参矩阵:\n", newCameraMatrix)

    # 设置新的图像尺寸（根据裁剪边界和图像放大系数）
    new_size = (int((x_max - x_min) * image_magnification),
                int((y_max - y_min) * image_magnification))

    # 计算左右图像的畸变校正映射
    mapl1, mapl2 = cv2.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, newCameraMatrix, new_size, cv2.CV_32FC1)
    mapr1, mapr2 = cv2.initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, newCameraMatrix, new_size, cv2.CV_32FC1)

    # 应用校正映射，生成校正后的左右图像
    corrected_left = cv2.remap(img_left, mapl1, mapl2, cv2.INTER_NEAREST)
    corrected_right = cv2.remap(img_right, mapr1, mapr2, cv2.INTER_NEAREST)

    # 保存校正后的图像
    cv2.imwrite(corrected_left_path, corrected_left)
    cv2.imwrite(corrected_right_path, corrected_right)

    # 输出完成消息
    print("✅ 双目校正完成！")

    return corrected_left, corrected_right  # 返回校正后的左右图像
