import cv2
import numpy as np
from stereo_math import (
    is_rotation_matrix,
    compute_rotation_matrix,
    calculate_bounding_box,
    process_camera_coordinates,
    update_intrinsic_matrix
)


def iterate_rectification(R, T, delta_angle, R1, R2):
    """执行旋转"""
    R_rect = compute_rotation_matrix(T)
    R1 = R_rect @ R @ R1
    R2 = R_rect @ R2

    R_adjust = np.array([
        [1, 0, 0],
        [0, np.cos(delta_angle), -np.sin(delta_angle)],
        [0, np.sin(delta_angle), np.cos(delta_angle)]
    ])

    R1 = R_adjust @ R1
    R2 = R_adjust @ R2

    R = R2 @ R @ R1.T
    T = R2 @ T

    print("校正后旋转矩阵:\n", R)
    print("校正后平移向量:\n", T)
    print("T 向量模长:", np.linalg.norm(T))
    print("旋转矩阵合法性:", is_rotation_matrix(R))
    return R, T, R1, R2


def rectify_images(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2,
                   R, T, R1, R2, left_path, right_path,
                   corrected_left_path, corrected_right_path,
                   delta_angle, image_magnification):
    """执行完整的双目校正流程"""
    R, T, R1, R2 = iterate_rectification(R, T, delta_angle, R1, R2)

    newCameraMatrix = cameraMatrix1.copy()
    newCameraMatrix[0, 0] = (cameraMatrix1[0, 0] + cameraMatrix2[0, 0]) / 2 * image_magnification
    newCameraMatrix[1, 1] = (cameraMatrix1[1, 1] + cameraMatrix2[1, 1]) / 2 * image_magnification

    img_left = cv2.imread(left_path)
    img_right = cv2.imread(right_path)
    if img_left is None or img_right is None:
        raise FileNotFoundError("图像加载失败，请检查路径")

    coords_left = process_camera_coordinates(img_left, cameraMatrix1, R1, newCameraMatrix)
    coords_right = process_camera_coordinates(img_right, cameraMatrix2, R2, newCameraMatrix)
    x_min, x_max, y_min, y_max = calculate_bounding_box([coords_left, coords_right])

    new_cx = (x_max - x_min) / 2 * image_magnification
    new_cy = (y_max - y_min) / 2 * image_magnification
    newCameraMatrix = update_intrinsic_matrix(newCameraMatrix, new_cx, new_cy)
    print("更新后的内参矩阵:\n", newCameraMatrix)

    new_size = (int((x_max - x_min) * image_magnification),
                int((y_max - y_min) * image_magnification))

    mapl1, mapl2 = cv2.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, newCameraMatrix, new_size, cv2.CV_32FC1)
    mapr1, mapr2 = cv2.initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, newCameraMatrix, new_size, cv2.CV_32FC1)

    corrected_left = cv2.remap(img_left, mapl1, mapl2, cv2.INTER_NEAREST)
    corrected_right = cv2.remap(img_right, mapr1, mapr2, cv2.INTER_NEAREST)

    cv2.imwrite(corrected_left_path, corrected_left)
    cv2.imwrite(corrected_right_path, corrected_right)
    print("✅ 双目校正完成！")

    return corrected_left, corrected_right