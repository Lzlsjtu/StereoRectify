import cv2
import numpy as np
import json
from stereo_math import (
    is_rotation_matrix,
    compute_rotation_matrix,
    calculate_bounding_box,
    unified_intrinsics_rectify,
    unified_dual_intrinsics_rectify
)


def save_rectification_params(json_path, camera_matrix_left, camera_matrix_right, 
                             R, T, image_size, rectification_type="unified"):
    """
    保存校正后的相机参数到JSON文件（不包含畸变参数）
    
    参数:
        json_path (str): JSON文件保存路径
        camera_matrix_left, camera_matrix_right (np.ndarray): 校正后的左右相机内参矩阵
        R, T (np.ndarray): 校正后的旋转矩阵和平移向量
        image_size (tuple): 图像尺寸 (width, height)
        rectification_type (str): 校正类型标识
    """
    # 计算基线长度
    baseline = float(np.linalg.norm(T))
    
    # 提取焦距和主点坐标
    fx_left, fy_left = camera_matrix_left[0, 0], camera_matrix_left[1, 1]
    cx_left, cy_left = camera_matrix_left[0, 2], camera_matrix_left[1, 2]
    
    fx_right, fy_right = camera_matrix_right[0, 0], camera_matrix_right[1, 1]
    cx_right, cy_right = camera_matrix_right[0, 2], camera_matrix_right[1, 2]
    
    # 构建参数字典
    params = {
        "rectification_type": rectification_type,
        "timestamp": np.datetime64('now').astype(str),
        "image_size": {
            "width": int(image_size[0]),
            "height": int(image_size[1])
        },
        "left_camera": {
            "camera_matrix": camera_matrix_left.tolist(),
            "focal_length": {
                "fx": float(fx_left),
                "fy": float(fy_left)
            },
            "principal_point": {
                "cx": float(cx_left),
                "cy": float(cy_left)
            }
        },
        "right_camera": {
            "camera_matrix": camera_matrix_right.tolist(),
            "focal_length": {
                "fx": float(fx_right),
                "fy": float(fy_right)
            },
            "principal_point": {
                "cx": float(cx_right),
                "cy": float(cy_right)
            }
        },
        "extrinsic_params": {
            "rotation_matrix": R.tolist(),
            "translation_vector": T.tolist(),
            "baseline": baseline
        },
        "stereo_config": {
            "is_rectified": True,
            "has_distortion": False,
            "baseline_meters": baseline
        }
    }
    
    # 保存到JSON文件
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(params, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 校正参数已保存到: {json_path}")
    print(f"📊 基线长度: {baseline:.6f}")
    print(f"📐 左相机焦距: fx={fx_left:.2f}, fy={fy_left:.2f}")
    print(f"📐 右相机焦距: fx={fx_right:.2f}, fy={fy_right:.2f}")


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


def unified_rectify_images(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2,
                   R, T, R1, R2, left_path, right_path,
                   corrected_left_path, corrected_right_path,
                   delta_angle, res_scale=1.0, fov_scale=1.0, 
                   json_save_path=None):
    """执行完整的双目校正流程"""
    # 首先进行旋转矩阵的校正
    R, T, R1, R2 = rectification(R, T, delta_angle, R1, R2)

    (newCameraMatrix_left, newCameraMatrix_right), new_size, (
    coords_left, coords_right) = unified_intrinsics_rectify(
        left_path, right_path,
        cameraMatrix1, cameraMatrix2,
        R1, R2,
        res_scale=res_scale, fov_scale=fov_scale
    )

    # 输出角点边界信息
    x1_min, x1_max, y1_min, y1_max = calculate_bounding_box(coords_left)
    x2_min, x2_max, y2_min, y2_max = calculate_bounding_box(coords_right)
    print(f"x1_min, x1_max, y1_min, y1_max: {x1_min}, {x1_max}, {y1_min}, {y1_max}")
    print(f"x2_min, x2_max, y2_min, y2_max: {x2_min}, {x2_max}, {y2_min}, {y2_max}")

    # ==============================
    # 4️⃣ 计算畸变校正映射
    # ==============================
    mapl1, mapl2 = cv2.initUndistortRectifyMap(
        cameraMatrix1, distCoeffs1, R1, newCameraMatrix_left, new_size, cv2.CV_32FC1
    )
    mapr1, mapr2 = cv2.initUndistortRectifyMap(
        cameraMatrix2, distCoeffs2, R2, newCameraMatrix_right, new_size, cv2.CV_32FC1
    )

    # ==============================
    # 5️⃣ 生成校正图像
    # ==============================
    img_left = cv2.imread(left_path)
    img_right = cv2.imread(right_path)
    corrected_left = cv2.remap(img_left, mapl1, mapl2, cv2.INTER_NEAREST)
    corrected_right = cv2.remap(img_right, mapr1, mapr2, cv2.INTER_NEAREST)

    # 保存结果
    cv2.imwrite(corrected_left_path, corrected_left)
    cv2.imwrite(corrected_right_path, corrected_right)

    # ==============================
    # 6️⃣ 保存校正参数到JSON文件（不包含畸变参数）
    # ==============================
    if json_save_path:
        save_rectification_params(
            json_path=json_save_path,
            camera_matrix_left=newCameraMatrix_left,
            camera_matrix_right=newCameraMatrix_right,
            R=R, T=T,
            image_size=new_size,
            rectification_type="unified"
        )

    print("✅ 双目校正完成！")

    return corrected_left, corrected_right


def unified_dual_rectify_images(
    cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2,
    R, T, R1, R2, left_path, right_path,
    corrected_left_path, corrected_right_path,
    delta_angle, res_scale=1.0, fov_scale=1.0, json_save_path=None
):
    """
    执行完整的双目校正流程（左右独立光心、统一焦距、相同分辨率）

    参数:
        json_save_path (str): JSON参数保存路径，如果为None则不保存
        其他参数保持不变...
    """
    # -------------------------------
    # 1️⃣ 校正旋转矩阵
    # -------------------------------
    R, T, R1, R2 = rectification(R, T, delta_angle, R1, R2)

    # -------------------------------
    # 2️⃣ 生成自适应双目统一内参与尺寸
    # -------------------------------
    (newCameraMatrix_left, newCameraMatrix_right), new_size, (coords_left, coords_right) = (
        unified_dual_intrinsics_rectify(
        left_path=left_path,
        right_path=right_path,
        cameraMatrix1=cameraMatrix1,
        cameraMatrix2=cameraMatrix2,
        R1=R1,
        R2=R2,
        res_scale=res_scale,
        fov_scale=fov_scale
    ))

    # -------------------------------
    # 3️⃣ 输出角点边界信息
    # -------------------------------
    x1_min, x1_max, y1_min, y1_max = calculate_bounding_box(coords_left)
    x2_min, x2_max, y2_min, y2_max = calculate_bounding_box(coords_right)
    print(f"左图边界: x=[{x1_min:.2f},{x1_max:.2f}], y=[{y1_min:.2f},{y1_max:.2f}]")
    print(f"右图边界: x=[{x2_min:.2f},{x2_max:.2f}], y=[{y2_min:.2f},{y2_max:.2f}]")
    print(f"✅ 统一输出分辨率: {new_size}")

    # -------------------------------
    # 4️⃣ 计算左右畸变校正映射
    # -------------------------------
    mapl1, mapl2 = cv2.initUndistortRectifyMap(
        cameraMatrix1, distCoeffs1, R1, newCameraMatrix_left, new_size, cv2.CV_32FC1
    )
    mapr1, mapr2 = cv2.initUndistortRectifyMap(
        cameraMatrix2, distCoeffs2, R2, newCameraMatrix_right, new_size, cv2.CV_32FC1
    )

    # -------------------------------
    # 5️⃣ 应用映射生成校正图像
    # -------------------------------
    img_left = cv2.imread(left_path)
    img_right = cv2.imread(right_path)

    if img_left is None or img_right is None:
        raise FileNotFoundError("❌ 图像加载失败，请检查输入路径！")

    corrected_left = cv2.remap(img_left, mapl1, mapl2, cv2.INTER_NEAREST)
    corrected_right = cv2.remap(img_right, mapr1, mapr2, cv2.INTER_NEAREST)

    # -------------------------------
    # 6️⃣ 保存校正结果
    # -------------------------------
    cv2.imwrite(corrected_left_path, corrected_left)
    cv2.imwrite(corrected_right_path, corrected_right)

    # -------------------------------
    # 7️⃣ 保存校正参数到JSON文件（不包含畸变参数）
    # -------------------------------
    if json_save_path:
        save_rectification_params(
            json_path=json_save_path,
            camera_matrix_left=newCameraMatrix_left,
            camera_matrix_right=newCameraMatrix_right,
            R=R, T=T,
            image_size=new_size,
            rectification_type="unified_dual"
        )

    print("✅ 双目统一内参与校正图像生成完成！")

    return corrected_left, corrected_right