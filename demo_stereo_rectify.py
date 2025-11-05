import os
import numpy as np
from stereo_io_vis import load_camera_parameters, visualize_rectification
from stereo_rectify import rectify_images


if __name__ == "__main__":
    left_image_path = "../1030/viewpoint1/left.png"
    right_image_path = "../1030/viewpoint1/right.png"
    json_path = "../1030/viewpoint1/params.json"
    output_dir = "../1030_rectified"
    os.makedirs(output_dir, exist_ok=True)

    corrected_left_path = os.path.join(output_dir, "left_rectified.png")
    corrected_right_path = os.path.join(output_dir, "right_rectified.png")

    params = load_camera_parameters(json_path)
    left_K = params['left_K']
    left_dist = params['left_distortion']
    right_K = params['right_K']
    right_dist = params['right_distortion']
    Rt = params['Rt']

    R = Rt[:3, :3]
    T = Rt[:3, 3:4]
    R1 = np.eye(3)
    R2 = np.eye(3)

    delta_angle = np.deg2rad(12)
    image_magnification = 1.0

    corrected_left, corrected_right = rectify_images(
        left_K, left_dist,
        right_K, right_dist,
        R, T, R1, R2,
        left_image_path, right_image_path,
        corrected_left_path, corrected_right_path,
        delta_angle, image_magnification
    )

    vis_path = os.path.join(output_dir, "rectified_pair.png")
    visualize_rectification(corrected_left, corrected_right, vis_path, num_lines=20)
    print(f"\n✅ 校正完成，结果已保存至：{output_dir}")