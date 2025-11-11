import os
import numpy as np
from stereo_io_vis import load_camera_parameters, visualize_rectification
from stereo_rectify import unified_rectify_images, unified_dual_rectify_images


def run_rectification(mode_name, rectify_func, output_prefix,
                      left_K, left_dist, right_K, right_dist,
                      R, T, R1, R2,
                      left_image_path, right_image_path,
                      output_dir, delta_angle, res_scale, fov_scale,json_save_path):
    """
    通用的双目校正执行函数

    参数说明:
        mode_name (str): 模式名称（用于输出提示）
        rectify_func (function): 校正函数（unified_rectify_images 或 unified_dual_rectify_images）
        output_prefix (str): 输出文件名前缀
        其他参数与原函数相同
    """
    print("\n" + "=" * 40)
    print(f"▶ 模式启动：{mode_name}")
    print("=" * 40)

    # 输出路径设置
    left_output = os.path.join(output_dir, f"{output_prefix}_left.png")
    right_output = os.path.join(output_dir, f"{output_prefix}_right.png")
    vis_output = os.path.join(output_dir, f"{output_prefix}_pair.png")

    # 执行校正
    corrected_left, corrected_right = rectify_func(
        left_K, left_dist,
        right_K, right_dist,
        R, T, R1, R2,
        left_image_path, right_image_path,
        left_output, right_output,
        delta_angle,
        res_scale=res_scale, fov_scale=fov_scale,json_save_path=json_save_path
    )

    # 可视化结果
    visualize_rectification(corrected_left, corrected_right, vis_output, num_lines=20)

    print(f"✅ {mode_name} 完成！")
    print(f"🖼️ 结果文件：{vis_output}")
    return vis_output


if __name__ == "__main__":
    # -------------------------------
    # 1️⃣ 路径与参数加载
    # -------------------------------
    left_image_path = "./StereoRectify/viewpoint1/left.png"
    right_image_path = "./StereoRectify/viewpoint1/right.png"
    json_path = "./StereoRectify/viewpoint1/params.json"
    output_dir = "./rectified"
    json_save_path="./rectified/rectification_params.json"
    os.makedirs(output_dir, exist_ok=True)

    # 加载相机参数
    params = load_camera_parameters(json_path)
    left_K = params["left_K"]
    left_dist = params["left_distortion"]
    right_K = params["right_K"]
    right_dist = params["right_distortion"]
    Rt = params["Rt"]

    R, T = Rt[:3, :3], Rt[:3, 3:4]
    R1, R2 = np.eye(3), np.eye(3)

    # -------------------------------
    # 2️⃣ 全局控制参数
    # -------------------------------
    delta_angle = np.deg2rad(12)  # 校正旋转角
    res_scale = 1.0               # 分辨率缩放
    fov_scale = 0.3             # 视场缩放

    print("\n📸 开始执行双模式校正演示...")

    # # -------------------------------
    # # 3️⃣ 模式一：统一内参校正
    # # -------------------------------
    # unified_vis = run_rectification(
    #     mode_name="统一内参视场自适应校正",
    #     rectify_func=unified_rectify_images,
    #     output_prefix="unified",
    #     left_K=left_K, left_dist=left_dist,
    #     right_K=right_K, right_dist=right_dist,
    #     R=R, T=T, R1=R1, R2=R2,
    #     left_image_path=left_image_path, right_image_path=right_image_path,
    #     output_dir=output_dir,
    #     delta_angle=delta_angle, res_scale=res_scale, fov_scale=fov_scale, json_save_path
    # )

    # -------------------------------
    # 4️⃣ 模式二：独立光心统一焦距校正
    # -------------------------------
    dual_vis = run_rectification(
        mode_name="独立光心统一焦距校正",
        rectify_func=unified_dual_rectify_images,
        output_prefix="dual",
        left_K=left_K, left_dist=left_dist,
        right_K=right_K, right_dist=right_dist,
        R=R, T=T, R1=R1, R2=R2,
        left_image_path=left_image_path, right_image_path=right_image_path,
        output_dir=output_dir,
        delta_angle=delta_angle, res_scale=res_scale, fov_scale=fov_scale,json_save_path=json_save_path
    )

    # -------------------------------
    # 5️⃣ 最终结果总结
    # -------------------------------
    print("\n" + "=" * 40)
    print("🎯 校正任务完成汇总")
    print("=" * 40)
    # print(f"🟢 统一内参结果: {unified_vis}")
    print(f"🔵 独立光心结果: {dual_vis}")
    print(f"\n📁 所有输出已保存至目录: {output_dir}")
