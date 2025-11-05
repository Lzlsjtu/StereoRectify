import cv2
import json
import numpy as np
import matplotlib.pyplot as plt


def load_camera_parameters(json_path):
    """从 JSON 文件读取相机标定参数并转换为 numpy 格式"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return {
        'left_K': np.array(data['left_K']),
        'left_distortion': np.array(data['left_distortion']),
        'right_K': np.array(data['right_K']),
        'right_distortion': np.array(data['right_distortion']),
        'Rt': np.array(data['Rt'])
    }


def visualize_rectification(left_img, right_img, output_path="rectified_comparison.png", num_lines=15):
    """拼接左右校正图并绘制水平线验证极线对齐"""
    if left_img.shape != right_img.shape:
        raise ValueError("左右图像尺寸不一致，无法拼接")

    combined = np.hstack((left_img, right_img))
    h, w, _ = combined.shape
    step = h // num_lines

    for y in range(0, h, step):
        cv2.line(combined, (0, y), (w, y), (0, 255, 0), 1)

    cv2.imwrite(output_path, combined)
    print(f"✅ 拼接结果已保存: {output_path}")

    plt.figure(figsize=(12, 6))
    plt.imshow(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title("Rectification Check (Green lines should be aligned)")
    plt.show()
    return combined
