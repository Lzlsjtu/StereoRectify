import cv2
import json
import numpy as np
import matplotlib.pyplot as plt

def load_camera_parameters(json_path):
    """从 JSON 文件读取相机标定参数并转换为 numpy 格式"""
    # 打开并读取 JSON 文件中的相机标定参数
    with open(json_path, 'r') as f:
        data = json.load(f)  # 使用 json.load() 读取 JSON 文件数据
    # 将读取到的相机参数转换为 NumPy 数组并返回
    return {
        'left_K': np.array(data['left_K']),  # 左相机的内参矩阵
        'left_distortion': np.array(data['left_distortion']),  # 左相机的畸变系数
        'right_K': np.array(data['right_K']),  # 右相机的内参矩阵
        'right_distortion': np.array(data['right_distortion']),  # 右相机的畸变系数
        'Rt': np.array(data['Rt'])  # 相机间的旋转矩阵和平移向量
    }


def visualize_rectification(left_img, right_img, output_path="rectified_comparison.png", num_lines=15):
    """拼接左右校正图并绘制水平线验证极线对齐"""
    # 检查左右图像的尺寸是否一致
    if left_img.shape != right_img.shape:
        raise ValueError("左右图像尺寸不一致，无法拼接")

    # 将左右图像拼接在一起，形成一个横向对比图
    combined = np.hstack((left_img, right_img))  # 水平拼接左右图像
    h, w, _ = combined.shape  # 获取拼接后的图像的高度和宽度
    step = h // num_lines  # 计算绘制水平线的间隔

    # 在拼接后的图像上绘制水平线，以验证极线对齐
    for y in range(0, h, step):
        cv2.line(combined, (0, y), (w, y), (0, 255, 0), 1)  # 绘制绿色水平线

    # 将拼接后的图像保存到指定路径
    cv2.imwrite(output_path, combined)
    print(f"✅ 拼接结果已保存: {output_path}")

    # 使用 matplotlib 显示拼接图像
    plt.figure(figsize=(12, 6))  # 设置显示窗口的大小
    plt.imshow(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))  # 转换为 RGB 格式显示
    plt.axis('off')  # 关闭坐标轴
    plt.title("Rectification Check (Green lines should be aligned)")  # 设置标题
    plt.show()  # 显示图像
    return combined  # 返回拼接后的图像
