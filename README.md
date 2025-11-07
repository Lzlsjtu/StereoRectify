# StereoRectify —— 基于完整视场保留的双目立体校正

---

## 🚀 项目简介

这是一个简单但实用的双目立体校正算法项目，主要目标是：
> **在保证极线对齐的同时，尽可能保留完整的原始视场。**

传统的 OpenCV `stereoRectify` 在校正后常常会裁剪掉图像边缘，导致有效像素损失。  
本项目通过对角点投影范围的计算与统一内参矩阵的自适应生成，有效解决了这一问题。

项目地址 👉 [StereoRectify on GitHub](https://github.com/Lzlsjtu/StereoRectify/tree/main)

---

## 🧠 核心思路

- 基于几何关系计算左右相机的旋转矩阵，使得校正后光轴平行；
- 通过角点逆投影与旋转，确定完整视场下的新图像边界；
- 生成统一的新内参矩阵 $K^{new}$，避免视场裁剪；
- 引入缩放因子 $\alpha$ 调整输出比例；
- 可选自适应主点优化，减少黑边区域。

详细数学推导与原理说明请参考 👉 📄 **[基于完整视场保留的双目立体校正算法.md](./基于完整视场保留的双目立体校正算法.md)**

---

## ⚙️ 使用方法

### 1️⃣ 安装依赖
```bash
pip install -r requirements.txt
````

### 2️⃣ 运行示例

```bash
python rectification.py \
  --left data/left.png \
  --right data/right.png \
  --output results/
```

运行后将在 `results/` 文件夹下生成校正后的左右图像。

---

## 📦 主要文件结构

```
StereoRectify/
├── rectification.py        # 校正主流程
├── intrinsics_update.py    # 新内参生成
├── utils.py                # 几何计算辅助函数
├── data/                   # 示例输入图像
├── results/                # 校正输出结果
└── 基于完整视场保留的双目立体校正算法.md
```

---

## 💬 项目说明

这个项目更偏向于**教学与研究辅助**，重点放在：

* 理解双目校正的几何原理；
* 掌握完整视场保留的处理逻辑；
* 为后续立体匹配与深度估计提供标准化输入。

---

## 🧩 更多内容

如果你想了解：

* 校正几何的数学推导；
* 内参变化的物理意义；
* 缩放系数 $\alpha$ 对视场的影响；

可以阅读 👉📘 **[基于完整视场保留的双目立体校正算法.md](./基于完整视场保留的双目立体校正算法.md)**

---

## 🤝 贡献与交流

欢迎提出问题或提交改进建议：

* 提交 Issue 或 Pull Request；
* 直接在 [GitHub Discussions](https://github.com/Lzlsjtu/StereoRectify/discussions) 参与讨论；
* 一起完善这份开源小工具！

---

## 📜 License

本项目基于 **MIT License** 开源，欢迎自由使用与修改。
