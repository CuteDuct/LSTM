# 基于LSTM的车辆轨迹预测

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-orange)](https://pytorch.org/)

> 车辆工程 × 深度学习 | 自动驾驶轨迹预测实践项目

---

## 📌 项目简介

本项目基于 PyTorch 构建 LSTM 编码器-解码器网络，实现车辆未来轨迹预测。针对原始轨迹数据中的噪声和震荡问题，提出**基于层次聚类的主轨迹提取方法**和**改进的解码器初始化策略**。

**关键词**：自动驾驶、轨迹预测、LSTM、多特征融合、PyTorch

---

## 🚗 项目背景

本人为车辆工程专业学生，辅修计算机，本项目为自动驾驶方向的个人实践项目。旨在探索深度学习在车辆运动预测中的应用，为未来研究生阶段从事自动驾驶决策规划方向打基础。

---

## 🛠️ 技术栈

| 工具 | 用途 |
|------|------|
| Python 3.8+ | 主编程语言 |
| PyTorch 1.9+ | 深度学习框架 |
| NumPy / Pandas | 数据处理 |
| Matplotlib | 可视化 |
| Scikit-learn | 层次聚类 |

---

## 📂 项目结构

```
LSTM/
├── pro_max.py          # 主程序：训练 + 预测 + 可视化
├── requirements.txt    # 依赖列表
├── README.md           # 项目说明
└── data/               # 数据集（需自行下载）用的argoverse中的一小部分哦，我的电脑配置只有cpu
    └── forecasting_sample/
```

---

## 🔧 运行方式

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备数据集

将 Argoverse 轨迹数据集放入 `./data/` 目录下。

### 3. 开始训练

```bash
python pro_max.py --data_path ./data
```

---

## 🧠 核心方法

### 1. 主轨迹提取
利用层次聚类过滤传感器数据中的离群点，提取连续稳定的主轨迹。

### 2. 多特征融合
将轨迹坐标、车道特征（距离/方向/角度差）、运动学特征（速度/加速度/航向角）拼接输入 LSTM。

### 3. 复合损失函数
加权组合 Huber Loss、ADE（平均位移误差）、FDE（最终位移误差）和 Heading Loss。

### 4. 解码器初始化优化
用编码器末步趋势特征初始化 LSTM 解码器，替代全零初始化，提升预测稳定性。

---

## 📊 结果展示

![预测结果](amazon_close_price_trend.png)


*（上图为预测轨迹与真实轨迹对比，蓝色为历史轨迹，绿色为真实未来轨迹，红色虚线为预测轨迹）*

---

## 📖 参考资料

- Argoverse 轨迹数据集
- ECCV 2024 UniTraj 框架
- PyTorch 官方文档

---

## 📧 联系我

- 邮箱：2016291992@qq.com
- GitHub：https://github.com/CuteDuct
- 学校：长安大学 车辆工程专业

---

## 📝 待优化方向

- [ ] 引入注意力机制
- [ ] 支持多模态轨迹预测
- [ ] 添加 TensorBoard 训练日志
- [ ] 迁移到更多数据集测试

---

**⭐ 如果对你有帮助，欢迎 Star！**
