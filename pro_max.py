# 解决中文乱码+基础配置
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import fcluster, linkage

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# 全局配置
device = torch.device("cpu")
HISTORY_STEPS = 10
FUTURE_STEPS = 5
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 0.001
DATA_PATH = r"D:\车辆工程-刘永权\数据集\forecasting_sample\data"

# ===================== 1. 核心损失函数（微调权重） =====================
huber_loss = nn.HuberLoss(delta=1.0)


class DisplacementErrorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target, loss_type="ADE"):
        displacement = torch.norm(pred - target, dim=-1)
        return displacement[:, -1].mean() if loss_type == "FDE" else displacement.mean()


displacement_loss = DisplacementErrorLoss()


class HeadingLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        if pred.shape[1] < 2 or target.shape[1] < 2:
            return torch.tensor(0.0, device=pred.device)
        pred_dx = pred[:, 1:, 0] - pred[:, :-1, 0]
        pred_dy = pred[:, 1:, 1] - pred[:, :-1, 1]
        target_dx = target[:, 1:, 0] - target[:, :-1, 0]
        target_dy = target[:, 1:, 1] - target[:, :-1, 1]
        pred_heading = torch.atan2(pred_dy, pred_dx)
        target_heading = torch.atan2(target_dy, target_dx)
        heading_diff = torch.abs(pred_heading - target_heading)
        heading_diff = torch.where(heading_diff > np.pi, 2 * np.pi - heading_diff, heading_diff)
        return heading_diff.mean()


heading_loss = HeadingLoss()


# ===================== 2. 关键修复：提取主轨迹（解决震荡） =====================
def extract_main_trajectory(trajectory, threshold=5.0):
    """通过层次聚类提取连续的主轨迹，过滤杂点"""
    if len(trajectory) < 5:
        return trajectory
    # 聚类：按坐标距离分组
    Z = linkage(trajectory, method='single', metric='euclidean')
    clusters = fcluster(Z, threshold, criterion='distance')
    # 取最大的聚类作为主轨迹
    unique_clusters, counts = np.unique(clusters, return_counts=True)
    main_cluster = unique_clusters[np.argmax(counts)]
    main_traj = trajectory[clusters == main_cluster]
    # 按时间排序（恢复轨迹连续性）
    return main_traj


def smooth_trajectory(trajectory, window=3):
    if len(trajectory) < window:
        return trajectory
    smoothed = np.copy(trajectory).astype(np.float32)
    for i in range(1, len(trajectory) - 1):
        smoothed[i] = np.mean(trajectory[i - 1:i + 2], axis=0)
    return smoothed


def normalize_trajectory(trajectory):
    if len(trajectory) == 0:
        return trajectory, np.array([0, 0])
    # 第一步：提取主轨迹（核心修复）
    main_traj = extract_main_trajectory(trajectory)
    # 第二步：平滑
    smoothed_traj = smooth_trajectory(main_traj)
    # 第三步：归一化
    start_point = smoothed_traj[0].copy()
    normalized_traj = smoothed_traj - start_point
    return normalized_traj, start_point


# ===================== 3. 特征函数（保留） =====================
def get_lane_features(trajectory):
    lane_features = []
    for i, (x, y) in enumerate(trajectory):
        if i == 0:
            lane_dist, lane_dir, angle_diff = 0.0, 0.0, 0.0
        else:
            center_x = np.mean(trajectory[:i + 1, 0])
            center_y = np.mean(trajectory[:i + 1, 1])
            lane_dist = np.linalg.norm([x - center_x, y - center_y])
            lane_dir = np.arctan2(y - trajectory[i - 1, 1], x - trajectory[i - 1, 0])
            angle_diff = np.abs(lane_dir - np.arctan2(trajectory[i - 1, 1] - trajectory[i - 2, 1],
                                                      trajectory[i - 1, 0] - trajectory[i - 2, 0])) if i > 1 else 0.0
        lane_features.append([lane_dist, lane_dir, angle_diff])
    return np.array(lane_features, dtype=np.float32)


def get_kinematic_features(trajectory):
    kinematic_feats = []
    for i, (x, y) in enumerate(trajectory):
        if i == 0:
            speed, acc, heading = 0.0, 0.0, 0.0
        elif i == 1:
            dx, dy = x - trajectory[i - 1, 0], y - trajectory[i - 1, 1]
            speed, heading = np.linalg.norm([dx, dy]), np.arctan2(dy, dx)
            acc = 0.0
        else:
            dx1, dy1 = x - trajectory[i - 1, 0], y - trajectory[i - 1, 1]
            dx0, dy0 = trajectory[i - 1, 0] - trajectory[i - 2, 0], trajectory[i - 1, 1] - trajectory[i - 2, 1]
            speed1, speed0 = np.linalg.norm([dx1, dy1]), np.linalg.norm([dx0, dy0])
            speed, acc, heading = speed1, speed1 - speed0, np.arctan2(dy1, dx1)
        kinematic_feats.append([speed, acc, heading])
    return np.array(kinematic_feats, dtype=np.float32)


# ===================== 4. 数据集类（修复后） =====================
class ArgoverseTrajectoryDataset(Dataset):
    def __init__(self, data_path, history_steps=10, future_steps=5):
        self.history_steps = history_steps
        self.future_steps = future_steps
        self.trajectories, self.start_points, self.lane_feats, self.kinematic_feats = self.load_trajectories(data_path)
        self.samples = self.create_samples()

    def load_trajectories(self, data_path):
        trajectories, start_points, lane_feats, kinematic_feats = [], [], [], []
        print(f"🔍 正在加载数据集：{data_path}")
        file_list = [f for f in os.listdir(data_path) if f.endswith(".csv")]
        print(f"📂 找到 {len(file_list)} 个csv文件：{file_list}")
        if len(file_list) == 0:
            print("❌ 路径下没有csv文件！")
            return [], [], [], []

        for file_name in file_list:
            file_path = os.path.join(data_path, file_name)
            df = pd.read_csv(file_path)
            if "X" not in df.columns or "Y" not in df.columns:
                print(f"⚠️ 跳过 {file_name}：缺少X/Y列")
                continue
            traj = df[["X", "Y"]].dropna().values
            print(f"📄 {file_name}：原始轨迹长度={len(traj)}")
            if len(traj) < self.history_steps + self.future_steps:
                print(f"⚠️ 跳过 {file_name}：轨迹长度不足")
                continue
            traj_norm, start_point = normalize_trajectory(traj)
            print(f"📄 {file_name}：主轨迹长度={len(traj_norm)}")
            if len(traj_norm) < self.history_steps + self.future_steps:
                continue
            lane_feat = get_lane_features(traj_norm)
            kinematic_feat = get_kinematic_features(traj_norm)
            trajectories.append(traj_norm)
            start_points.append(start_point)
            lane_feats.append(lane_feat)
            kinematic_feats.append(kinematic_feat)

        print(f"\n✅ 加载完成：{len(trajectories)} 条有效轨迹")
        return trajectories, start_points, lane_feats, kinematic_feats

    def create_samples(self):
        samples = []
        for i, traj in enumerate(self.trajectories):
            lane_feat, kinematic_feat = self.lane_feats[i], self.kinematic_feats[i]
            max_j = len(traj) - self.history_steps - self.future_steps
            if max_j <= 0:
                continue
            for j in range(max_j):
                history_traj = traj[j:j + self.history_steps]
                history_lane = lane_feat[j:j + self.history_steps]
                history_kinematic = kinematic_feat[j:j + self.history_steps]
                future_traj = traj[j + self.history_steps:j + self.history_steps + self.future_steps]
                history_input = np.concatenate([history_traj, history_lane, history_kinematic], axis=1)
                samples.append((history_input, future_traj))
        print(f"✅ 生成 {len(samples)} 个训练样本（修复后）")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        h, f = self.samples[idx]
        return torch.FloatTensor(h).to(device), torch.FloatTensor(f).to(device)


# ===================== 5. 模型类（修复解码器初始化） =====================
class TrajectoryLSTM(nn.Module):
    def __init__(self, traj_dim=2, map_dim=3, kinematic_dim=3, hidden_dim=128, num_layers=2, output_dim=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.future_steps = FUTURE_STEPS

        self.feat_fusion = nn.Sequential(
            nn.Linear(traj_dim + map_dim + kinematic_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)  # 降低dropout，提升拟合能力
        )

        self.encoder_lstm = nn.LSTM(
            hidden_dim, hidden_dim, num_layers, batch_first=True, dropout=0.1
        )

        self.trend_extractor = nn.Sequential(
            nn.Linear(hidden_dim + traj_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.decoder_lstm = nn.LSTM(
            traj_dim, hidden_dim, num_layers, batch_first=True, dropout=0.1
        )

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        batch_size = x.shape[0]
        traj_feat = x[:, :, :2]
        fusion_feat = self.feat_fusion(x)

        # 编码
        encoder_out, (enc_h, enc_c) = self.encoder_lstm(fusion_feat)
        last_step_feat = encoder_out[:, -1, :]
        last_step_coord = traj_feat[:, -1, :]
        trend_feat = self.trend_extractor(torch.cat([last_step_feat, last_step_coord], dim=1))

        # 修复解码器初始化：用编码器的隐藏状态，而非全零
        decoder_hidden = trend_feat.unsqueeze(0).repeat(self.num_layers, 1, 1)
        decoder_cell = enc_c  # 复用编码器的细胞状态，保留更多特征

        # 解码
        pred_trajectory = []
        decoder_input = traj_feat[:, -1:, :]
        for _ in range(self.future_steps):
            decoder_out, (decoder_hidden, decoder_cell) = self.decoder_lstm(decoder_input,
                                                                            (decoder_hidden, decoder_cell))
            pred_step = self.fc(decoder_out)
            pred_trajectory.append(pred_step)
            decoder_input = pred_step

        return torch.cat(pred_trajectory, dim=1)


# ===================== 6. 训练函数（微调损失权重） =====================
def train_model():
    dataset = ArgoverseTrajectoryDataset(DATA_PATH, HISTORY_STEPS, FUTURE_STEPS)
    if len(dataset) == 0:
        print("❌ 没有有效训练样本！")
        return None, dataset
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    model = TrajectoryLSTM().to(device)

    # 微调损失权重：降低方向损失，提升坐标拟合
    def combined_loss(pred, target):
        return 0.5 * huber_loss(pred, target) + 0.4 * displacement_loss(pred, target, "ADE") + 0.1 * displacement_loss(
            pred, target, "FDE") + 0.05 * heading_loss(pred, target)

    criterion = combined_loss

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)  # 加权重衰减防过拟合
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)

    model.train()
    print("\n🚀 开始训练（修复主轨迹+微调模型）...")
    for epoch in range(EPOCHS):
        total_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS}")
        for history, future in pbar:
            pred = model(history)
            loss = criterion(pred, future)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪防爆炸
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({"Loss": loss.item(), "LR": optimizer.param_groups[0]['lr']})
        scheduler.step()
        avg_loss = total_loss / len(dataloader)
        print(f"📌 Epoch {epoch + 1} 平均损失: {avg_loss:.6f}")

    torch.save(model.state_dict(), "trajectory_lstm_final.pth")
    print("\n✅ 模型训练完成，已保存为 trajectory_lstm_final.pth")
    return model, dataset


# ===================== 7. 可视化函数（保留调试） =====================
def predict_and_visualize(model, dataset):
    if model is None:
        return
    model.eval()
    sample_idx = 0
    history_input, future = dataset[sample_idx]
    start_point = dataset.start_points[0]
    history_traj = history_input[:, :2]

    # 调试打印
    print("\n🔍 最终调试信息")
    print("=" * 50)
    print(f"历史轨迹坐标:\n{history_traj.cpu().numpy()}")
    print(f"反归一化后历史轨迹:\n{history_traj.cpu().numpy() + start_point}")
    print("=" * 50)

    # 预测
    with torch.no_grad():
        pred = model(history_input.unsqueeze(0))

    # 反归一化
    history_original = history_traj.cpu().numpy() + start_point
    future_original = future.cpu().numpy() + start_point
    pred_original = pred.squeeze(0).cpu().numpy() + start_point

    # 绘图
    plt.figure(figsize=(12, 8))
    plt.plot(history_original[:, 0], history_original[:, 1], "b-", linewidth=2.5, label="历史轨迹（主轨迹）")
    plt.scatter(history_original[0, 0], history_original[0, 1], color="blue", s=60, label="历史起点")
    plt.scatter(history_original[-1, 0], history_original[-1, 1], color="darkblue", s=60, label="历史终点")
    future_full = np.concatenate([history_original[-1:], future_original], axis=0)
    plt.plot(future_full[:, 0], future_full[:, 1], "g-", linewidth=2.5, label="真实未来轨迹")
    plt.scatter(future_original[-1, 0], future_original[-1, 1], color="darkgreen", s=60, label="真实终点")
    pred_full = np.concatenate([history_original[-1:], pred_original], axis=0)
    plt.plot(pred_full[:, 0], pred_full[:, 1], "r--", linewidth=2.5, label="预测未来轨迹（最终版）")
    plt.scatter(pred_original[-1, 0], pred_original[-1, 1], color="darkred", s=60, label="预测终点")

    plt.xlabel("X 坐标 (m)")
    plt.ylabel("Y 坐标 (m)")
    plt.title("车辆轨迹预测结果（主轨迹提取+模型微调）")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis("equal")
    plt.tight_layout()
    plt.show()

    print("\n===== 最终轨迹数值 =====")
    print("历史轨迹：\n", history_original)
    print("真实未来轨迹：\n", future_original)
    print("预测未来轨迹：\n", pred_original)


# 执行主流程
if __name__ == "__main__":
    model, dataset = train_model()
    predict_and_visualize(model, dataset)