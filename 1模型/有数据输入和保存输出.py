import os
import numpy as np
import matplotlib.pyplot as plt
import csv


def load_csv(file_path):
    """从本地 CSV 文件读取数据"""
    with open(file_path, "r") as file:
        reader = csv.reader(file)
        data = list(reader)
    data = np.array(data[1:], dtype=float)  # 跳过标题行并转换为浮点数
    X = data[:, :-1]  # 前几列为特征
    y = data[:, -1]   # 最后一列为目标
    return X, y


def normalize_features(X):
    """归一化数据"""
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_normalized = (X - mean) / std
    return X_normalized, mean, std


def compute_cost(X, y, theta):
    """计算损失函数"""
    m = len(y)
    predictions = X @ theta
    errors = predictions - y
    cost = (1 / (2 * m)) * np.sum(errors**2)
    return cost


def gradient_descent(X, y, theta, alpha, epochs):
    """梯度下降优化"""
    m = len(y)
    cost_history = []
    for epoch in range(epochs):
        predictions = X @ theta
        errors = predictions - y
        gradients = (1 / m) * (X.T @ errors)
        theta -= alpha * gradients
        cost = compute_cost(X, y, theta)
        cost_history.append(cost)
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Cost = {cost}")
    return theta, cost_history


def save_results(theta, cost_history, output_dir):
    """保存模型结果和图片"""
    os.makedirs(output_dir, exist_ok=True)

    # 保存参数
    theta_file = os.path.join(output_dir, "model_parameters.txt")
    with open(theta_file, "w") as f:
        f.write("Learned Parameters (Theta):\n")
        for i, value in enumerate(theta):
            f.write(f"Theta[{i}]: {value:.4f}\n")

    # 保存损失曲线
    loss_curve_file = os.path.join(output_dir, "loss_curve.png")
    plt.figure()
    plt.plot(cost_history)
    plt.title("Loss Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Cost")
    plt.savefig(loss_curve_file)
    plt.close()
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    # 文件路径由用户定义
    csv_file_path = r"D:\BaiduNetdiskDownload\Harvard university 人工智能\多元线性下降模型（销量预测）\1模型\advertising (1).csv"
    output_dir = r"D:\BaiduNetdiskDownload\Harvard university 人工智能\多元线性下降模型（销量预测）\模型训练结果\模型1"

    # 1. 从 CSV 文件读取数据
    X, y = load_csv(csv_file_path)

    # 2. 数据归一化
    X_normalized, mean, std = normalize_features(X)

    # 3. 添加偏置项
    X_with_bias = np.c_[np.ones(X_normalized.shape[0]), X_normalized]

    # 4. 初始化参数
    theta = np.zeros(X_with_bias.shape[1])
    alpha = 0.1  # 学习率
    epochs = 1000  # 训练轮数

    # 5. 梯度下降
    theta, cost_history = gradient_descent(X_with_bias, y, theta, alpha, epochs)

    # 6. 保存结果
    save_results(theta, cost_history, output_dir)
