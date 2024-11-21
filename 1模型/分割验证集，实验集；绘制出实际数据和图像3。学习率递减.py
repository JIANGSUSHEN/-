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


def split_data(X, y, test_ratio=0.2):
    """将数据分割为训练集和验证集"""
    m = len(y)
    indices = np.arange(m)
    np.random.shuffle(indices)
    split_idx = int(m * (1 - test_ratio))
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    return X[train_indices], y[train_indices], X[val_indices], y[val_indices]


def compute_cost(X, y, theta, reg_lambda=0.0):
    """计算损失函数（带 L2 正则化）"""
    m = len(y)
    predictions = X @ theta
    errors = predictions - y
    cost = (1 / (2 * m)) * np.sum(errors**2)
    reg_term = (reg_lambda / (2 * m)) * np.sum(theta[1:]**2)  # 不对偏置项正则化
    return cost + reg_term


def gradient_descent(X, y, theta, alpha, epochs, reg_lambda=0.0, decay_rate=0.01):
    """梯度下降优化，动态调整学习率"""
    m = len(y)
    cost_history = []
    for epoch in range(epochs):
        predictions = X @ theta
        errors = predictions - y
        gradients = (1 / m) * (X.T @ errors)
        gradients[1:] += (reg_lambda / m) * theta[1:]  # 正则化项
        theta -= alpha * gradients
        cost = compute_cost(X, y, theta, reg_lambda)
        cost_history.append(cost)

        # 动态调整学习率
        alpha *= (1 / (1 + decay_rate * epoch))

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Cost = {cost:.4f}, Learning rate = {alpha:.4f}")
    return theta, cost_history


def plot_results(y_actual, y_pred, output_dir):
    """绘制实际数据和预测数据的对比曲线"""
    plt.figure()
    plt.plot(y_actual, label="Actual")
    plt.plot(y_pred, label="Predicted")
    plt.title("Actual vs Predicted")
    plt.xlabel("Sample Index")
    plt.ylabel("Value")
    plt.legend()
    output_path = os.path.join(output_dir, "actual_vs_predicted.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Actual vs Predicted plot saved to: {output_path}")


def save_results(theta, cost_history, output_dir):
    """保存模型结果和损失曲线"""
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
    print(f"Loss curve plot saved to: {loss_curve_file}")


if __name__ == "__main__":
    # 文件路径由用户定义
    csv_file_path = r"D:\BaiduNetdiskDownload\Harvard university 人工智能\多元线性下降模型（销量预测）\1模型\advertising (1).csv"
    output_dir = r"D:\BaiduNetdiskDownload\Harvard university 人工智能\多元线性下降模型（销量预测）\模型训练结果\模型2"

    # 1. 从 CSV 文件读取数据
    X, y = load_csv(csv_file_path)

    # 2. 数据归一化
    X_normalized, mean, std = normalize_features(X)

    # 3. 数据集分割
    X_train, y_train, X_val, y_val = split_data(X_normalized, y)

    # 4. 添加偏置项
    X_train_bias = np.c_[np.ones(X_train.shape[0]), X_train]
    X_val_bias = np.c_[np.ones(X_val.shape[0]), X_val]

    # 5. 初始化参数
    theta = np.zeros(X_train_bias.shape[1])
    alpha = 0.1  # 初始学习率
    epochs = 200  # 训练轮数
    reg_lambda = 0.1  # 正则化强度
    decay_rate = 0.001  # 学习率衰减率

    # 6. 模型训练
    theta, train_loss = gradient_descent(X_train_bias, y_train, theta, alpha, epochs, reg_lambda, decay_rate)

    # 7. 验证集预测
    y_val_pred = X_val_bias @ theta

    # 8. 保存结果
    save_results(theta, train_loss, output_dir)

    # 9. 绘制实际与预测对比图
    plot_results(y_val, y_val_pred, output_dir)
