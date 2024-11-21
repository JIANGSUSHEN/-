import matplotlib.pyplot as plt
import csv
import os
import json
import numpy as np



def save_training_checkpoint(output_dir, epoch, theta, cost, learning_rate):
    """保存每轮训练的存根信息到文件"""
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_file = os.path.join(output_dir, "training_checkpoints.json")

    # 如果文件不存在，则初始化存根文件
    if not os.path.exists(checkpoint_file):
        with open(checkpoint_file, "w") as f:
            json.dump({"epochs": []}, f)

    # 加载现有存根信息
    with open(checkpoint_file, "r") as f:
        data = json.load(f)

    # 添加新的存根信息
    data["epochs"].append({
        "epoch": epoch,
        "theta": theta.tolist(),
        "cost": cost,
        "learning_rate": learning_rate
    })

    # 保存到文件
    with open(checkpoint_file, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Checkpoint saved for epoch {epoch}: cost = {cost:.4f}, learning_rate = {learning_rate:.6f}")


def gradient_descent(X, y, theta, alpha, epochs, reg_lambda=0.0, decay_rate=0.01, output_dir=None):
    """梯度下降优化，动态调整学习率，支持存根保存"""
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

        # 保存训练存根
        if output_dir:
            save_training_checkpoint(output_dir, epoch, theta, cost, alpha)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Cost = {cost:.4f}, Learning rate = {alpha:.6f}")

    return theta, cost_history


if __name__ == "__main__":
    # 文件路径由用户定义
    csv_file_path = r"D:\BaiduNetdiskDownload\Harvard university 人工智能\多元线性下降模型（销量预测）\1模型\advertising (1).csv"
    output_dir = r"D:\BaiduNetdiskDownload\Harvard university 人工智能\多元线性下降模型（销量预测）\模型训练结果\模型4以及模型四每一轮训练存根"

    # 1. 从 CSV 文件读取数据
    X, y, header = load_csv(csv_file_path)

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
    epochs = 1000  # 训练轮数
    reg_lambda = 0.1  # 正则化强度
    decay_rate = 0.01  # 学习率衰减率

    # 6. 模型训练，保存存根信息
    theta, train_loss = gradient_descent(
        X_train_bias, y_train, theta, alpha, epochs, reg_lambda, decay_rate, output_dir
    )

    # 7. 验证集预测
    y_val_pred = X_val_bias @ theta

    # 8. 保存结果
    save_results(theta, train_loss, output_dir, header)

    # 9. 绘制实际与预测对比图
    plot_results(y_val, y_val_pred, output_dir)
