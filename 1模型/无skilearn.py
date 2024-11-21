import numpy as np
import matplotlib.pyplot as plt

# 生成数据
def generate_data():
    np.random.seed(42)
    # 人工生成特征变量 X 和目标变量 y
    m = 100  # 样本数
    X = np.random.rand(m, 3)  # 三个特征，每个特征值范围 [0, 1)
    true_theta = np.array([4, 3, 2, 1])  # 假设真实参数为 [4, 3, 2, 1]
    y = true_theta[0] + np.dot(X, true_theta[1:]) + np.random.randn(m) * 0.5  # 添加噪声?????????????????????????????????
    return X, y

# 数据归一化
def normalize_features(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_normalized = (X - mean) / std
    return X_normalized, mean, std

# 损失函数
def compute_cost(X, y, theta):
    m = len(y)
    predictions = X @ theta  # 矩阵乘法计算预测值??????????????????????????????????????????????????????????????????????????
    errors = predictions - y
    cost = (1 / (2 * m)) * np.sum(errors**2)
    return cost

# 梯度下降
def gradient_descent(X, y, theta, alpha, epochs):
    m = len(y)
    cost_history = []  # 保存每轮的损失值
    for epoch in range(epochs):
        predictions = X @ theta  # 计算预测值
        errors = predictions - y
        gradients = (1 / m) * (X.T @ errors)  # 梯度计算
        theta -= alpha * gradients  # 更新参数
        cost = compute_cost(X, y, theta)  # 计算损失
        cost_history.append(cost)
        if epoch % 100 == 0:  # 每 100 轮打印一次损失
            print(f"Epoch {epoch}: Cost = {cost}")
    return theta, cost_history

# 主函数
if __name__ == "__main__":#？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？
    # 1. 生成数据
    X, y = generate_data()

    # 2. 数据归一化
    X_normalized, mean, std = normalize_features(X)

    # 3. 添加偏置项（全为 1 的列）意义？
    X_with_bias = np.c_[np.ones(X_normalized.shape[0]), X_normalized]#????????????????????????????????????????????????????

    # 4. 初始化参数
    theta = np.zeros(X_with_bias.shape[1])  # 初始化参数为 0
    alpha = 0.1  # 学习率
    epochs = 1000  # 训练轮数

    # 5. 梯度下降
    theta, cost_history = gradient_descent(X_with_bias, y, theta, alpha, epochs)

    # 6. 输出结果
    print("Final parameters (theta):", theta)

    # 7. 绘制损失曲线
    plt.plot(cost_history)
    plt.title("Loss Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Cost")
    plt.show()
