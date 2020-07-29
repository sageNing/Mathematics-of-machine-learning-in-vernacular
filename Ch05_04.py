import numpy as np
import matplotlib.pyplot as plt

# 读取训练数据
train = np.loadtxt('click.csv', delimiter=',', skiprows=1)
train_x = train[:, 0]
train_y = train[:, 1]

# 参数初始化
theta0 = np.random.rand()
theta1 = np.random.rand()

# 预测函数
def f(x):
    return theta0 + theta1 * x

# 目标函数
def E(x, y):
    return 0.5 * np.sum((y -f(x)) ** 2)

print(train_x.mean())

# 标准化
mu = train_x.mean()
sigma = train_x.std()
def standardize(x):
    return (x - mu) / sigma

train_z = standardize(train_x)

# 学习率
ETA = 1e-3

# 误差的差值
diff = 1

# 更新的次数
count = 0

# 重复学习
error = E(train_z, train_y)
while diff > 1e-2:
    tmp0 = theta0 - ETA * np.sum((f(train_z) - train_y))
    tmp1 = theta1 - ETA * np.sum((f(train_z) - train_y) * train_z)
    # 更新值
    theta0 = tmp0
    theta1 = tmp1
    # 计算与上一次误差的差值
    current_error = E(train_z, train_y)
    diff = error - current_error
    error = current_error
    # 输出日志
    count += 1
    log = '第{}次：theta0 = {:.3f}, theta1 = {:.3f}, 差值={:.4f}'
    print(log.format(count, theta0, theta1, diff))