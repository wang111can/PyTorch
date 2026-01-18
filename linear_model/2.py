# 第一步：先设置Agg后端，解决Linux无图形界面的渲染问题（Conda环境必备）
import matplotlib
matplotlib.use('Agg')

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

# 目标模型：y = 2x + 2，对应的样本数据
x_data = [1, 2, 3]
y_data = [4, 6, 8]

# 前向传播函数：实现线性模型 y = w*x + b
def forward(x, w, b):
    return x * w + b

# 损失函数：计算单个样本的均方误差（MSE）
def loss(x, y, w, b):
    y_pred = forward(x, w, b)
    return (y_pred - y) ** 2

# 修正变量名拼写错误（mse_lilst → mse_list，虽未使用但规范代码）
mse_list = []
# 生成权重w的取值范围：0.0 ~ 4.0，步长0.1
W = np.arange(0.0, 4.0, 0.1)
# 生成偏置b的取值范围：0.0 ~ 4.0，步长0.1
B = np.arange(0.0, 4.0, 0.1)
# 生成w和b的网格矩阵（核心：将一维数组转为二维网格，适配3D绘图）
[W, B] = np.meshgrid(W, B)

# 累加所有样本的损失值
loss_sum = 0
for x_val, y_val in zip(x_data, y_data):
    y_pred_val = forward(x_val, W, B)  # 计算预测值（网格矩阵运算，批量计算所有w/b组合）
    loss_val = loss(x_val, y_val, W, B)  # 计算单个样本的损失（矩阵形式）
    loss_sum += loss_val  # 累加所有样本的损失

# 创建画布和3D坐标轴（新版matplotlib推荐写法，避免警告）
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')  # 替代直接Axes3D(fig)
# 绘制3D损失曲面（损失值取平均，得到MSE）
ax.plot_surface(W, B, loss_sum / 3, cmap='viridis')  # 添加cmap让曲面有颜色渐变，更直观
# 设置坐标轴标签
ax.set_xlabel('Weight (w)')
ax.set_ylabel('Bias (b)')
ax.set_zlabel('MSE Loss')
# 保存图片（Conda环境下无报错）
plt.savefig('conda_matplotlib_test.png')
print("3D损失曲面图已保存为 conda_matplotlib_test.png")