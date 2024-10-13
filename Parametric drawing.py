# -*- coding:utf-8 -*-
# author:ymh
# @Time : 2024/09/19 23:08
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



##固定β，变化α
# 假设你有9个不同的Excel文件，每个对应一个网络
file_paths = [
    'C:/Users/59666/Desktop/修改WTOPSIS/Email标签/β=0.01.xlsx',
    'C:/Users/59666/Desktop/修改WTOPSIS/FaceBook标签/β=0.01.xlsx',
    'C:/Users/59666/Desktop/修改WTOPSIS/Infect标签/β=0.01.xlsx',
    'C:/Users/59666/Desktop/修改WTOPSIS/Jazz标签/β=0.01.xlsx',
    'C:/Users/59666/Desktop/修改WTOPSIS/Message标签/β=0.01.xlsx',
    'C:/Users/59666/Desktop/修改WTOPSIS/Road标签/β=0.01.xlsx',
    'C:/Users/59666/Desktop/修改WTOPSIS/supply标签/β=0.01.xlsx',
    'C:/Users/59666/Desktop/修改WTOPSIS/USAir97标签/β=0.01.xlsx',
    'C:/Users/59666/Desktop/修改WTOPSIS/wiki-vote标签/β=0.01.xlsx',
]

labels = [
    'Email',
    'FaceBook',
    'Infect',
    'Jazz',
    'Message',
    'Road',
    'supply',
    'USAir',
    'Vote'
]


# 创建一个图形
plt.figure(figsize=(9, 7))

# 通过循环加载每个文件并绘制其 α vs. τ
for i, file_path in enumerate(file_paths):
    # 加载每个 Excel 文件
    data = pd.read_excel(file_path)

    # 将 α 和 τ 转换为 NumPy 数组
    alpha = data['α'].to_numpy()
    tau = data['τ'].to_numpy()

    # 画出当前网络的数据
    plt.plot(alpha, tau, marker='o', linestyle='-', lw=2, label=labels[i])


plt.ylim(0,1)

# 设置坐标轴刻度的字体大小
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

# 添加标签和标题
plt.xlabel(r'$\alpha$', fontsize=18)
plt.ylabel(r'$\tau$', fontsize=18)
plt.title(r'$\beta$ = 0.01', fontsize=20)
# 设置图例位置到下方
plt.legend(bbox_to_anchor=(0.85, -0.15), ncol=3, fontsize=16)
plt.subplots_adjust(bottom=0.3)

plt.savefig('C:/Users/59666/Desktop/修改WTOPSIS/β=0.01.png', dpi=600)
# 显示图表
plt.show()



##固定α，变化β

file_paths = [
    'C:/Users/59666/Desktop/修改WTOPSIS/Email标签/α=0.5.xlsx',
    'C:/Users/59666/Desktop/修改WTOPSIS/FaceBook标签/α=0.5.xlsx',
    'C:/Users/59666/Desktop/修改WTOPSIS/Infect标签/α=0.5.xlsx',
    'C:/Users/59666/Desktop/修改WTOPSIS/Jazz标签/α=0.5.xlsx',
    'C:/Users/59666/Desktop/修改WTOPSIS/Message标签/α=0.5.xlsx',
    'C:/Users/59666/Desktop/修改WTOPSIS/Road标签/α=0.5.xlsx',
    'C:/Users/59666/Desktop/修改WTOPSIS/supply标签/α=0.5.xlsx',
    'C:/Users/59666/Desktop/修改WTOPSIS/USAir97标签/α=0.5.xlsx',
    'C:/Users/59666/Desktop/修改WTOPSIS/wiki-vote标签/α=0.5.xlsx',
]

labels = [
    'Email',
    'FaceBook',
    'Infect',
    'Jazz',
    'Message',
    'Road',
    'supply',
    'USAir',
    'Vote'
]


# 创建一个图形
plt.figure(figsize=(9, 7))

# 通过循环加载每个文件并绘制其 α vs. τ
for i, file_path in enumerate(file_paths):
    # 加载每个 Excel 文件
    data = pd.read_excel(file_path)

    # 将 α 和 τ 转换为 NumPy 数组
    alpha = data['β'].to_numpy()
    tau = data['τ'].to_numpy()

    # 画出当前网络的数据
    plt.plot(alpha, tau, marker='o', linestyle='-', lw=2, label=labels[i])


plt.ylim(0,1)

# 设置坐标轴刻度的字体大小
plt.xticks(np.arange(0.01, 0.06, 0.01), fontsize=18)
plt.yticks(fontsize=18)

# 添加标签和标题
plt.xlabel(r'$\beta$', fontsize=18)
plt.ylabel(r'$\tau$', fontsize=18)
plt.title(r'$\alpha$ = 0.5', fontsize=20)
# 设置图例位置到下方
plt.legend(bbox_to_anchor=(0.85, -0.15), ncol=3, fontsize=16)
plt.subplots_adjust(bottom=0.3)

plt.savefig('C:/Users/59666/Desktop/修改WTOPSIS/α=0.5.png', dpi=600)
# 显示图表
plt.show()