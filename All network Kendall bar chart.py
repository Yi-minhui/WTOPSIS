# -*- coding:utf-8 -*-
# author:ymh
# @Time : 2024/01/02 15:44
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取CSV文件
df1 = pd.read_csv('C:/Users/59666/Desktop/修改WTOPSIS/supply标签/tau_values.csv')
df2 = pd.read_csv('C:/Users/59666/Desktop/修改WTOPSIS/jazz标签/tau_values.csv')
df3 = pd.read_csv('C:/Users/59666/Desktop/修改WTOPSIS/Email标签/tau_values.csv')
df4 = pd.read_csv('C:/Users/59666/Desktop/修改WTOPSIS/USAir97标签/tau_values.csv')
df5 = pd.read_csv('C:/Users/59666/Desktop/修改WTOPSIS/FaceBook标签/tau_values.csv')
df6 = pd.read_csv('C:/Users/59666/Desktop/修改WTOPSIS/wiki-vote标签/tau_values.csv')
df7 = pd.read_csv('C:/Users/59666/Desktop/修改WTOPSIS/Road标签/tau_values.csv')
df8 = pd.read_csv('C:/Users/59666/Desktop/修改WTOPSIS/Infect标签/tau_values.csv')
df9 = pd.read_csv('C:/Users/59666/Desktop/修改WTOPSIS/Message标签/tau_values.csv')

data1 = df1.values.tolist()[0]
data2 = df2.values.tolist()[0]
data3 = df3.values.tolist()[0]
data4 = df4.values.tolist()[0]
data5 = df5.values.tolist()[0]
data6 = df6.values.tolist()[0]
data7 = df7.values.tolist()[0]
data8 = df8.values.tolist()[0]
data9 = df9.values.tolist()[0]

# 自定义颜色列表
custom_colors = ['darkturquoise', '#ff7f0e', '#9467bd', '#1f77b4', '#d62728']

# 创建子图
fig, axs = plt.subplots(3, 3, figsize=(18, 12))

# 绘制子图
for i, (data, ax) in enumerate(zip([data1, data2, data3, data4, data5, data6,data7,data8,data9], axs.flatten())):
    bars = ax.bar(df1.columns[0:], data, color=custom_colors,alpha = 0.7)
    ax.set_title(['Supply', 'Jazz', 'Email', 'USAir', 'FaceBook', 'Vote', 'Road', 'Infect', 'Message'][i], fontweight='bold')
    if i == 0 or i == 3 or i == 6:  # 只在第1个和第4个子图上加 y 轴标签
        ax.set_ylabel(r'$\tau$', fontsize=20, fontweight='bold')

    ax.tick_params(axis='both', which='major', labelsize=10)

    # 在每个柱形上显示数字
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height, '%.4f' % height,
                ha='center', va='bottom', fontsize=8)


# 添加总图例
# fig.legend(df1.columns[1:], loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=len(df1.columns)-1, fontsize=8)
# plt.legend(df1.columns[1:],bbox_to_anchor=(0.65, -0.2), ncol=5, fontsize=18)
# 调整布局
plt.tight_layout()
# 保存图形
plt.savefig('C:/Users/59666/Desktop/修改WTOPSIS/肯德尔对比总.png', dpi=600, bbox_inches='tight')
# 显示图形
plt.show()

