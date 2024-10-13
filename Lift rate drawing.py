# -*- coding:utf-8 -*-
# author:ymh
# @Time : 2024/01/02 15:44

import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
df1 = pd.read_csv('C:/Users/59666/Desktop/修改WTOPSIS/wiki-vote标签/tau_values.csv')
df2 = pd.read_csv('C:/Users/59666/Desktop/修改WTOPSIS/supply标签/tau_values.csv')
df3 = pd.read_csv('C:/Users/59666/Desktop/修改WTOPSIS/jazz标签/tau_values.csv')
df4 = pd.read_csv('C:/Users/59666/Desktop/修改WTOPSIS/Email标签/tau_values.csv')
df5 = pd.read_csv('C:/Users/59666/Desktop/修改WTOPSIS/USAir97标签/tau_values.csv')
df6 = pd.read_csv('C:/Users/59666/Desktop/修改WTOPSIS/FaceBook标签/tau_values.csv')
df7 = pd.read_csv('C:/Users/59666/Desktop/修改WTOPSIS/Road标签/tau_values.csv')
df8 = pd.read_csv('C:/Users/59666/Desktop/修改WTOPSIS/Infect标签/tau_values.csv')
df9 = pd.read_csv('C:/Users/59666/Desktop/修改WTOPSIS/Message标签/tau_values.csv')


# 提取每个DataFrame的最后一列和倒数第二列数据
data1_last_col = df1.iloc[:, -1]
data1_second_last_col = df1.iloc[:, -2]
data2_last_col = df2.iloc[:, -1]
data2_second_last_col = df2.iloc[:, -2]
data3_last_col = df3.iloc[:, -1]
data3_second_last_col = df3.iloc[:, -2]
data4_last_col = df4.iloc[:, -1]
data4_second_last_col = df4.iloc[:, -2]
data5_last_col = df5.iloc[:, -1]
data5_second_last_col = df5.iloc[:, -2]
data6_last_col = df6.iloc[:, -1]
data6_second_last_col = df6.iloc[:, -2]
data7_last_col = df7.iloc[:, -1]
data7_second_last_col = df7.iloc[:, -2]
data8_last_col = df8.iloc[:, -1]
data8_second_last_col = df8.iloc[:, -2]
data9_last_col = df9.iloc[:, -1]
data9_second_last_col = df9.iloc[:, -2]

# 计算每个算法的提升率（最后一列减去倒数第二列）
data1_increase_rate = data1_last_col - data1_second_last_col
data2_increase_rate = data2_last_col - data2_second_last_col
data3_increase_rate = data3_last_col - data3_second_last_col
data4_increase_rate = data4_last_col - data4_second_last_col
data5_increase_rate = data5_last_col - data5_second_last_col
data6_increase_rate = data6_last_col - data6_second_last_col
data7_increase_rate = data7_last_col - data7_second_last_col
data8_increase_rate = data8_last_col - data8_second_last_col
data9_increase_rate = data9_last_col - data9_second_last_col

# 绘制柱状图
fig, ax = plt.subplots(figsize=(10, 6))

x_labels = ['Vote','Supply', 'Jazz', 'Email', 'USAir', 'FaceBook', 'Road','Infect','Message']
x = range(len(x_labels))

bars = ax.bar(x, [data1_increase_rate.mean(), data2_increase_rate.mean(), data3_increase_rate.mean(), data4_increase_rate.mean(), data5_increase_rate.mean(), data6_increase_rate.mean(),data7_increase_rate.mean(), data8_increase_rate.mean(), data9_increase_rate.mean()],
       color=['b', 'g', 'r', 'darkturquoise', 'm', 'goldenrod','#9467bd','#1f77b4','#ff7f0e'], alpha=0.5)

# 画折线
# line = ax.plot(x, [data1_increase_rate.mean(), data2_increase_rate.mean(), data3_increase_rate.mean(), data4_increase_rate.mean(), data5_increase_rate.mean(), data6_increase_rate.mean()], marker='o', color='black', linestyle='-')

# 在柱形顶部标上数字
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height, '{:.4f}'.format(height), ha='center', va='bottom', fontsize=12)

# ax.set_xlabel('Algorithm')
ax.set_ylabel('Improve Rate', fontsize=18)
plt.yticks(fontsize=14)
# ax.set_title('Mean Increase Rate of Last Column Minus Second Last Column')
ax.set_xticks(x)
ax.set_xticklabels(x_labels, fontsize=14)
plt.savefig('C:/Users/59666/Desktop/修改WTOPSIS/提升率.png', dpi=600)

plt.show()




