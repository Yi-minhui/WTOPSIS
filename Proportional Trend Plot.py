# -*- coding:utf-8 -*-
# author:ymh
# @Time : 2024/03/10 10:49
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import seaborn as sns

warnings.filterwarnings('ignore')
sns.set_style('ticks')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 从CSV文件读取数据
df1 = pd.read_csv('C:/Users/59666/Desktop/节点重要性代码/supply标签/tau_list1.csv')
df2 = pd.read_csv('C:/Users/59666/Desktop/节点重要性代码/jazz标签/tau_list1.csv')
df3 = pd.read_csv('C:/Users/59666/Desktop/节点重要性代码/Email标签/tau_list1.csv')
df4 = pd.read_csv('C:/Users/59666/Desktop/节点重要性代码/USAir97标签/tau_list1.csv')
df5 = pd.read_csv('C:/Users/59666/Desktop/节点重要性代码/FaceBook标签/tau_list1.csv')
df6 = pd.read_csv('C:/Users/59666/Desktop/节点重要性代码/wiki-vote标签/tau_list1.csv')
df7 = pd.read_csv('C:/Users/59666/Desktop/TOPSIS/Road标签/tau_list.csv')
df8 = pd.read_csv('C:/Users/59666/Desktop/TOPSIS/Infect标签/tau_list.csv')
df9 = pd.read_csv('C:/Users/59666/Desktop/TOPSIS/Message标签/tau_list.csv')
# 提取需要绘制的数据列
a_list = np.arange(0.01, 0.11, 0.01)

supply_ws_topsis_tau = df1['Tau'].to_numpy()
jazz_ws_topsis_tau = df2['Tau'].to_numpy()
Email_ws_topsis_tau = df3['Tau'].to_numpy()
USAir_ws_topsis_tau = df4['Tau'].to_numpy()
FaceBook_ws_topsis_tau = df5['Tau'].to_numpy()
vote_ws_topsis_tau = df6['Tau'].to_numpy()
Road_ws_topsis_tau = df7['Tau'].to_numpy()
Infect_ws_topsis_tau = df8['Tau'].to_numpy()
Message_ws_topsis_tau = df9['Tau'].to_numpy()


# 绘图
plt.figure(figsize=(9,7), dpi=120)

plt.plot(a_list, supply_ws_topsis_tau, marker='H', markersize=12, linewidth=2, markeredgecolor='b', markeredgewidth=2, c='b', label='Supply')
plt.plot(a_list, jazz_ws_topsis_tau, marker='o', markersize=12, linewidth=2, markeredgecolor='darkturquoise', markeredgewidth=2, c='darkturquoise', label='Jazz')
plt.plot(a_list, Email_ws_topsis_tau, marker='s', markersize=12, linewidth=2,  markeredgecolor='brown', markeredgewidth=2, c='brown', label='Email')
plt.plot(a_list, USAir_ws_topsis_tau, marker='>', markersize=12,  linewidth=2, markeredgecolor='green', markeredgewidth=2, c='green', label='USAir')
plt.plot(a_list, FaceBook_ws_topsis_tau, marker='d', markersize=12, linewidth=2, markeredgecolor='orangered', markeredgewidth=2, c='orangered', label='FaceBook')
plt.plot(a_list, vote_ws_topsis_tau, marker='h', markersize=12, linewidth=2, markeredgecolor='steelblue', markeredgewidth=2, c='steelblue', label='Vote')
plt.plot(a_list, Road_ws_topsis_tau, marker='P', markersize=12, linewidth=2, markeredgecolor='#9467bd', markeredgewidth=2, c='#9467bd', label='Road')
plt.plot(a_list, Infect_ws_topsis_tau, marker='D', markersize=12, linewidth=2, markeredgecolor='#1f77b4', markeredgewidth=2, c='#1f77b4', label='Infect')
plt.plot(a_list, Message_ws_topsis_tau, marker='<', markersize=12, linewidth=2, markeredgecolor='#ff7f0e', markeredgewidth=2, c='#ff7f0e', label='Message')


# plt.title('improve_rate', fontsize=20, fontweight='bold')
plt.ylabel(r'$\tau$',fontsize=24)
plt.xlabel('rate', fontsize=20, fontweight='bold')
plt.yticks(np.arange(0.5, 1, 0.04), fontsize=18)
plt.xticks(np.arange(0.01, 0.11, 0.01), fontsize=18)
plt.text(1.4,-0.07,'',fontsize=16,fontweight='bold')


plt.tight_layout()
# 调整图例的位置
plt.legend(bbox_to_anchor=(0.85, -0.15), ncol=3, fontsize=18)
plt.subplots_adjust(bottom=0.3)
plt.savefig('C:/Users/59666/Desktop/TOPSIS/比例趋势.png',dpi=600)
plt.show()



