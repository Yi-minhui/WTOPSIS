# -*- coding:utf-8 -*-
# author:ymh
# @Time : 2024/03/12 22:40
# -*- coding:utf-8 -*-
# author:ymh
# @Time : 2024/03/07 16:03

import pandas as pd
from matplotlib.ticker import FormatStrFormatter

import SIR
import numpy as np

import networkx as nx
import community.community_louvain as community

from scipy import stats
import csv
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist


def nodesRank(rank):
    SR = sorted(rank)
    re = []
    for i in SR:
        re.append(rank.index(i))
    return re


def load_graph(path):
    """根据边连边读取网络
    Parameters:
        path:网络存放的路径
    return:
        G:读取后的网络
    """
    G = nx.read_edgelist(path, create_using=nx.Graph())
    return G


def Louvain(G):
    """使用Louvain算法进行社团划分"""

    def com_number(G, partition, community_dic):
        """获得每个节点所连社团个数与社团大小"""
        com_num = {}
        com_size = {}
        for node in G.nodes():
            com_size[node] = len(community_dic[partition[node]])
            com_set = set([partition[node]])
            for nei in list(G.adj[node]):
                if partition[nei] not in com_set:
                    com_set.add(partition[nei])
            com_num[node] = len(com_set)
        return com_num, com_size

    partition = community.best_partition(G)
    community_name = set(list(partition.values()))
    community_dic = {}
    for each in community_name:
        a = []
        for node in list(partition.keys()):
            if partition[node] == each:
                a.append(node)
        community_dic[each] = a
    com_num, com_size = com_number(G, partition, community_dic)
    return community_dic, com_num, com_size


# 读取Excel文件
excel_file = "C:/Users/59666/Desktop/TOPSIS/Road标签/属性权重2.xlsx"
df = pd.read_excel(excel_file)

num_trials = 1000 # 每个比例下的随机采样次数
def ws_topsis(G, rate):
    weights_sum = pd.Series(0, index=['D', 'C', 'B', 'E'])
    # 对每个比例进行10次随机采样
    for _ in range(num_trials):
        # 计算要提取的数据行数
        num_rows_to_extract = int(len(df) * rate)
        # 随机提取的数据行
        random_percent_data = df[['D', 'C', 'B', 'E']].sample(n=num_rows_to_extract)
        # 计算提取的数据和
        total_sum = random_percent_data.sum().sum()
        # 计算每列数据的权重值并累加
        weights_sum += random_percent_data.sum() / total_sum

    # 计算10次随机采样的平均权重值
    weights = weights_sum / num_trials

    # # 计算每列数据的和
    # num_rows_to_extract = int(len(df) * rate)
    # random_percent_data = df[['D', 'C', 'B', 'E']].sample(n=num_rows_to_extract)
    # total_sum = random_percent_data.sum().sum()
    # weights = random_percent_data.sum() / total_sum

    degree_centrality = nx.degree_centrality(G)  # 度中心性，计算每个节点的度（连接数）与总节点数的比例。
    betweenness_centrality = nx.betweenness_centrality(G)  # 介数中心性，度量节点在网络中作为最短路径上的中介者的程度。
    closeness_centrality = nx.closeness_centrality(G)  # 接近中心性，衡量节点与网络中其他节点之间的平均距离。
    eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000, tol=1e-6)  # 特征向量中心性，考虑节点与其邻居之间的相互关系。

    # 使用中心性度量创建一个矩阵
    # 其中每一行代表一个节点，每一列代表一种中心性度量。
    matrix = np.array([
        list(degree_centrality.values()),
        list(betweenness_centrality.values()),
        list(closeness_centrality.values()),
        list(eigenvector_centrality.values())
    ])
    matrix = matrix.T

    # 归一化处理权重
    weights_normalized = weights / np.sum(weights)
    # 扩展权重数组为二维，与矩阵相乘
    weights_expanded = np.tile(weights_normalized, (matrix.shape[0], 1))
    normalized_matrix = matrix / np.sqrt((matrix ** 2).sum(axis=0))
    weighted_matrix = normalized_matrix * weights_expanded

    ideal_best = np.max(weighted_matrix, axis=0)
    ideal_worst = np.min(weighted_matrix, axis=0)

    distance_best = cdist(weighted_matrix, ideal_best.reshape(1, -1), metric="euclidean").flatten()
    distance_worst = cdist(weighted_matrix, ideal_worst.reshape(1, -1), metric="euclidean").flatten()

    s = distance_worst / (distance_best + distance_worst)
    # result_rank = np.argsort(s) + 1
    # 存储结果到字典
    ws_topsis_scores = {}
    for i, node in enumerate(G.nodes()):
        ws_topsis_scores[node] = s[i]
    result_rank = np.array(nodesRank([i for i, j in sorted(ws_topsis_scores.items(), key=lambda x: x[1], reverse=True)]),dtype=float)
    print("jisuanwanbi ")
    return result_rank


def csv_to_dict(filename):
    with open(filename, 'r') as file:
        reader = csv.DictReader(file)
        data = {}
        for row in reader:
            node = row['Node']
            centrality = float(row['Influence Range'])
            data[node] = centrality
    return data


def compare_tau(G):
    ws_topsis_tau_list = []

    # 计算每个比例下的 ws_topsis_rank
    for i in range(1,11):
        ws_topsis_rank = ws_topsis(G, rate=(i+1)/100)
        cascade = csv_to_dict("C:/Users/59666/Desktop/TOPSIS/Road标签/influence_range2.csv")

        # 根据字典的值对键进行排序
        cascade_sort = [i for i, j in sorted(cascade.items(), key=lambda x: x[1], reverse=True)]
        cascade_rank  = np.array(nodesRank(cascade_sort), dtype=float)
        tau, _ = stats.kendalltau(ws_topsis_rank, cascade_rank)

        # 将 Tau 值添加到列表中
        ws_topsis_tau_list.append(tau)
        # print(ws_topsis_tau_list)

        # 保存结果为 CSV 文件
    with open('tau_list.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Proportion', 'Tau'])
        for i, tau in enumerate(ws_topsis_tau_list):
            proportion = (i + 1) / 100
            writer.writerow([proportion, tau])

    return ws_topsis_tau_list


# powergrid_graph=load_graph("C:/Users/59666/Desktop/节点重要性代码/jazz标签/jazz.txt")
# powergrid_graph=load_graph("C:/Users/59666/Desktop/节点重要性代码/supply标签/企业供应链网络.txt")
# powergrid_graph=load_graph("C:/Users/59666/Desktop/节点重要性代码/Email标签/email.txt")
# powergrid_graph=load_graph("C:/Users/59666/Desktop/节点重要性代码/wiki-vote标签/wiki-vote.txt")
# powergrid_graph = load_graph("C:/Users/59666/Desktop/节点重要性代码/FaceBook标签/FaceBook.txt")
# powergrid_graph=load_graph("C:/Users/59666/Desktop/节点重要性代码/USAir97标签/USAir.txt")
powergrid_graph=load_graph("C:/Users/59666/Desktop/TOPSIS/Road标签/Road.txt")
# powergrid_graph=load_graph("C:/Users/59666/Desktop/TOPSIS/Message标签/message.txt")
print(2)

powergrid_graph.remove_edges_from(nx.selfloop_edges(powergrid_graph))

print(3)

_, powergrid_community, _ = Louvain(powergrid_graph)
print(4)

# powergrid_ws_topsis_tau_list0, powergrid_ws_topsis_tau_list1, powergrid_ws_topsis_tau_list2, powergrid_ws_topsis_tau_list3, powergrid_ws_topsis_tau_list4, powergrid_ws_topsis_tau_list5, powergrid_ws_topsis_tau_list6, powergrid_ws_topsis_tau_list7, powergrid_ws_topsis_tau_list8, powergrid_ws_topsis_tau_list9 = compare_tau(
#     powergrid_graph)
print(6)

# a_list = np.arange(0.01, 0.11, 0.01)

# 获取比例列表
proportions = [i / 100 for i in range(1, 11)]

# 获取 Tau 值列表
ws_topsis_tau_list = compare_tau(powergrid_graph)

# 绘制图表
plt.plot(proportions, ws_topsis_tau_list, marker='o', markersize=10, c='teal',)
plt.title('Road', fontsize=20, fontweight='bold')
plt.xlabel('rate', fontsize=20, fontweight='bold')
plt.ylabel(r'$\tau$', fontsize=24, fontweight='bold')
plt.yticks(np.arange(0.6,0.7, 0.02), fontsize=18)
plt.xticks(np.arange(0.01, 0.11, 0.01), fontsize=18)
plt.text(0.97, 0.94, '', fontsize=16, fontweight='bold')

plt.tight_layout()
# plt.legend()
# plt.savefig('C:/Users/59666/Desktop/节点重要性代码/jazz标签/比例肯德尔系数2.png', dpi=600)
# plt.savefig('C:/Users/59666/Desktop/节点重要性代码/Email标签/比例肯德尔系数2.png', dpi=600)
# plt.savefig('C:/Users/59666/Desktop/节点重要性代码/supply标签/比例肯德尔系数2.png', dpi=600)
# plt.savefig('C:/Users/59666/Desktop/节点重要性代码/wiki-vote标签/比例肯德尔系数2.png', dpi=600)
# plt.savefig('C:/Users/59666/Desktop/节点重要性代码/FaceBook标签/比例肯德尔系数1.png', dpi=600)
# plt.savefig('C:/Users/59666/Desktop/节点重要性代码/USAir97标签/比例肯德尔系数2.png', dpi=600)
plt.savefig('C:/Users/59666/Desktop/TOPSIS/Road标签/比例肯德尔系数.png', dpi=600)
plt.show()
