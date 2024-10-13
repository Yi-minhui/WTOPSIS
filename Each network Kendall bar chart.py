# -*- coding:utf-8 -*-
# author:ymh
# @Time : 2024/03/16 17:21
from random import random

import pandas as pd

import failure
import SIR
import numpy as np

import networkx as nx
import community.community_louvain as community
import community as community_detection

from scipy import stats
import csv
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# 设置全局随机种子
np.random.seed(42)

def nodesRank(rank):
    SR = sorted(rank)
    re = []
    for i in SR:
        re.append(rank.index(i))
    return re

def Vc(G,community):
    vc = {}
    for node in list(G.nodes()):
        com_set = set({community[node]})
        for nei in list(G.adj[node]):
            if community[nei] not in com_set:
                com_set.add(community[nei])
        vc[node] = len(com_set)
    return vc

def neighbor_degree(G):
    """邻居度：邻居节点的度之和"""
    nodes = list(G.nodes())
    degree = dict(G.degree())
    n_degree = {}
    for node in nodes:
        nd = 0
        neighbors = G.adj[node]
        for nei in neighbors:
            nd+=degree[nei]
        n_degree[node]=nd
    return n_degree
# neighbor_degree 函数：计算邻居度，即邻居节点的度之和。

def load_graph(path):
    """根据边连边读取网络
    Parameters:
        path:网络存放的路径
    return:
        G:读取后的网络
    """
    G = nx.read_edgelist(path,create_using=nx.Graph())
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


def percolation(G):
    """
    计算图 G 中每个节点的渗滤中心性 (Percolation Centrality, PC)

    参数:
    G -- 网络图 (networkx Graph)

    返回:
    每个节点的渗滤中心性值 (字典)
    """
    # 初始化中心性字典
    pc = {node: 0 for node in G.nodes}

    # 对于每一个节点，计算它在不同时间步骤下的最短路径影响
    for node in G.nodes:
        # 计算包含该节点的所有最短路径
        sp_with_node = 0
        sp_total = 0
        for source in G.nodes:
            for target in G.nodes:
                if source != target and node in nx.shortest_path(G, source=source, target=target):
                    sp_with_node += 1
                if source != target:
                    sp_total += 1

        # 计算该节点的渗滤中心性值
        pc[node] = sp_with_node / sp_total if sp_total > 0 else 0

    return pc


# def social_capital(G):
#     """
#     计算网络中每个节点的社会资本中心性，并进行最小-最大归一化。
#
#     参数：
#     G (networkx.Graph): 网络图。
#
#     返回：
#     dict: 节点及其归一化后的社会资本中心性。
#     """
#     centrality = {}
#
#     # 计算社会资本中心性
#     for node in G.nodes:
#         # 计算节点的度数
#         node_degree = G.degree(node)
#
#         # 计算邻居节点的度数之和
#         neighbor_degrees_sum = sum(G.degree(neighbor) for neighbor in G.neighbors(node))
#
#         # 计算社会资本中心性
#         centrality[node] = node_degree + neighbor_degrees_sum
#
#     # 进行最小-最大归一化
#     centrality_values = list(centrality.values())
#     centrality_min = min(centrality_values)
#     centrality_max = max(centrality_values)
#
#     # 避免除以零
#     if centrality_max == centrality_min:
#         normalized_centrality = {node: 0.0 for node in centrality}
#     else:
#         normalized_centrality = {
#             node: (value - centrality_min) / (centrality_max - centrality_min)
#             for node, value in centrality.items()
#         }
#
#     return normalized_centrality


def collective_influence(G, l):
    """
    计算网络中每个节点的集体影响力，并进行最小-最大归一化。

    参数：
    G (networkx.Graph): 网络图。
    l (int): 半径值，通常大中型网络设为 3，小型网络设为 2。

    返回：
    dict: 节点及其归一化后的集体影响力。
    """
    influence = {}

    # 计算集体影响力
    for node in G.nodes:
        node_degree = G.degree(node)

        # 收集半径为 l 的球内的所有节点
        neighborhood = set()
        for i in range(1, l + 1):
            nodes_at_distance_i = nx.single_source_shortest_path_length(G, node, cutoff=i).keys()
            neighborhood.update(nodes_at_distance_i)

        # 移除节点本身
        neighborhood.discard(node)

        # 计算集体影响力
        influence_sum = 0
        for neighbor in neighborhood:
            neighbor_degree = G.degree(neighbor)
            influence_sum += (neighbor_degree - 1)

        influence[node] = (node_degree - 1) * influence_sum

    # 进行最小-最大归一化
    CI_values = list(influence.values())
    CI_min = min(CI_values)
    CI_max = max(CI_values)

    # 避免除以零
    if CI_max == CI_min:
        normalized_influence = {node: 0.0 for node in influence}
    else:
        normalized_influence = {
            node: (value - CI_min) / (CI_max - CI_min)
            for node, value in influence.items()
        }

    return normalized_influence



def topsis(G):
    community = community_detection.best_partition(G)
    vc = Vc(G, community)
    nd = neighbor_degree(G)
    collective_influence_centrality = collective_influence(G, 2)

  # 特征向量中心性，考虑节点与其邻居之间的相互关系。
    # pagerank = nx.pagerank(G)    # PageRank 算法，用于衡量网页在互联网上的重要性，也可用于网络中的节点。

    # 使用中心性度量创建一个矩阵
    # 其中每一行代表一个节点，每一列代表一种中心性度量。
    matrix = np.array([
        list(vc.values()),
        list(nd.values()),
        list(collective_influence_centrality.values()),
        # list(pagerank.values())
    ])
    matrix = matrix.T

    weights = [1/3,1/3,1/3]

    # 归一化处理权重
    weights_normalized = weights / np.sum(weights)
    normalized_matrix = matrix / np.sqrt((matrix ** 2).sum(axis=0))
    weighted_matrix = normalized_matrix * weights_normalized

    ideal_best = np.max(weighted_matrix, axis=0)
    ideal_worst = np.min(weighted_matrix, axis=0)

    distance_best = cdist(weighted_matrix, ideal_best.reshape(1, -1), metric="euclidean").flatten()
    distance_worst = cdist(weighted_matrix, ideal_worst.reshape(1, -1), metric="euclidean").flatten()

    s = distance_worst / (distance_best + distance_worst)
    # result_rank = np.argsort(s) + 1
    # 存储结果到字典
    topsis_scores = {}
    for i, node in enumerate(G.nodes()):
        topsis_scores[node] = s[i]

    return topsis_scores


# 读取Excel文件
excel_file = "C:/Users/59666/Desktop/修改WTOPSIS/wiki-vote标签/属性权重_alpha_1.0_lambda_0.01.xlsx"
df = pd.read_excel(excel_file)

num_trials = 1000  # 每个比例下的随机采样次数
def ws_topsis(G):
    # 去除包含空值的行
    df_cleaned = df[['VC','ND','CI']].dropna()  # 确保只使用不含空值的行

    weights_sum = pd.Series(0, index=['VC','ND','CI'])
    # 对每个比例进行num_trials次随机采样
    for _ in range(num_trials):
        # 计算要提取的数据行数（1%）
        num_rows_to_extract = int(len(df_cleaned) * 0.01)
        # 随机提取没有空值的行
        random_percent_data = df_cleaned.sample(n=num_rows_to_extract)
        # 计算提取的数据和
        total_sum = random_percent_data.sum().sum()
        # 计算每列数据的权重值并累加
        weights_sum += random_percent_data.sum() / total_sum

    # 计算10次随机采样的平均权重值
    weights =  weights_sum  / num_trials

    community = community_detection.best_partition(G)
    vc = Vc(G, community)
    nd = neighbor_degree(G)
    collective_influence_centrality = collective_influence(G, 2)

    # 使用中心性度量创建一个矩阵
    # 其中每一行代表一个节点，每一列代表一种中心性度量。
    matrix = np.array([
        list(vc.values()),
        list(nd.values()),
        list(collective_influence_centrality.values()),
        # list(pagerank.values())
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

    return ws_topsis_scores


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
    """使用肯德尔相关系数对比不同方法
    Parameters:
        G:目标网络
        dc:度中心性
        bc:介数中心性
        ks:k-shell
        sir_list:不同beta下的SIR模拟结果
        model:训练好的模型
        p:选择比较节点的比例
    return:
        tau_list:在不同beta情况下的tau值
    """
    # dc = dict(nx.degree_centrality(G))
    community = community_detection.best_partition(G)
    vc = Vc(G, community)
    # bc = dict(nx.betweenness_centrality(G))
    nd = neighbor_degree(G)
    ci = dict(collective_influence(G,2))
    # cc = dict(nx.closeness_centrality(G))
    # ec = dict(nx.eigenvector_centrality(G, max_iter=1000, tol=1e-6))
    Topsis = topsis(G)
    WS_topsis = ws_topsis(G)

    # pagerank = dict(nx.pagerank(G))

    # dc_rank = np.array(nodesRank([i for i, j in sorted(dc.items(), key=lambda x: x[1], reverse=True)]), dtype=float)
    # bc_rank = np.array(nodesRank([i for i, j in sorted(bc.items(), key=lambda x: x[1], reverse=True)]), dtype=float)
    vc_rank = np.array(nodesRank([i for i, j in sorted(vc.items(), key=lambda x: x[1], reverse=True)]), dtype=float)
    nd_rank = np.array(nodesRank([i for i, j in sorted(nd.items(), key=lambda x: x[1], reverse=True)]), dtype=float)
    ci_rank = np.array(nodesRank([i for i, j in sorted(ci.items(), key=lambda x: x[1], reverse=True)]), dtype=float)
    topsis_rank = np.array(nodesRank([i for i, j in sorted(Topsis.items(), key=lambda x: x[1], reverse=True)]), dtype=float)
    ws_topsis_rank = np.array(nodesRank([i for i, j in sorted(WS_topsis.items(), key=lambda x: x[1], reverse=True)]), dtype=float)
    # cc_rank = np.array(nodesRank([i for i, j in sorted(cc.items(), key=lambda x: x[1], reverse=True)]), dtype=float)
    # ec_rank = np.array(nodesRank([i for i, j in sorted(ec.items(), key=lambda x: x[1], reverse=True)]), dtype=float)

    topsis_tau_list = []
    ws_topsis_tau_list = []

    # dc_tau_list = []
    vc_tau_list = []
    # bc_tau_list = []
    nd_tau_list = []
    ci_tau_list = []
    # cc_tau_list = []
    # ec_tau_list = []


    # ws_topsis_rank = ws_topsis(G)
    cascade = csv_to_dict("C:/Users/59666/Desktop/不同参数下级联失效得分/results/wiki-vote/influence_range_alpha_1.0_lambda_0.01.csv")
    # cascade1 = csv_to_dict("C:/Users/59666/Desktop/TOPSIS/Road标签/influence_range2.csv")
    # 根据字典的值对键进行排序
    cascade_sort = [i for i, j in sorted(cascade.items(), key=lambda x: x[1], reverse=True)]
    # cascade_sort1 = [i for i, j in sorted(cascade1.items(), key=lambda x: x[1], reverse=True)]

    cascade_rank  = np.array(nodesRank(cascade_sort), dtype=float)
    # cascade_rank1 = np.array(nodesRank(cascade_sort), dtype=float)

    tau1, _ = stats.kendalltau(vc_rank, cascade_rank)
    tau2, _ = stats.kendalltau(nd_rank, cascade_rank)
    tau3, _ = stats.kendalltau(ci_rank, cascade_rank)
    # tau3, _ = stats.kendalltau(ks_rank, cascade_rank)
    # tau4, _ = stats.kendalltau(ec_rank, cascade_rank)
    # tau5, _ = stats.kendalltau(vc_rank, failure_rank)
    tau4, _ = stats.kendalltau(topsis_rank, cascade_rank)
    tau5, _ = stats.kendalltau(ws_topsis_rank, cascade_rank)

    vc_tau_list.append(tau1)
    nd_tau_list.append(tau2)
    ci_tau_list.append(tau3)
    # ec_tau_list.append(tau4)
    topsis_tau_list.append(tau4)
    ws_topsis_tau_list.append(tau5)
    # ks_tau_list.append(tau3)
    # nd_tau_list.append(tau4)

    # vc_tau_list.append(tau5)



    # 创建一个列表来存储Tau值
    tau_values = [
        vc_tau_list,
        nd_tau_list,
        ci_tau_list,
        # ec_tau_list,
        # ks_tau_list,
        topsis_tau_list,
        ws_topsis_tau_list
    ]
    # 转置列表，使每种方法的Tau值在一行中
    tau_values_transposed = list(map(list, zip(*tau_values)))

    # 将Tau值保存到CSV文件
    header = ['VC', 'ND', 'CI','Topsis', 'WC_Topsis']
    with open('tau_values.csv', 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(header)
        csv_writer.writerows(tau_values_transposed)

    return vc_tau_list, nd_tau_list, ci_tau_list, topsis_tau_list, ws_topsis_tau_list


# powergrid_graph=load_graph("C:/Users/59666/Desktop/节点重要性代码/supply标签/企业供应链网络.txt")
# powergrid_graph=load_graph("C:/Users/59666/Desktop/节点重要性代码/jazz标签/jazz.txt")
# powergrid_graph=load_graph("C:/Users/59666/Desktop/节点重要性代码/Email标签/email.txt")
powergrid_graph=load_graph("C:/Users/59666/Desktop/节点重要性代码/wiki-vote标签/wiki-vote.txt")
# powergrid_graph=load_graph("C:/Users/59666/Desktop/节点重要性代码/FaceBook标签/FaceBook.txt")
# powergrid_graph=load_graph("C:/Users/59666/Desktop/节点重要性代码/USAir97标签/USAir.txt")
# powergrid_graph=load_graph("C:/Users/59666/Desktop/TOPSIS/jazz标签/jazz.txt")
# powergrid_graph=load_graph("C:/Users/59666/Desktop/TOPSIS/Infect标签/infect.txt")
# powergrid_graph=load_graph("C:/Users/59666/Desktop/TOPSIS/Message标签/message.txt")
# powergrid_graph=load_graph("C:/Users/59666/Desktop/TOPSIS/Road标签/Road.txt")

print(2)

powergrid_graph.remove_edges_from(nx.selfloop_edges(powergrid_graph))

print(3)


_, powergrid_community, _ = Louvain(powergrid_graph)
print(4)

powergrid_vc_tau,powergrid_nd_tau, powergrid_ci_tau, powergrid_topsis_tau, powergrid_ws_topsis_tau=compare_tau(powergrid_graph)
print(5)


# 算法名称
algorithms = ['VC', 'ND', 'CI','Topsis', 'WC_Topsis']

# 每个算法的肯德尔相关系数值
tau_values = [
    powergrid_vc_tau,
    powergrid_nd_tau,
    powergrid_ci_tau,
    # powergrid_ec_tau,
    powergrid_topsis_tau,
    powergrid_ws_topsis_tau
]

# 将tau_values展平成一维数组
tau_values_flat = [item for sublist in tau_values for item in sublist]

# 不同算法的颜色'cyan',
colors = ['darkturquoise', 'green','orange', 'blue', 'red']

# 绘图
plt.figure(figsize=(10, 6))
bars = plt.bar(algorithms, tau_values_flat, color=colors)
plt.title('Vote', fontsize=20, fontweight='bold')
# plt.xlabel('algorithms ', fontsize=20, fontweight='bold')
plt.ylabel(r'$\tau$',fontsize=20,fontweight='bold')
plt.xticks(fontsize=20)
plt.yticks(np.arange(0, 1.1, 0.2), fontsize=20)
plt.text(0.97, 0.94,'', fontsize=16, fontweight='bold')
# 标注每个柱形上方的 y 值
for bar, value in zip(bars, tau_values_flat):
    plt.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f'{value:.4f}', ha='center', va='bottom', fontsize=14)

plt.tight_layout()
# plt.legend()
# plt.savefig('C:/Users/59666/Desktop/TOPSIS/jazz标签/肯德尔系1.png', dpi=600)
plt.savefig('C:/Users/59666/Desktop/修改WTOPSIS/肯德尔系数alpha=1.0, lamda=0.01.png', dpi=600)
# plt.savefig('C:/Users/59666/Desktop/TOPSIS/Message标签/肯德尔系数.png', dpi=600)
# plt.savefig('C:/Users/59666/Desktop/TOPSIS/supply标签/肯德尔系数.png', dpi=600)
# plt.savefig('C:/Users/59666/Desktop/TOPSIS/wiki-vote标签/肯德尔系数1.png', dpi=600)
# plt.savefig('C:/Users/59666/Desktop/TOPSIS/FaceBook标签/肯德尔系数.png', dpi=600)
# plt.savefig('C:/Users/59666/Desktop/TOPSIS/USAir97标签/肯德尔系数1.png', dpi=600)
plt.show()