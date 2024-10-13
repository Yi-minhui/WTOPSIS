# -*- coding:utf-8 -*-
# author:ymh
# @Time : 2023/12/01 12:12
import pandas as pd

import failure
import SIR
import numpy as np

import networkx as nx
import community.community_louvain as community

from scipy import stats
import csv
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

print(1)
a_list = np.arange(1, 2, 0.1)

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

def topsis(G):

    degree_centrality = nx.degree_centrality(G)  # 度中心性，计算每个节点的度（连接数）与总节点数的比例。
    betweenness_centrality = nx.betweenness_centrality(G)  # 介数中心性，度量节点在网络中作为最短路径上的中介者的程度。
    closeness_centrality = nx.closeness_centrality(G)  # 接近中心性，衡量节点与网络中其他节点之间的平均距离。
    eigenvector_centrality = nx.eigenvector_centrality(G)  # 特征向量中心性，考虑节点与其邻居之间的相互关系。
    # pagerank = nx.pagerank(G)    # PageRank 算法，用于衡量网页在互联网上的重要性，也可用于网络中的节点。

    # 使用中心性度量创建一个矩阵
    # 其中每一行代表一个节点，每一列代表一种中心性度量。
    matrix = np.array([
        list(degree_centrality.values()),
        list(betweenness_centrality.values()),
        list(closeness_centrality.values()),
        list(eigenvector_centrality.values()),
        # list(pagerank.values())
    ])
    matrix = matrix.T

    weights = [0.25,0.25,0.25,0.25]

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
excel_file = "C:/Users/59666/Desktop/节点重要性代码/jazz标签/属性权重.xlsx"
df = pd.read_excel(excel_file)

num_trials = 10  # 每个比例下的随机采样次数
def ws_topsis(G):
    weights_sum = pd.Series(0, index=['D', 'C', 'B', 'E'])
    # 对每个比例进行10次随机采样
    for _ in range(num_trials):
        # 计算要提取的数据行数
        num_rows_to_extract = int(len(df) * 0.1)
        # 随机提取的数据行
        random_percent_data = df[['D', 'C', 'B', 'E']].sample(n=num_rows_to_extract)
        # 计算提取的数据和
        total_sum = random_percent_data.sum().sum()
        # 计算每列数据的权重值并累加
        weights_sum += random_percent_data.sum() / total_sum

    # 计算10次随机采样的平均权重值
    weights =  weights_sum  / num_trials

    degree_centrality = nx.degree_centrality(G)  # 度中心性，计算每个节点的度（连接数）与总节点数的比例。
    betweenness_centrality = nx.betweenness_centrality(G)  # 介数中心性，度量节点在网络中作为最短路径上的中介者的程度。
    closeness_centrality = nx.closeness_centrality(G)  # 接近中心性，衡量节点与网络中其他节点之间的平均距离。
    eigenvector_centrality = nx.eigenvector_centrality(G)  # 特征向量中心性，考虑节点与其邻居之间的相互关系。

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

    return ws_topsis_scores


def csv_to_dict(filename):
    with open(filename, 'r') as file:
        reader = csv.DictReader(file)
        data = {}
        for row in reader:
            node = row['Node']
            centrality = float(row['TOPSIS Scores'])
            data[node] = centrality
    return data

def compare_tau(G,sir_list):
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
    dc = dict(nx.degree_centrality(G))
    # ks = dict(nx.core_number(G))
    bc = dict(nx.betweenness_centrality(G))
    # vc = Vc(G, community)
    cc = dict(nx.closeness_centrality(G))
    ec = dict(nx.eigenvector_centrality(G))
    Topsis = topsis(G)
    WS_topsis = ws_topsis(G)

    # pagerank = dict(nx.pagerank(G))

    dc_rank = np.array(nodesRank([i for i, j in sorted(dc.items(), key=lambda x: x[1], reverse=True)]), dtype=float)
    # ks_rank = np.array(nodesRank([i for i, j in sorted(ks.items(), key=lambda x: x[1], reverse=True)]), dtype=float)
    bc_rank = np.array(nodesRank([i for i, j in sorted(bc.items(), key=lambda x: x[1], reverse=True)]), dtype=float)
    # vc_rank = np.array(nodesRank([i for i, j in sorted(vc.items(), key=lambda x: x[1], reverse=True)]), dtype=float)
    topsis_rank = np.array(nodesRank([i for i, j in sorted(Topsis.items(), key=lambda x: x[1], reverse=True)]), dtype=float)
    ws_topsis_rank = np.array(nodesRank([i for i, j in sorted(WS_topsis.items(), key=lambda x: x[1], reverse=True)]), dtype=float)
    cc_rank = np.array(nodesRank([i for i, j in sorted(cc.items(), key=lambda x: x[1], reverse=True)]), dtype=float)
    ec_rank = np.array(nodesRank([i for i, j in sorted(ec.items(), key=lambda x: x[1], reverse=True)]), dtype=float)

    topsis_tau_list = []
    ws_topsis_tau_list = []

    dc_tau_list = []
    # ks_tau_list = []
    bc_tau_list = []
    # vc_tau_list = []
    cc_tau_list = []
    ec_tau_list = []

    for sir in sir_list:
        sir_sort = [i for i, j in sorted(sir.items(), key=lambda x: x[1], reverse=True)]
        sir_rank = np.array(nodesRank(sir_sort), dtype=float)

        tau2, _ = stats.kendalltau(topsis_rank, sir_rank)
        tau3, _ = stats.kendalltau(dc_rank, sir_rank)
        # tau3, _ = stats.kendalltau(ks_rank, failure_rank)
        tau4, _ = stats.kendalltau(bc_rank, sir_rank)
        # tau5, _ = stats.kendalltau(vc_rank, failure_rank)
        tau5, _ = stats.kendalltau(ws_topsis_rank, sir_rank)
        tau6, _ = stats.kendalltau(cc_rank, sir_rank)
        tau7, _ = stats.kendalltau(ec_rank, sir_rank)



        topsis_tau_list.append(tau2)
        dc_tau_list.append(tau3)
        # ks_tau_list.append(tau3)
        # nd_tau_list.append(tau4)
        bc_tau_list.append(tau4)
        # vc_tau_list.append(tau5)
        ws_topsis_tau_list.append(tau5)
        cc_tau_list.append(tau6)
        ec_tau_list.append(tau7)


        # 创建一个列表来存储Tau值
        tau_values = [
            dc_tau_list,
            cc_tau_list,
            # nd_tau_list,
            bc_tau_list,
            ec_tau_list,
            topsis_tau_list,
            ws_topsis_tau_list
        ]
        # 转置列表，使每种方法的Tau值在一行中
        tau_values_transposed = list(map(list, zip(*tau_values)))

        # 将Tau值保存到CSV文件
        header = ['DC', 'CC', 'BC','EC','Topsis', 'WS_Topsis']
        with open('tau_values.csv', 'w', newline='', encoding='utf-8') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(header)
            csv_writer.writerows(tau_values_transposed)

    return dc_tau_list, bc_tau_list, cc_tau_list, ec_tau_list, topsis_tau_list, ws_topsis_tau_list


powergrid_graph=load_graph("C:/Users/59666/Desktop/节点重要性代码/jazz标签/jazz.txt")
# powergrid_graph=load_graph("C:/Users/59666/Desktop/节点重要性代码/supply标签/企业供应链网络.txt")
# powergrid_graph=load_graph("C:/Users/59666/Desktop/节点重要性代码/Email标签/email.txt")
# powergrid_graph=load_graph("C:/Users/59666/Desktop/节点重要性代码/wiki-vote标签/wiki-vote.txt")
# powergrid_graph=load_graph("C:/Users/59666/Desktop/节点重要性代码/FaceBook标签/FaceBook.txt")
# powergrid_graph=load_graph("C:/Users/59666/Desktop/节点重要性代码/USAir97标签/USAir.txt")
print(2)

powergrid_graph.remove_edges_from(nx.selfloop_edges(powergrid_graph))

# powergrid_SIR = SIR.load_sir_list('C:/Users/59666/Desktop/节点重要性代码/supply标签/SIR标签/supply_')

powergrid_SIR = SIR.load_sir_list('C:/Users/59666/Desktop/节点重要性代码/jazz标签/SIR标签/jazz_')

# powergrid_SIR = SIR.load_sir_list('C:/Users/59666/Desktop/节点重要性代码/Email标签/SIR标签/Email_')

# powergrid_SIR = SIR.load_sir_list('C:/Users/59666/Desktop/节点重要性代码/wiki-vote标签/SIR标签/vote_')

# powergrid_SIR = SIR.load_sir_list('C:/Users/59666/Desktop/节点重要性代码/FaceBook标签/SIR标签/FaceBook_')

# powergrid_SIR = SIR.load_sir_list('C:/Users/59666/Desktop/节点重要性代码/USAir97标签/SIR标签/USAir_')
print(3)


_, powergrid_community, _ = Louvain(powergrid_graph)
print(4)

# powergrid_topsis = csv_to_dict('C:/Users/59666/Desktop/节点重要性代码/jazz标签/新topsis_output0.01.csv')
# powergrid_ws_topsis = csv_to_dict('C:/Users/59666/Desktop/节点重要性代码/jazz标签/新topsis_output0.05.csv')
# powergrid_w_topsis = csv_to_dict('C:/Users/59666/Desktop/节点重要性代码/jazz标签/SIR加权排序/新topsis_output2.csv')
# powergrid_topsis = csv_to_dict('C:/Users/59666/Desktop/节点重要性代码/supply标签/新topsis_output0.01.csv')
# powergrid_ws_topsis = csv_to_dict('C:/Users/59666/Desktop/节点重要性代码/supply标签/新topsis_output0.05.csv')
# powergrid_w_topsis = csv_to_dict('C:/Users/59666/Desktop/节点重要性代码/supply标签/SIR加权排序/新topsis_output1.csv')
# powergrid_topsis = csv_to_dict('C:/Users/59666/Desktop/节点重要性代码/Email标签/原topsis_output.csv')
# powergrid_ws_topsis = csv_to_dict('C:/Users/59666/Desktop/节点重要性代码/Email标签/新topsis_output0.04.csv')
# powergrid_w_topsis = csv_to_dict('C:/Users/59666/Desktop/节点重要性代码/Email标签/SIR加权排序/新topsis_output0.csv')
# powergrid_topsis = csv_to_dict('C:/Users/59666/Desktop/节点重要性代码/wiki-vote标签/新topsis_output0.01.csv')
# powergrid_ws_topsis = csv_to_dict('C:/Users/59666/Desktop/节点重要性代码/wiki-vote标签/新topsis_output0.05.csv')
# powergrid_w_topsis = csv_to_dict('C:/Users/59666/Desktop/节点重要性代码/wiki-vote标签/SIR加权排序/新topsis_output0.csv')
# powergrid_topsis = csv_to_dict('C:/Users/59666/Desktop/节点重要性代码/FaceBook标签/新topsis_output0.01.csv')
# powergrid_ws_topsis = csv_to_dict('C:/Users/59666/Desktop/节点重要性代码/FaceBook标签/新topsis_output0.02.csv')
# powergrid_w_topsis = csv_to_dict('C:/Users/59666/Desktop/节点重要性代码/FaceBook标签/SIR加权排序/新topsis_output2.csv')
# powergrid_topsis = csv_to_dict('C:/Users/59666/Desktop/节点重要性代码/USAir97标签/原topsis_output.csv')
# powergrid_ws_topsis = csv_to_dict('C:/Users/59666/Desktop/节点重要性代码/USAir97标签/新topsis_output0.02.csv')
# powergrid_w_topsis = csv_to_dict('C:/Users/59666/Desktop/节点重要性代码/USAir97标签/SIR加权排序/新topsis_output8.csv')
print(5)

powergrid_topsis_rank=topsis(powergrid_graph)
powergrid_ws_topsis_rank=ws_topsis(powergrid_graph)
print(6)

powergrid_dc_tau, powergrid_bc_tau, powergrid_cc_tau, powergrid_ec_tau, powergrid_topsis_tau, powergrid_ws_topsis_tau=compare_tau(powergrid_graph,powergrid_SIR)
print(7)

# plt.subplot(111)
plt.plot(a_list, powergrid_ws_topsis_tau, marker='o', markersize=10, c='r', label='WS-TOPSIS')
plt.plot(a_list, powergrid_topsis_tau, marker='d', markersize=10, c='b', label='TOPSIS')
plt.plot(a_list, powergrid_dc_tau, marker='<', markersize=10, c='fuchsia', label='DC')
# plt.plot(a_list, powergrid_ks_tau, marker='>', markersize=10, c='g', label='K-core')
plt.plot(a_list, powergrid_bc_tau, marker='h', markersize=10, c='y', label='BC')
plt.plot(a_list, powergrid_cc_tau, marker='H', markersize=10, c='orange', label='CC')
# plt.plot(a_list, powergrid_ec_tau, marker='>', markersize=10, c='g', label='EC')
# plt.plot(a_list, powergrid_vc_tau, marker='H', markersize=10, c='orange', label='Vc')
plt.title('jazz', fontsize=20, fontweight='bold')
plt.xlabel(r'$β/β_{th}$', fontsize=20, fontweight='bold')
plt.ylabel(r'$\tau$',fontsize=24,fontweight='bold')
plt.yticks(np.arange(0, 1.1, 0.2), fontsize=18)
plt.xticks(np.arange(1, 2, 0.3), fontsize=18)
plt.text(0.97, 0.94,'', fontsize=16, fontweight='bold')


plt.tight_layout()
plt.legend()
plt.savefig('C:/Users/59666/Desktop/节点重要性代码/jazz标签/肯德尔系数0 .png', dpi=600)
# plt.savefig('C:/Users/59666/Desktop/节点重要性代码/Email标签/肯德尔系数0.04 .png', dpi=600)
# plt.savefig('C:/Users/59666/Desktop/节点重要性代码/supply标签/肯德尔系数.png', dpi=600)
# plt.savefig('C:/Users/59666/Desktop/节点重要性代码/wiki-vote标签/肯德尔系数0 .png', dpi=600)
# plt.savefig('C:/Users/59666/Desktop/节点重要性代码/FaceBook标签/肯德尔系数0.png', dpi=600)
# plt.savefig('C:/Users/59666/Desktop/节点重要性代码/USAir97标签/肯德尔系数0.02 .png', dpi=600)
plt.show()

