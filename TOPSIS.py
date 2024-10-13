# -*- coding:utf-8 -*-
# author:ymh
# @Time : 2024/09/15 10:31
import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import community as community_detection
import Utils




np.random.seed(42)
# # 处理单个网络的函数

def process_network(file_path, output_prefix):
    # 读取网络数据
    df = pd.read_csv(file_path)

    # 创建无向图
    G = nx.Graph()

    # 添加边
    for index, row in df.iterrows():
        G.add_edge(row['FromNodeId'], row['ToNodeId'])


    # def social_capital(G):
    #     centrality = {}
    #     for node in G.nodes:
    #         node_degree = G.degree(node)
    #         neighbor_degrees_sum = sum(G.degree(neighbor) for neighbor in G.neighbors(node))
    #         centrality[node] = node_degree + neighbor_degrees_sum
    #     centrality_values = list(centrality.values())
    #     centrality_min = min(centrality_values)
    #     centrality_max = max(centrality_values)
    #     if centrality_max == centrality_min:
    #         normalized_centrality = {node: 0.0 for node in centrality}
    #     else:
    #         normalized_centrality = {
    #             node: (value - centrality_min) / (centrality_max - centrality_min)
    #             for node, value in centrality.items()
    #         }
    #     return normalized_centrality

    def collective_influence(G, l):
        influence = {}
        for node in G.nodes:
            node_degree = G.degree(node)
            neighborhood = set()
            for i in range(1, l + 1):
                nodes_at_distance_i = nx.single_source_shortest_path_length(G, node, cutoff=i).keys()
                neighborhood.update(nodes_at_distance_i)
            neighborhood.discard(node)
            influence_sum = 0
            for neighbor in neighborhood:
                neighbor_degree = G.degree(neighbor)
                influence_sum += (neighbor_degree - 1)
            influence[node] = (node_degree - 1) * influence_sum
        CI_values = list(influence.values())
        CI_min = min(CI_values)
        CI_max = max(CI_values)
        if CI_max == CI_min:
            normalized_influence = {node: 0.0 for node in influence}
        else:
            normalized_influence = {
                node: (value - CI_min) / (CI_max - CI_min)
                for node, value in influence.items()
            }
        return normalized_influence


    community = community_detection.best_partition(G)
    vc = Utils.Vc(G,community)
    nd = Utils.neighbor_degree(G)
    # social_capital_centrality = social_capital(G)
    collective_influence = collective_influence(G,2)
    # effg_centrality = compute_effg(G)



    # 使用中心性度量创建一个矩阵
    matrix = np.array([
        list(vc.values()),
        list(nd.values()),
        list(collective_influence.values()),
        # list(effg_centrality.values()),

    ])
    matrix = matrix.T

    def topsis(matrix, G):
        weights = [1/3,1/3,1/3]
        weights_normalized = weights / np.sum(weights)
        normalized_matrix = matrix / np.sqrt((matrix ** 2).sum(axis=0))
        weighted_matrix = normalized_matrix * weights_normalized
        ideal_best = np.max(weighted_matrix, axis=0)
        ideal_worst = np.min(weighted_matrix, axis=0)
        distance_best = cdist(weighted_matrix, ideal_best.reshape(1, -1), metric="euclidean").flatten()
        distance_worst = cdist(weighted_matrix, ideal_worst.reshape(1, -1), metric="euclidean").flatten()
        s = distance_worst / (distance_best + distance_worst)
        topsis_scores = {}
        for i, node in enumerate(G.nodes()):
            topsis_scores[node] = s[i]
        return topsis_scores

    topsis_scores = topsis(matrix, G)
    topsis_series = pd.Series(topsis_scores)

    # 保存 TOPSIS 结果
    topsis_df = pd.DataFrame({
        "Node": topsis_series.index,
        "TOPSIS Scores": topsis_series.values
    })
    topsis_df.to_csv(f"{output_prefix}_原topsis_output.csv", index=False)

    # 保存社区中心性等其他指标，加入Node列
    metrics_data = pd.DataFrame({
        "Node": list(G.nodes()),  # 添加节点列
        "vc": list(vc.values()),
        "nd": list(nd.values()),
        "ci": list(collective_influence.values()),
        # "effg_centrality": list(effg_centrality.values()),

    })
    metrics_data.to_excel(f"{output_prefix}_metrics_data_output.xlsx", index=False)

# 文件路径列表
file_paths = [
    "C:/Users/59666/Desktop/节点重要性代码/supply标签/企业供应链网络.csv",
    # "C:/Users/59666/Desktop/节点重要性代码/Jazz标签/jazz.csv",
    "C:/Users/59666/Desktop/节点重要性代码/Email标签/email.csv",
    "C:/Users/59666/Desktop/节点重要性代码/wiki-vote标签/wiki-vote.csv",
    "C:/Users/59666/Desktop/节点重要性代码/USAir97标签/USAir.csv",
    "C:/Users/59666/Desktop/节点重要性代码/FaceBook标签/FaceBook.csv",
    "C:/Users/59666/Desktop/TOPSIS/Road标签/Road.csv",
    "C:/Users/59666/Desktop/TOPSIS/Infect标签/infect.csv",
    "C:/Users/59666/Desktop/TOPSIS/Message标签/message.csv"
]

# 对多个网络文件执行分析
for file_path in file_paths:
    # 提取文件名作为输出前缀
    output_prefix = file_path.split('/')[-1].split('.')[0]
    process_network(file_path, output_prefix)
