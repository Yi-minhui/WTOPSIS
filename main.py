# -*- coding:utf-8 -*-
# author:ymh
# @Time : 2024/01/03 12:18
import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import community as community_detection
import Utils

np.random.seed(42)
# 读取Excel文件
df = pd.read_csv("C:/Users/59666/Desktop/节点重要性代码/supply标签/企业供应链网络.csv")
# df = pd.read_csv("C:/Users/59666/Desktop/节点重要性代码/jazz标签/jazz.csv")
# df = pd.read_csv("C:/Users/59666/Desktop/节点重要性代码/Email标签/email.csv")
# df = pd.read_csv("C:/Users/59666/Desktop/节点重要性代码/wiki-vote标签/wiki-vote.csv")
# df = pd.read_csv("C:/Users/59666/Desktop/节点重要性代码/USAir97标签/USAir.csv")
# df = pd.read_csv("C:/Users/59666/Desktop/节点重要性代码/FaceBook标签/FaceBook.csv")
# df = pd.read_csv("C:/Users/59666/Desktop/TOPSIS/Road标签/Road.csv")
# df = pd.read_csv("C:/Users/59666/Desktop/TOPSIS/Infect标签/infect.csv")
# df = pd.read_csv("C:/Users/59666/Desktop/TOPSIS/Message标签/message.csv")
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
vc = Utils.Vc(G, community)
nd = Utils.neighbor_degree(G)
# social_capital_centrality = social_capital(G)
collective_influence = collective_influence(G, 2)
# effg_centrality = compute_effg(G)


# 使用中心性度量创建一个矩阵
# 其中每一行代表一个节点，每一列代表一种中心性度量。
matrix = np.array([
    list(vc.values()),
    list(nd.values()),
    list(collective_influence.values()),
    # list(effg_centrality.values()),

])
# 转置矩阵，使节点成为行，中心性度量成为列
matrix = matrix.T
# print(matrix)

# 读取Excel文件
excel_file = "C:/Users/59666/Desktop/修改WTOPSIS/supply标签/属性权重_alpha_0.5_lambda_0.01.xlsx"
df = pd.read_excel(excel_file)

num_trials = 1000  # 每个比例下的随机采样次数
def wtopsis(G):
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
    wtopsis_scores = {}
    for i, node in enumerate(G.nodes()):
        wtopsis_scores[node] = s[i]

    return wtopsis_scores


wtopsis_scores=wtopsis(G)

# 将 wtopsis_scores 字典转换为 pandas Series

wtopsis_series = pd.Series(wtopsis_scores)

wtopsis_df = pd.DataFrame({
    "Node": wtopsis_series.index,
    "TOPSIS Scores": wtopsis_series.values
})

wtopsis_df.to_csv("C:/Users/59666/Desktop/修改WTOPSIS/新topsis_output.csv", index=False)

