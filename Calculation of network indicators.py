# -*- coding:utf-8 -*-
# author: ymh
# @Time : 2024/01/02 21:59

# import numpy as np
# import pandas as pd
# import networkx as nx
#
# network_files = [
#     # "C:/Users/59666/Desktop/节点重要性代码/supply标签/企业供应链网络.csv",
#     # "C:/Users/59666/Desktop/节点重要性代码/jazz标签/jazz.csv",
#     # "C:/Users/59666/Desktop/节点重要性代码/Email标签/email.csv",
#     # "C:/Users/59666/Desktop/节点重要性代码/USAir97标签/USAir.csv",
#     # "C:/Users/59666/Desktop/节点重要性代码/FaceBook标签/FaceBook.csv",
#     # "C:/Users/59666/Desktop/节点重要性代码/wiki-vote标签/wiki-vote.csv"
#     "C:/Users/59666/Desktop/TOPSIS/Road标签/Road.csv",
#     "C:/Users/59666/Desktop/TOPSIS/Infect标签/infect.csv",
#     "C:/Users/59666/Desktop/TOPSIS/Message标签/message.csv"
#
# ]
#
#
# results = []
# for file_path in network_files:
#     # 读取CSV文件
#     df = pd.read_csv(file_path)
#
#     # 创建图对象
#     G = nx.from_pandas_edgelist(df, 'FromNodeId', 'ToNodeId')
#
#     # 计算节点数和边数
#     num_nodes = G.number_of_nodes()
#     num_edges = G.number_of_edges()
#
#     # 计算平均度和最大度
#     degree_sequence = list(dict(G.degree()).values())
#     avg_degree = sum(degree_sequence) / num_nodes
#     max_degree = max(degree_sequence)
#
#     # 计算平均聚类系数
#     avg_clustering_coefficient = nx.average_clustering(G)
#
#     # 计算网络密度
#     density = nx.density(G)
#
#     # 计算网络传播阈值
#     dc_list = np.array(list(dict(G.degree()).values()))
#
#     beta = float(dc_list.mean()) / (float((dc_list ** 2).mean()) - float(dc_list.mean()))
#
#     # 将结果存入列表
#     results.append({
#         'Network': file_path,
#         'N': num_nodes,
#         'E': num_edges,
#         '<k>': avg_degree,
#         'kmax': max_degree,
#         'c': avg_clustering_coefficient,
#         'd': density,
#         # 'βc': beta
#     })
#
# # 将结果转为DataFrame
# results_df = pd.DataFrame(results)
#
# # 保存到CSV文件
# results_df.to_csv('network_metrics.csv', index=False)



import pandas as pd
import networkx as nx

# 读取网络数据
def read_network_from_csv(csv_file):
    df = pd.read_csv(csv_file)
    G = nx.from_pandas_edgelist(df, source='FromNodeId', target='ToNodeId', create_using=nx.Graph())
    return G

# 计算网络中所有节点对之间的平均最短路径长度
def calculate_average_distance(G):
    total_distance = 0
    num_pairs = 0
    for node in G.nodes():
        # 使用单源最短路径算法计算节点到其他所有节点的最短路径长度
        distances = nx.single_source_shortest_path_length(G, node)
        total_distance += sum(distances.values())
        num_pairs += len(distances) - 1
    average_distance = total_distance / num_pairs
    return average_distance

# 主函数
def main():
    # 读取网络
    csv_files = [
    # "C:/Users/59666/Desktop/节点重要性代码/supply标签/企业供应链网络.csv",
    # "C:/Users/59666/Desktop/节点重要性代码/jazz标签/jazz.csv",
    # "C:/Users/59666/Desktop/节点重要性代码/Email标签/email.csv",
    # "C:/Users/59666/Desktop/节点重要性代码/USAir97标签/USAir.csv",
    # "C:/Users/59666/Desktop/节点重要性代码/FaceBook标签/FaceBook.csv",
    # "C:/Users/59666/Desktop/节点重要性代码/wiki-vote标签/wiki-vote.csv"]
    # csv_files=["C:/Users/59666/Desktop/TOPSIS/WC-TOPSIS算法/风筝网络.csv"
        "C:/Users/59666/Desktop/TOPSIS/Road标签/Road.csv",
        "C:/Users/59666/Desktop/TOPSIS/Infect标签/infect.csv",
        "C:/Users/59666/Desktop/TOPSIS/Message标签/message.csv"
    ]
    for csv_file in csv_files:
        G = read_network_from_csv(csv_file)
        print("Network:", csv_file)
        # 计算平均最短路径长度
        average_distance = calculate_average_distance(G)
        print("Average distance:", average_distance)
        print()

if __name__ == "__main__":
    main()


