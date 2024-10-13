# -*- coding:utf-8 -*-
# author:ymh
# @Time : 2024/02/28 16:23
import networkx as nx
import pandas as pd


def cascade_failure_model(G, alpha, lambda_value):
    # 初始化节点
    num_nodes = G.number_of_nodes()
    degrees = dict(G.degree())

    # 计算节点初始负载
    initial_loads = []
    for node in G.nodes():
        k_i = degrees[node]
        sum_k_m = sum(degrees[neighbor] for neighbor in G.neighbors(node))
        initial_load = (k_i * sum_k_m) ** alpha
        initial_loads.append(initial_load)

    # 计算节点初始容量
    initial_capacities = [(1 + lambda_value) * load for load in initial_loads]

    # 记录节点的影响范围
    node_influence_range = {}

    def propagate_failure(node, loads, capacities, failed_nodes, max_distance=2):
        influenced_nodes = set()
        excess_load = loads[node - 1]

        # 如果节点的度为0，则跳过该节点
        if G.degree(node) == 0:
            return influenced_nodes

        # 平均分配负载给邻居节点
        neighbors = [neighbor for neighbor in G.neighbors(node) if neighbor not in failed_nodes]
        num_neighbors = len(neighbors)
        if num_neighbors > 0:
            distributed_load = excess_load / num_neighbors
            for neighbor in neighbors:
                loads[neighbor - 1] += distributed_load

            # 如果邻居节点的负载大于其容量且距离不超过 max_distance，则继续失效
            for neighbor in neighbors:
                if loads[neighbor - 1] > capacities[neighbor - 1]:
                    influenced_nodes.add(neighbor)
                    failed_nodes.add(neighbor)
                    # 递归调用处理新的失效节点，但仅限于 max_distance 范围内
                    if max_distance > 1:
                        new_influenced_nodes = propagate_failure(neighbor, loads, capacities, failed_nodes,
                                                                 max_distance - 1)
                        influenced_nodes.update(new_influenced_nodes)

        return influenced_nodes

    for node in range(1, num_nodes + 1):  # 节点从 1 开始编号
        # 初始化失效节点和邻居节点
        failed_nodes = set()
        influenced_nodes = set()

        # 复制初始负载和容量
        loads = initial_loads.copy()
        capacities = initial_capacities.copy()

        failed_nodes.add(node)
        influenced_nodes.add(node)

        # 递归调用处理失效节点
        influenced_nodes.update(propagate_failure(node, loads, capacities, failed_nodes, max_distance=2))

        # 记录失效节点的影响范围，至少为1（节点自身）
        node_influence_range[node] = len(influenced_nodes) - 1

    return node_influence_range

def load_graph(path):
    """根据边连边读取网络
    Parameters:
        path:网络存放的路径
    return:
        G:读取后的网络
    """
    G = nx.read_edgelist(path, nodetype=int, create_using=nx.Graph())
    return G


# 示例用法
alpha = 0.5
lambda_value = 0.01    # lambda 值

# 加载网络
G = load_graph("C:/Users/59666/Desktop/TOPSIS/jazz标签/jazz.txt")


influence_range = cascade_failure_model(G, alpha,lambda_value)
# 创建DataFrame对象
df = pd.DataFrame({"Node": range(1, len(influence_range) + 1), "Influence Range": influence_range.values()})

# 保存到CSV文件
df.to_csv("C:/Users/59666/Desktop/TOPSIS/jazz标签/influence_range3.csv", index=False)


# import os
# import networkx as nx
# import pandas as pd
#
#
# def cascade_failure_model(G, alpha, lambda_value):
#     num_nodes = G.number_of_nodes()
#     degrees = dict(G.degree())
#
#     # 计算节点初始负载
#     initial_loads = []
#     for node in G.nodes():
#         k_i = degrees[node]
#         sum_k_m = sum(degrees[neighbor] for neighbor in G.neighbors(node))
#         initial_load = (k_i * sum_k_m) ** alpha
#         initial_loads.append(initial_load)
#
#     # 计算节点初始容量
#     initial_capacities = [(1 + lambda_value) * load for load in initial_loads]
#
#     node_influence_range = {}
#
#     def propagate_failure(node, loads, capacities, failed_nodes, max_distance=2):
#         influenced_nodes = set()
#         excess_load = loads[node - 1]
#
#         if G.degree(node) == 0:
#             return influenced_nodes
#
#         neighbors = [neighbor for neighbor in G.neighbors(node) if neighbor not in failed_nodes]
#         num_neighbors = len(neighbors)
#         if num_neighbors > 0:
#             distributed_load = excess_load / num_neighbors
#             for neighbor in neighbors:
#                 loads[neighbor - 1] += distributed_load
#
#             for neighbor in neighbors:
#                 if loads[neighbor - 1] > capacities[neighbor - 1]:
#                     influenced_nodes.add(neighbor)
#                     failed_nodes.add(neighbor)
#                     if max_distance > 1:
#                         new_influenced_nodes = propagate_failure(neighbor, loads, capacities, failed_nodes,
#                                                                  max_distance - 1)
#                         influenced_nodes.update(new_influenced_nodes)
#
#         return influenced_nodes
#
#     for node in range(1, num_nodes + 1):
#         failed_nodes = set()
#         influenced_nodes = set()
#
#         loads = initial_loads.copy()
#         capacities = initial_capacities.copy()
#
#         failed_nodes.add(node)
#         influenced_nodes.add(node)
#
#         influenced_nodes.update(propagate_failure(node, loads, capacities, failed_nodes, max_distance=2))
#
#         node_influence_range[node] = len(influenced_nodes) - 1
#
#     return node_influence_range
#
#
# def load_graph(path):
#     G = nx.read_edgelist(path, nodetype=int, create_using=nx.Graph())
#     return G
#
#
# def save_results(influence_range, folder, alpha, lambda_value):
#     df = pd.DataFrame({"Node": range(1, len(influence_range) + 1), "Influence Range": influence_range.values()})
#     filename = f"influence_range_alpha_{alpha}_lambda_{lambda_value}.csv"
#     filepath = os.path.join(folder, filename)
#     df.to_csv(filepath, index=False)
#     print(f"Saved results to {filepath}")
#
#
# # 主程序
# alphas = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # 变化的 α 值
# lambdas = [0.01, 0.02, 0.03, 0.04, 0.05]  # 变化的 λ 值
# network_paths = [
#     "C:/Users/59666/Desktop/节点重要性代码/supply标签/企业供应链网络.txt",
#     "C:/Users/59666/Desktop/节点重要性代码/Jazz标签/jazz.txt",
#     "C:/Users/59666/Desktop/节点重要性代码/Email标签/email.txt",
#     "C:/Users/59666/Desktop/节点重要性代码/wiki-vote标签/wiki-vote.txt",
#     "C:/Users/59666/Desktop/节点重要性代码/USAir97标签/USAir.txt",
#     "C:/Users/59666/Desktop/节点重要性代码/FaceBook标签/FaceBook.txt",
#     "C:/Users/59666/Desktop/TOPSIS/Road标签/Road.txt",
#     "C:/Users/59666/Desktop/TOPSIS/Infect标签/infect.txt",
#     "C:/Users/59666/Desktop/TOPSIS/Message标签/message.txt"]  # 你的网络文件路径列表
# output_folder = "C:/Users/59666/Desktop/级联失效/results"  # 输出文件夹路径
#
# if not os.path.exists(output_folder):
#     os.makedirs(output_folder)
#
# # 加载网络并保存结果
# for network_path in network_paths:
#     G = load_graph(network_path)
#     network_name = os.path.splitext(os.path.basename(network_path))[0]
#     network_folder = os.path.join(output_folder, network_name)
#
#     if not os.path.exists(network_folder):
#         os.makedirs(network_folder)
#
#     # 变化 α 值
#     for alpha in alphas:
#         influence_range = cascade_failure_model(G, alpha, 0.01)
#         save_results(influence_range, network_folder, alpha, 0.01)
#
#     # 变化 λ 值
#     alpha_fixed = 0.5
#     for lambda_value in lambdas:
#         influence_range = cascade_failure_model(G, alpha_fixed, lambda_value)
#         save_results(influence_range, network_folder, alpha_fixed, lambda_value)
