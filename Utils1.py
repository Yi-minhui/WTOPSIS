# -*- coding:utf-8 -*-
# author:ymh
# @Time : 2023/12/03 15:29
import numpy as np
import pandas as pd
import networkx as nx
import torch
import seaborn as sns
import random
from tqdm import tqdm

sns.set_style('ticks')

def setup_seed(seed):
    """固定种子"""
    torch.manual_seed(seed)
    #     torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def load_graph(path):
    """根据边连边读取网络
    Parameters:
        path:网络存放的路径
    return:
        G:读取后的网络
    """
    G = nx.read_edgelist(path, create_using=nx.Graph())
    return G


def load_failure_list(path):
    """
    读取不同beta情况模拟下的Failure Size的结果
    Parameters:
        path:存放Failure Size结果的根路径
    return:
        每个节点的Failure Size模拟结果
    """
    failure_list = []
    for i in range(10):
        failure = pd.read_csv(path + str(i) + '.csv')
        failure_list.append(dict(zip(np.array(failure['Node'], dtype=str), failure['Failure Size'])))
    return failure_list


def cascade_failure_model(G, initial_failure_nodes, parameter, load_parameter, beta=0.1, mu=1):
    """
    级联失效模型
    Input:
        G: 原始网络
        load_parameter: 负载参数
        initial_failure_nodes: 初始失效节点列表
        beta: 传播概率
        mu: 恢复概率
    Return:
        re: 模拟N次之后，该节点的平均失效规模
    """
    N = 1000
    re = 0

    while N > 0:
        current_load = {node: node_degree ** parameter for node, node_degree in dict(G.degree()).items()}
        capacity = {node: (1 + load_parameter) * current_load[node] for node in G.nodes()}

        failed_nodes = set(initial_failure_nodes)  # 初始失效的节点
        recovered_nodes = set()   # 恢复的节点

        while len(failed_nodes) != 0:
            new_failures = set()
            for node in failed_nodes:
                for neighbor in G.neighbors(node):
                    if current_load[node] > capacity[node]:
                        # 当节点负载超过容量时，将负载平均分配给相邻的节点
                        distribute_load = current_load[node] / len(list(G.neighbors(node)))
                        current_load[neighbor] += distribute_load
                        if current_load[neighbor] > capacity[neighbor]:
                            new_failures.add(neighbor)  # 邻居节点可能失效
                        # 超过容量的节点失效
                        new_failures.add(node)

                    k = random.uniform(0, 1)
                    if k < beta and neighbor not in failed_nodes and neighbor not in recovered_nodes:
                        new_failures.add(neighbor)

                k2 = random.uniform(0, 1)
                if k2 > mu:
                    new_failures.add(node)
                else:
                    recovered_nodes.add(node)

            failed_nodes = set(new_failures)

        re += len(recovered_nodes) + len(failed_nodes)
        N -= 1

    return re / 1000.0


def cascade_failure_dict(G,parameter, load_parameter, beta=0.1, mu=1, real_beta=None, a=1.5):
    """
    获得整个网络的所有节点的级联失效结果
    Input:
        G: 目标网络
        load_parameter: 负载参数
        beta: 传播概率
        mu: 恢复概率
        real_beta: 按公式计算的传播概率
    Return:
        cascade_failure_dic: 记录所有节点失效规模的字典
    """

    node_list = list(G.nodes())
    cascade_failure_dic = {}
    if real_beta:
        dc_list = np.array(list(dict(G.degree()).values()))
        beta = a*(float(dc_list.mean())/(float((dc_list**2).mean())-float(dc_list.mean())))

    print('beta:', beta)

    for node in tqdm(node_list):
        failure_size = cascade_failure_model(G, initial_failure_nodes=[node],parameter = parameter, load_parameter=load_parameter, beta=beta, mu=mu)
        cascade_failure_dic[node] = failure_size

    return cascade_failure_dic

def save_cascade_failure_dict(dic, path):
    """
    存放级联失效模型的结果
    Parameters:
        dic: 级联失效结果字典
        path: 目标存放路径
    """
    node = list(dic.keys())
    failure_size = list(dic.values())
    FailureSize = pd.DataFrame({'Node': node, 'FailureSize': failure_size})
    FailureSize.to_csv(path, index=False)


def cascade_failure_betas(G,parameter, load_parameter, a_list, root_path):
    """
    不同beta情况下的级联失效模型
    Parameters:
        G: 目标网络
        load_parameter: 负载参数
        a_list: 存放传播概率是传播阈值的多少倍的列表
        root_path: 存放结果的根路径
    """
    failure_list = []

    for idx, a in enumerate(a_list):
        failure_dict = cascade_failure_dict(G,parameter = parameter, load_parameter=load_parameter, real_beta=True, a=a)
        failure_list.append(failure_dict)
        path = root_path + str(idx) + '.csv'
        save_cascade_failure_dict(failure_dict, path)

    return failure_list
