import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import os
import csv
import failure
import community.community_louvain as community
import warnings

warnings.filterwarnings('ignore')
# 代码将警告信息设置为不显示，这意味着在代码执行期间，警告消息将被忽略，不会在控制台上显示。
# 这在某些情况下可以使输出更清晰，但需要注意，如果存在潜在问题，这些问题可能不会被及时发现。
sns.set_style('ticks')
# 这一行代码设置了Seaborn库的绘图风格。Seaborn是一个用于数据可视化的Python库，它提供了不同的绘图风格。
# 通过sns.set_style('ticks')，代码选择了一种名为"ticks"的绘图风格。
# 这个风格通常包括坐标轴上的刻度线和坐标轴周围的小刻度线，以帮助更好地显示数据。这可以使图表看起来更加美观和易于理解。
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# 这一行代码设置了一个环境变量。这个环境变量名为KMP_DUPLICATE_LIB_OK，其值设置为'TRUE'。
# 这个环境变量通常与Intel Math Kernel Library (MKL)相关，MKL是一种数学库，用于提高数值计算的性能。
# 设置KMP_DUPLICATE_LIB_OK为'TRUE'可以允许多个Python进程共享MKL库的实例，从而减少内存占用。
# 这个设置通常在使用一些需要MKL的科学计算库时会出现，以避免一些潜在的问题。
# In[6] Experiment 4: Node ranking similarity and discrimination ability
# 这段代码表示一个实验，其目的是评估节点排名的相似性和区分能力。


def load_graph(path):
    """根据边连边读取网络
    Parameters:
        path:网络存放的路径
    return:
        G:读取后的网络
    """
    G = nx.read_edgelist(path, create_using=nx.Graph())
    return G

def csv_to_dict(filename):
    with open(filename, 'r') as file:
        reader = csv.DictReader(file)
        data = {}
        for row in reader:
            node = row['Node']
            centrality = float(row['TOPSIS Scores'])
            data[node] = centrality
    return data

def csv_to_dict1(filename):
    with open(filename, 'r') as file:
        reader = csv.DictReader(file)
        data = {}
        for row in reader:
            node = row['Node']
            centrality = float(row['Influence Range'])
            data[node] = centrality
    return data


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


def calculate_similarity(G,cascading_dic,topsis_dict,ws_topsis_dict):
    """使用肯德尔相关系数对比不同方法
    Parameters:
        G:目标网络
        failure_dic: FAILURE值字典
        topsis_dict: TOPSIS值字典
        ws_topsis_dict: WS_TOPSIS值字典
    return:
        result_pd: 包含FAILURE、TOPSIS和WS_TOPSIS列的Pandas DataFrame
    """

    # 从目标网络G中获取所有节点，并将它们存储在列表nodes中
    nodes = list(G.nodes())

    # 初始化三个空列表，用于存储每个节点的TOPSIS、WS_TOPSIS和FAILURE值
    topsis_tau_list = []
    ws_topsis_tau_list = []
    cascading_list = []

    for node in nodes:
        # 从topsis_dict和ws_topsis_dict获取节点对应的值
        topsis_tau_list.append(topsis_dict[node])
        ws_topsis_tau_list.append(ws_topsis_dict[node])
        # 从failure_dic获取FAILURE值
        cascading_list.append(cascading_dic[node])
    # 使用Pandas创建一个DataFrame，包含三列：'FAILURE'、'TOPSIS'和'WS_TOPSIS'
    result_pd = pd.DataFrame({'FAILURE':cascading_list,'TOPSIS': topsis_tau_list,'WC_TOPSIS':ws_topsis_tau_list})
    return result_pd

def normalization(data):
    data_norm = (data-data.min(axis=0))/(data.max(axis=0)-data.min(axis=0))
    return data_norm
# normalization 函数：对数据进行归一化处理。


supply=load_graph("C:/Users/59666/Desktop/节点重要性代码/supply标签/企业供应链网络.txt")
jazz=load_graph("C:/Users/59666/Desktop/节点重要性代码/jazz标签/jazz.txt")
Email=load_graph("C:/Users/59666/Desktop/节点重要性代码/Email标签/email.txt")
USAir=load_graph("C:/Users/59666/Desktop/节点重要性代码/USAir97标签/USAir.txt")
FaceBook=load_graph("C:/Users/59666/Desktop/节点重要性代码/FaceBook标签/FaceBook.txt")
Vote=load_graph("C:/Users/59666/Desktop/节点重要性代码/wiki-vote标签/wiki-vote.txt")
Road=load_graph("C:/Users/59666/Desktop/TOPSIS/Road标签/Road.txt")
Infect=load_graph("C:/Users/59666/Desktop/TOPSIS/Infect标签/infect.txt")
Message=load_graph("C:/Users/59666/Desktop/TOPSIS/Message标签/message.txt")


supply.remove_edges_from(nx.selfloop_edges(supply))
jazz.remove_edges_from(nx.selfloop_edges(jazz))
Email.remove_edges_from(nx.selfloop_edges(Email))
USAir.remove_edges_from(nx.selfloop_edges(USAir))
FaceBook.remove_edges_from(nx.selfloop_edges(FaceBook))
Vote.remove_edges_from(nx.selfloop_edges(Vote))
Road.remove_edges_from(nx.selfloop_edges(Road))
Infect.remove_edges_from(nx.selfloop_edges(Infect))
Message.remove_edges_from(nx.selfloop_edges(Message))


supply_failure = csv_to_dict1("C:/Users/59666/Desktop/不同参数下级联失效得分/results/企业供应链网络/influence_range_alpha_0.5_lambda_0.01.csv")
jazz_failure = csv_to_dict1("C:/Users/59666/Desktop/不同参数下级联失效得分/results/jazz/influence_range_alpha_0.5_lambda_0.01.csv")
Email_failure = csv_to_dict1("C:/Users/59666/Desktop/不同参数下级联失效得分/results/email/influence_range_alpha_0.5_lambda_0.01.csv")
USAir_failure = csv_to_dict1("C:/Users/59666/Desktop/不同参数下级联失效得分/results/USAir/influence_range_alpha_0.5_lambda_0.01.csv")
FaceBook_failure = csv_to_dict1("C:/Users/59666/Desktop/不同参数下级联失效得分/results/FaceBook/influence_range_alpha_0.5_lambda_0.01.csv")
Vote_failure = csv_to_dict1("C:/Users/59666/Desktop/不同参数下级联失效得分/results/wiki-vote/influence_range_alpha_0.5_lambda_0.01.csv")
Road_failure = csv_to_dict1("C:/Users/59666/Desktop/不同参数下级联失效得分/results/Road/influence_range_alpha_0.5_lambda_0.01.csv")
Infect_failure = csv_to_dict1("C:/Users/59666/Desktop/不同参数下级联失效得分/results/infect/influence_range_alpha_0.5_lambda_0.01.csv")
Message_failure = csv_to_dict1("C:/Users/59666/Desktop/不同参数下级联失效得分/results/message/influence_range_alpha_0.5_lambda_0.01.csv")


supply_topsis = csv_to_dict('C:/Users/59666/Desktop/修改WTOPSIS/supply标签/supply_原topsis_output.csv')
jazz_topsis = csv_to_dict('C:/Users/59666/Desktop/修改WTOPSIS/Jazz标签/jazz_原topsis_output.csv')
Email_topsis = csv_to_dict('C:/Users/59666/Desktop/修改WTOPSIS/Email标签/email_原topsis_output.csv')
USAir_topsis = csv_to_dict('C:/Users/59666/Desktop/修改WTOPSIS/USAir97标签/USAir_原topsis_output.csv')
FaceBook_topsis = csv_to_dict('C:/Users/59666/Desktop/修改WTOPSIS/FaceBook标签/FaceBook_原topsis_output.csv')
Vote_topsis = csv_to_dict('C:/Users/59666/Desktop/修改WTOPSIS/wiki-vote标签/wiki-vote_原topsis_output.csv')
Road_topsis = csv_to_dict('C:/Users/59666/Desktop/修改WTOPSIS/Road标签/Road_原topsis_output.csv')
Infect_topsis = csv_to_dict('C:/Users/59666/Desktop/修改WTOPSIS/Infect标签/infect_原topsis_output.csv')
Message_topsis = csv_to_dict('C:/Users/59666/Desktop/修改WTOPSIS/Message标签/message_原topsis_output.csv')

supply_ws_topsis = csv_to_dict('C:/Users/59666/Desktop/修改WTOPSIS/supply标签/新topsis_output.csv')
jazz_ws_topsis = csv_to_dict('C:/Users/59666/Desktop/修改WTOPSIS/Jazz标签/新topsis_output.csv')
Email_ws_topsis = csv_to_dict('C:/Users/59666/Desktop/修改WTOPSIS/Email标签/新topsis_output.csv')
USAir_ws_topsis = csv_to_dict('C:/Users/59666/Desktop/修改WTOPSIS/USAir97标签/新topsis_output.csv')
FaceBook_ws_topsis = csv_to_dict('C:/Users/59666/Desktop/修改WTOPSIS/FaceBook标签/新topsis_output.csv')
Vote_ws_topsis = csv_to_dict('C:/Users/59666/Desktop/修改WTOPSIS/wiki-vote标签/新topsis_output.csv')
Road_ws_topsis = csv_to_dict('C:/Users/59666/Desktop/修改WTOPSIS/Road标签/新topsis_output.csv')
Infect_ws_topsis = csv_to_dict('C:/Users/59666/Desktop/修改WTOPSIS/Infect标签/新topsis_output.csv')
Message_ws_topsis = csv_to_dict('C:/Users/59666/Desktop/修改WTOPSIS/Message标签/新topsis_output.csv')


supply_sim = calculate_similarity(supply,supply_failure,supply_topsis, supply_ws_topsis)
jazz_sim = calculate_similarity(jazz,jazz_failure,jazz_topsis,jazz_ws_topsis)
Email_sim = calculate_similarity(Email,Email_failure,Email_topsis, Email_ws_topsis)
USAir_sim = calculate_similarity(USAir,USAir_failure,USAir_topsis,USAir_ws_topsis)
FaceBook_sim = calculate_similarity(FaceBook,FaceBook_failure,FaceBook_topsis,FaceBook_ws_topsis)
Vote_sim = calculate_similarity(Vote,Vote_failure,Vote_topsis,Vote_ws_topsis)
Road_sim = calculate_similarity(Road,Road_failure,Road_topsis,Road_ws_topsis)
Infect_sim = calculate_similarity(Infect,Infect_failure,Infect_topsis,Infect_ws_topsis)
Message_sim = calculate_similarity(Message,Message_failure,Message_topsis,Message_ws_topsis)


# normalize

supply_sim = normalization(supply_sim)
jazz_sim = normalization(jazz_sim)
Email_sim = normalization(Email_sim)
USAir_sim = normalization(USAir_sim)
FaceBook_sim = normalization(FaceBook_sim)
Vote_sim = normalization(Vote_sim)
Road_sim = normalization(Road_sim)
Infect_sim = normalization(Infect_sim)
Message_sim = normalization(Message_sim)


# visualize results
plt.figure(figsize=(17, 10), dpi=120)
plt.subplot(331)
S1 = plt.scatter(supply_sim['WC_TOPSIS'], supply_sim['TOPSIS'], c=supply_sim['FAILURE'], cmap='jet')
plt.plot(np.arange(-0.1, 1.1, 0.1), np.arange(-0.1, 1.1, 0.1), color='black', linewidth=5, linestyle='--')
plt.title('Supply', fontsize=20, fontweight='bold')
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.ylabel('TOPSIS', fontsize=20, fontweight='bold')
plt.text(-0.1, 1.05, '(a)', fontsize=20, fontweight='bold')
plt.colorbar(S1)

plt.subplot(332)
S2 = plt.scatter(jazz_sim['WC_TOPSIS'], jazz_sim['TOPSIS'], c=jazz_sim['FAILURE'], cmap='jet')
plt.plot(np.arange(-0.1, 1.1, 0.1), np.arange(-0.1, 1.1, 0.1), color='black', linewidth=5, linestyle='--')
plt.title('Jazz', fontsize=20, fontweight='bold')
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.text(-0.1, 1.05, '(b)', fontsize=20, fontweight='bold')
plt.colorbar(S2)

plt.subplot(333)
S3 = plt.scatter(Email_sim['WC_TOPSIS'], Email_sim['TOPSIS'], c=Email_sim['FAILURE'], cmap='jet')
plt.plot(np.arange(-0.1, 1.1, 0.1), np.arange(-0.1, 1.1, 0.1), color='black', linewidth=5, linestyle='--')
plt.title('Email', fontsize=20, fontweight='bold')
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.text(-0.1, 1.05, '(c)', fontsize=20, fontweight='bold')
plt.colorbar(S3)

plt.subplot(334)
S4 = plt.scatter(USAir_sim['WC_TOPSIS'], USAir_sim['TOPSIS'], c=USAir_sim['FAILURE'], cmap='jet')
plt.plot(np.arange(-0.1, 1.1, 0.1), np.arange(-0.1, 1.1, 0.1), color='black', linewidth=5, linestyle='--')
plt.title('USAir', fontsize=20, fontweight='bold')
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.text(-0.1, 1.05, '(c)', fontsize=20, fontweight='bold')
plt.ylabel('TOPSIS', fontsize=20, fontweight='bold')
# plt.xlabel('WTOPSIS', fontsize=20, fontweight='bold')
plt.colorbar(S4)
#
plt.subplot(335)
S5 = plt.scatter(FaceBook_sim['WC_TOPSIS'], FaceBook_sim['TOPSIS'], c=FaceBook_sim['FAILURE'], cmap='jet')
plt.plot(np.arange(-0.1, 1.1, 0.1), np.arange(-0.1, 1.1, 0.1), color='black', linewidth=5, linestyle='--')
plt.title('FaceBook', fontsize=20, fontweight='bold')
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
# plt.xlabel('WTOPSIS', fontsize=20, fontweight='bold')
plt.text(-0.1, 1.05, '(d)', fontsize=20, fontweight='bold')
plt.colorbar(S5)

plt.subplot(336)
S6 = plt.scatter(Vote_sim['WC_TOPSIS'], Vote_sim['TOPSIS'], c=Vote_sim['FAILURE'], cmap='jet')
plt.plot(np.arange(-0.1, 1.1, 0.1), np.arange(-0.1, 1.1, 0.1), color='black', linewidth=5, linestyle='--')
plt.title('Vote', fontsize=20, fontweight='bold')
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
# plt.xlabel('WTOPSIS', fontsize=20, fontweight='bold')
plt.text(-0.1, 1.05, '(f)', fontsize=20, fontweight='bold')
plt.colorbar(S6)

plt.subplot(337)
S7 = plt.scatter(Road_sim['WC_TOPSIS'], Road_sim['TOPSIS'], c=Road_sim['FAILURE'], cmap='jet')
plt.plot(np.arange(-0.1, 1.1, 0.1), np.arange(-0.1, 1.1, 0.1), color='black', linewidth=5, linestyle='--')
plt.title('Road', fontsize=20, fontweight='bold')
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlabel('WTOPSIS', fontsize=20, fontweight='bold')
plt.ylabel('TOPSIS', fontsize=20, fontweight='bold')
plt.text(-0.1, 1.05, '(g)', fontsize=20, fontweight='bold')
plt.colorbar(S7)

plt.subplot(338)
S8 = plt.scatter(Infect_sim['WC_TOPSIS'], Infect_sim['TOPSIS'], c=Infect_sim['FAILURE'], cmap='jet')
plt.plot(np.arange(-0.1, 1.1, 0.1), np.arange(-0.1, 1.1, 0.1), color='black', linewidth=5, linestyle='--')
plt.title('Infect', fontsize=20, fontweight='bold')
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlabel('WTOPSIS', fontsize=20, fontweight='bold')
plt.text(-0.1, 1.05, '(h)', fontsize=20, fontweight='bold')
plt.colorbar(S8)

plt.subplot(339)
S9 = plt.scatter(Message_sim['WC_TOPSIS'], Message_sim['TOPSIS'], c=Message_sim['FAILURE'], cmap='jet')
plt.plot(np.arange(-0.1, 1.1, 0.1), np.arange(-0.1, 1.1, 0.1), color='black', linewidth=5, linestyle='--')
plt.title('Message', fontsize=20, fontweight='bold')
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlabel('WTOPSIS', fontsize=20, fontweight='bold')
plt.text(-0.1, 1.05, '(i)', fontsize=20, fontweight='bold')
plt.colorbar(S9)


plt.tight_layout()
# plt.subplots_adjust(bottom=-0.3)
plt.savefig("C:/Users/59666/Desktop/修改WTOPSIS/肯德尔系数相关性.png", dpi=600)
plt.show()
