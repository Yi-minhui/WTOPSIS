# -*- coding:utf-8 -*-
# author:ymh
# @Time : 2023/12/06 10:30
import numpy as np
import pandas as pd

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
        failure_list.append(dict(zip(np.array(failure['Node'], dtype=str), failure['FailureSize'])))
    return failure_list