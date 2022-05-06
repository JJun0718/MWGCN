# -*- coding: utf-8 -*-
# @Time : 2022/3/3 15:25
# @Author : JJun
# @Site : 构建包含药物节点的同构图
# @File : herb_herb_graph.py
# @Software: PyCharm

import pandas as pd
import numpy as np
from Data_Process.apriori import apriori
from node2vec import Node2Vec
import networkx as nx
import matplotlib.pyplot as plt

from config import CONFIG
cfg = CONFIG()
dataset = cfg.dataset

def get_formulae(dataset):
    formulae = []
    with open(f"../data/corpus/{dataset}.clean.txt", "r", encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip('\n')
            formulae.append(line)

    herb = set()
    for i in range(len(formulae)):
        herbs = formulae[i].split()
        for j in herbs:
            herb.add(j)
    herb = list(herb)
    herb.sort()

    herb_dict = {}
    for i in range(len(herb)):
        herb_dict[herb[i]] = i

    all_formula = []
    for i in range(len(formulae)):
        formula = []
        herbs = formulae[i].split(" ")
        for j in herbs:
            formula.append(herb_dict[j])
        all_formula.append(formula)

    df = pd.DataFrame.from_dict(herb_dict, orient='index', columns=['num'])
    df = df.reset_index().rename(columns={'index':'herb'})
    df.to_excel('../results/Herb2Num.xlsx', index=False)
    print("中药字典已生成!")

    return all_formula, herb_dict

def save_edge(L, freq_data):
    edge = []

    freq_2 = list(L[1])
    for i in freq_2:
        lst = []
        for j in i:
            lst.append(j)
        lst.append(freq_data[i])
        edge.append(lst)

    herb_edge = pd.DataFrame(edge, columns=['x', 'y', 'weight'])
    # print(edge)
    herb_edge.to_excel("../results/herb_edge.xlsx", index=False)
    print("中药同构图边生成成功！")
    return edge

def get_herb_embedding(edge, herb_dict):
    herb_num = []
    for k, v in herb_dict.items():
        herb_num.append(v)

    G = nx.Graph()  # 建立一个空的无向图G
    # 无权无向图
    # G.add_edge(2,3)  # 添加一条边2-3（隐含着添加了两个节点2、3）

    # 建立带权无向图
    for i in range(len(edge)):
        x = edge[i][0]
        y = edge[i][1]
        weight = edge[i][2]
        G.add_weighted_edges_from([(x, y, weight)])

    # G.add_weighted_edges_from([(2, 3, 3.0), (3, 4, 3.5), (3, 5, 7.0)])  # 对于无向图，边3-2与边2-3被认为是一条边
    # G.add_weighted_edges_from([(3, 6, 8.9)])

    # print("weight from 2 to 3", G.get_edge_data(2, 3))
    # print("weight from 3 to 4", G.get_edge_data(3, 4))
    # print("weight from 3 to 5", G.get_edge_data(3, 5))

    # # 画图
    # nx.draw(G, with_labels=True)
    # plt.savefig("../results/Herb Synergy Graph.png")
    # plt.show()

    node2vec = Node2Vec(G, dimensions=300, walk_length=200, num_walks=100, p=1.1, q=1.2, weight_key='weight', workers=4)
    model = node2vec.fit()  # 训练

    nodes = G.nodes
    node_embedding = []
    herb_synergy_num = []
    for i in nodes:
        embedding = []
        herb_synergy_num.append(i)
        embedding.append(i)
        ebd = model.wv[str(i)]         #  得到节点i的embedding
        for j in ebd:
            embedding.append(j)
        node_embedding.append(embedding)

    ret = [i for i in herb_num if i not in herb_synergy_num]
    for i in ret:
        embedding = []
        embedding.append(i)
        list1 = np.zeros(300, dtype=float)
        for j in list1:
            embedding.append(j)
        node_embedding.append(embedding)

    # 将所有节点embedding存excel
    herb_synergy_embedding = pd.DataFrame(node_embedding)
    herb_synergy_embedding.to_excel('../results/herb_synergy_embedding.xlsx', index=False)
    print("同构图中药节点embedding文件已生成！")
    return node_embedding

formulae, herb_dict = get_formulae(dataset)
L, freq_data = apriori(formulae, min_freq=3, max_len=2)
edge = save_edge(L, freq_data)
node_embedding = get_herb_embedding(edge, herb_dict)

print(f'Dataset results/herb_synergy_embedding.xlsx file is generated, please use formulae_herb_graph!')