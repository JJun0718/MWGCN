# -*- coding: utf-8 -*-
# @Time : 2022/3/4 8:53
# @Author : JJun
# @Site : 
# @File : formula_herb_graph.py
# @Software: PyCharm

import os
import random
import pandas as pd
import numpy as np
import pickle as pkl
# import networkx as nx
import scipy.sparse as sp
from math import log
from sklearn import svm
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import cosine
import re
import jieba

import sys
sys.path.append('../')
from utils.utils import loadHerb2Vec, get_synergy_dict, get_dosage_range, get_relative_dosage

from config import CONFIG
cfg = CONFIG()
dataset = cfg.dataset

herb_vector_file = '../data/corpus/' + dataset + '_herb_vectors.txt'
_, embd, herb_vector_map = loadHerb2Vec(herb_vector_file)
herb_embeddings_dim = len(embd[0])

herb_fussion = 1  # 1:sum  0: avg
formula_fussion = 1  # 1:sum 0:avg

# herb_embeddings_dim = 300
# herb_vector_map = {}

formula_name_list = []
formula_train_list = []
formula_test_list = []

with open('../data/' + dataset + '.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        formula_name_list.append(line.strip())
        temp = line.split("\t")
        if temp[1].find('test') != -1:
            formula_test_list.append(line.strip())
        elif temp[1].find('train') != -1:
            formula_train_list.append(line.strip())

formula_content_list = []
with open('../data/corpus/' + dataset + '.clean.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        formula_content_list.append(line.strip())

dosage_list = []
with open('../data/corpus/' + dataset + '.dosage.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        dosage_list.append(line.strip())

train_ids = []
for train_name in formula_train_list:
    train_id = formula_name_list.index(train_name)
    train_ids.append(train_id)
# print("train_ids:",train_ids)
random.shuffle(train_ids)

train_ids_str = '\n'.join(str(index) for index in train_ids)
with open('../data/' + dataset + '.train.index', 'w', encoding='utf8') as f:
    f.write(train_ids_str)


test_ids = []
for test_name in formula_test_list:
    test_id = formula_name_list.index(test_name)
    test_ids.append(test_id)
# print("test_ids:", test_ids)
random.shuffle(test_ids)

test_ids_str = '\n'.join(str(index) for index in test_ids)
with open('../data/' + dataset + '.test.index', 'w', encoding='utf8') as f:
    f.write(test_ids_str)

ids = train_ids + test_ids
print("ids", len(ids))
print('after', len(set(ids)))

shuffle_formula_name_list = []
shuffle_formula_herbs_list = []
shuffle_dosage_list = []

for id in ids:
    shuffle_formula_name_list.append(formula_name_list[int(id)])
    shuffle_formula_herbs_list.append(formula_content_list[int(id)])
    shuffle_dosage_list.append(dosage_list[int(id)])
shuffle_formula_name_str = '\n'.join(shuffle_formula_name_list)
shuffle_formula_herbs_str = '\n'.join(shuffle_formula_herbs_list)
shuffle_dosage_str = '\n'.join(shuffle_dosage_list)

with open('../data/' + dataset + '_shuffle.txt', 'w', encoding='utf8') as f:
    f.write(shuffle_formula_name_str)

with open('../data/corpus/' + dataset + '_shuffle.txt', 'w', encoding='utf8') as f:
    f.write(shuffle_formula_herbs_str)

with open('../data/corpus/' + dataset + '_dosage_shuffle.txt', 'w', encoding='utf8') as f:
    f.write(shuffle_dosage_str)

# build vocab
herb_freq = {}
herb_set = set()
for formula_herbs in shuffle_formula_herbs_list:
    herbs = formula_herbs.split()
    for herb in herbs:
        herb_set.add(herb)
        if herb in herb_freq:
            herb_freq[herb] += 1
        else:
            herb_freq[herb] = 1

vocab = list(herb_set)
vocab_size = len(vocab)

herb_formula_list = {}

for i in range(len(shuffle_formula_herbs_list)):
    formula_herbs = shuffle_formula_herbs_list[i]
    herbs = formula_herbs.split()
    appeared = set()
    for herb in herbs:
        if herb in appeared:
            continue
        if herb in herb_formula_list:
            formula_list = herb_formula_list[herb]
            formula_list.append(i)
            herb_formula_list[herb] = formula_list
        else:
            herb_formula_list[herb] = [i]
        appeared.add(herb)

herb_formula_freq = {}
for herb, formula_list in herb_formula_list.items():
    herb_formula_freq[herb] = len(formula_list)

herb_id_map = {}
for i in range(vocab_size):
    herb_id_map[vocab[i]] = i
# print(herb_id_map)

vocab_str = '\n'.join(vocab)

with open('../data/corpus/' + dataset + '_vocab.txt', 'w', encoding='utf-8') as f:
    f.write(vocab_str)

'''
Herb definitions begin
'''
'''
def get_describe():
    filename = f"../data/origin/herb_trait.xlsx"

    data = pd.read_excel(filename)

    all_vocab = data['名称'].tolist()
    describes = data['性状'].tolist()

    return all_vocab, describes

def get_stopwords():  #  Stopwords merge
    hit_file = 'E:/hit_stopwords.txt'
    hit_stopwords = [line.strip() for line in open(hit_file, encoding='UTF-8').readlines()]

    baidu_file = 'E:/baidu_stopwords.txt'
    baidu_stopwords = [line.strip() for line in open(baidu_file, encoding='UTF-8').readlines()]

    cn_file = 'E:/cn_stopwords.txt'
    cn_stopwords = [line.strip() for line in open(cn_file, encoding='UTF-8').readlines()]

    scu_file = 'E:/scu_stopwords.txt'
    scu_stopwords = [line.strip() for line in open(scu_file, encoding='UTF-8').readlines()]

    stopwords = set()
    for i in hit_stopwords:
        stopwords.add(i)
    for i in baidu_stopwords:
        stopwords.add(i)
    for i in cn_stopwords:
        stopwords.add(i)
    for i in scu_stopwords:
        stopwords.add(i)

    return stopwords

all_vocab, describes = get_describe()

origin_definitions = []

for describe in describes:
    describe = re.sub(r'[0-9a-zA-Z]+', '', describe)
    describe = jieba.lcut(describe.strip(), cut_all = False)  # 分词
    origin_definitions.append(describe)

definitions = []
for definition in origin_definitions:
    temp = ' '.join(definition)
    definitions.append(temp)

herb_def = {}
for i in range(len(all_vocab)):
    herb_def[all_vocab[i]] = definitions[i]

definitions = []
for herb in vocab:
    herb_defs = []
    syn_def = ''
    if herb in herb_def.keys():
        syn_def = herb_def[herb]
    herb_defs.append(syn_def)
    herb_des = ' '.join(herb_defs)
    if herb_des == '':
        herb_des = '<PAD>'
    definitions.append(herb_des)

string = '\n'.join(definitions)   # 将list转为str

ret = [i for i in vocab if i not in all_vocab]
if len(ret):
    print("性状待查：", ret)

#  存储词语定义
f = open('../data/corpus/' + dataset + '_vocab_def.txt', 'w', encoding="utf-8")
f.write(string)
f.close()

stopwords = get_stopwords()  # 获取停用词表
tfidf_vec = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", max_df=0.6, stop_words=stopwords, max_features=300)  # 实例化tf实例, max_features:词汇表长度, max_df/min_df:过滤出现在超过max_df/低于min_df比例的句子的词语
tfidf_matrix = tfidf_vec.fit_transform(definitions)    # 构建词汇表以及词项idf值，得到tf-idf矩阵（两个词语）
tfidf_matrix_array = tfidf_matrix.toarray()    # 格式转换
# print(tfidf_matrix_array[0], len(tfidf_matrix_array[0]))

# 异构图药向量
def get_herb_vector(vocab):
    herb_vectors = []    # 存储每个中药的特征表示

    for i in range(len(vocab)):
        herb = vocab[i]
        vector1 = tfidf_matrix_array[i]
        # 不要中药信息

        # vector1 = np.random.randn(herb_embeddings_dim)
        # vector1 = np.random.normal(size=300)
        # print("vector1:", vector1)

        str_vector = []

        for j in range(len(vector1)):
            str_vector.append(str(vector1[j]))
        temp = ' '.join(str_vector)
        herb_vector = herb + ' ' + temp
        herb_vectors.append(herb_vector)

    string = '\n'.join(herb_vectors)

    f = open('../data/corpus/' + dataset + '_herb_vectors.txt', 'w', encoding="utf-8")
    f.write(string)
    f.close()

get_herb_vector(vocab)
'''
'''
Herb definitions end
'''

fussion = True
if fussion:
    synergy_dict = get_synergy_dict()
    for i in range(len(vocab)):
        herb = vocab[i]
        vector_fh = herb_vector_map[herb]
        vector_hh = synergy_dict[herb]
        fussion_herb = []
        for j in range(herb_embeddings_dim):
            if herb_fussion:
                new_feature = vector_fh[j] + vector_hh[j]   # 药向量加和
            else:
                new_feature = (vector_fh[j] + vector_hh[j]) * 0.5   # 药向量平均
            fussion_herb.append(new_feature)
        herb_vector_map[herb] = fussion_herb

print("herb_embeddings_dim", herb_embeddings_dim)

'''
Herb confusion end
'''

# label list
label_set = set()
for formula_meta in shuffle_formula_name_list:
    temp = formula_meta.split('\t')
    for i in range(2, len(temp)):
        label_set.add(temp[i])
label_list = list(label_set)

label_list_str = '\n'.join(label_list)
with open('../data/corpus/' + dataset + '_labels.txt', 'w', encoding='utf8') as f:
    f.write(label_list_str)

# x: feature vectors of training formulae, no initial features
# slect 80% training set
train_size = len(train_ids)
val_size = int(0.2 * train_size)
real_train_size = train_size - val_size  # - int(0.5 * train_size)
# different training rates

real_train_formula_names = shuffle_formula_name_list[:real_train_size]
real_train_formula_names_str = '\n'.join(real_train_formula_names)

with open('../data/' + dataset + '.real_train.name', 'w', encoding='utf8') as f:
    f.write(real_train_formula_names_str)

row_x = []
col_x = []
data_x = []
for i in range(real_train_size):    # 生成训练集方向量，采用平均池化
    formula_vec = np.array([0.0 for k in range(herb_embeddings_dim)])
    formula_herbs = shuffle_formula_herbs_list[i]
    herbs = formula_herbs.split()
    formula_len = len(herbs)
    for herb in herbs:
        if herb in herb_vector_map:
            herb_vector = herb_vector_map[herb]
            # print(formula_vec)
            # print(np.array(herb_vector))
            formula_vec = formula_vec + np.array(herb_vector)

    for j in range(herb_embeddings_dim):
        row_x.append(i)
        col_x.append(j)
        # np.random.uniform(-0.25, 0.25)
        if formula_fussion:
            data_x.append(formula_vec[j]) # 方向量加和
        else:
            data_x.append(formula_vec[j] / formula_len)  # 方向量平均 formula_vec[j]/ formula_len

x = sp.csr_matrix((data_x, (row_x, col_x)), shape=(
    real_train_size, herb_embeddings_dim))

y = []
for i in range(real_train_size):    # 生成训练集标签
    formula_meta = shuffle_formula_name_list[i]
    temp = formula_meta.split('\t')
    multi_hot = [0 for l in range(len(label_list))]
    for label in temp[2:]:
        label_index = label_list.index(label)
        multi_hot[label_index] = 1
    y.append(multi_hot)
y = np.array(y)
# print("real_train_size labels:", y)

# tx: feature vectors of test formulae, no initial features
test_size = len(test_ids)
print("train_size:", train_size, "\nval_size:", val_size, "\nreal_train_size:", real_train_size, "\ntest_size", test_size)

row_tx = []
col_tx = []
data_tx = []
for i in range(test_size):  # 生成测试集方向量
    formula_vec = np.array([0.0 for k in range(herb_embeddings_dim)])
    formula_herbs = shuffle_formula_herbs_list[i + train_size]
    herbs = formula_herbs.split()
    formula_len = len(herbs)
    for herb in herbs:
        if herb in herb_vector_map:
            herb_vector = herb_vector_map[herb]
            formula_vec = formula_vec + np.array(herb_vector)

    for j in range(herb_embeddings_dim):
        row_tx.append(i)
        col_tx.append(j)
        # np.random.uniform(-0.25, 0.25)
        if formula_fussion:
            data_tx.append(formula_vec[j])  # 方向量加和
        else:
            data_tx.append(formula_vec[j] / formula_len)  # formula_vec[j] / formula_len

# tx = sp.csr_matrix((test_size, herb_embeddings_dim), dtype=np.float32)
tx = sp.csr_matrix((data_tx, (row_tx, col_tx)),
                   shape=(test_size, herb_embeddings_dim))

ty = []
for i in range(test_size):   # 生成测试集标签
    formula_meta = shuffle_formula_name_list[i + train_size]
    temp = formula_meta.split('\t')
    multi_hot = [0 for l in range(len(label_list))]
    for label in temp[2:]:
        label_index = label_list.index(label)
        multi_hot[label_index] = 1
    ty.append(multi_hot)
ty = np.array(ty)
# print("test lablels:", ty)

# allx: the the feature vectors of both labeled and unlabeled training instances
# (a superset of x)
# unlabeled training instances -> herbs

herb_vectors = np.random.uniform(-0.01, 0.01,
                                 (vocab_size, herb_embeddings_dim))

for i in range(len(vocab)):
    herb = vocab[i]
    if herb in herb_vector_map:
        vector = herb_vector_map[herb]
        herb_vectors[i] = vector

row_allx = []
col_allx = []
data_allx = []

for i in range(train_size):
    formula_vec = np.array([0.0 for k in range(herb_embeddings_dim)])
    formula_herbs = shuffle_formula_herbs_list[i]
    herbs = formula_herbs.split()
    formula_len = len(herbs)
    for herb in herbs:
        if herb in herb_vector_map:
            herb_vector = herb_vector_map[herb]
            formula_vec = formula_vec + np.array(herb_vector)

    for j in range(herb_embeddings_dim):
        row_allx.append(int(i))
        col_allx.append(j)
        # np.random.uniform(-0.25, 0.25)
        if formula_fussion:
            data_allx.append(formula_vec[j])  # formulae vector
        else:
            data_allx.append(formula_vec[j] / formula_len)  # formula_vec[j]/formula_len

for i in range(vocab_size):
    for j in range(herb_embeddings_dim):
        row_allx.append(int(i + train_size))
        col_allx.append(j)
        data_allx.append(herb_vectors.item((i, j)))

row_allx = np.array(row_allx)
col_allx = np.array(col_allx)
data_allx = np.array(data_allx)

allx = sp.csr_matrix(
    (data_allx, (row_allx, col_allx)), shape=(train_size + vocab_size, herb_embeddings_dim))

ally = []
for i in range(train_size):
    formula_meta = shuffle_formula_name_list[i]
    temp = formula_meta.split('\t')
    multi_hot = [0 for l in range(len(label_list))]
    for label in temp[2:]:
        label_index = label_list.index(label)
        multi_hot[label_index] = 1
    ally.append(multi_hot)

for i in range(vocab_size):
    one_hot = [0 for l in range(len(label_list))]
    ally.append(one_hot)

ally = np.array(ally)

print(x.shape, y.shape, tx.shape, ty.shape, allx.shape, ally.shape)

'''
Formula herb heterogeneous graph
'''

row = []
col = []
weight = []

# relative dosage as formula-herb edge weights
dosage_range_dict = get_dosage_range()

for i in range(len(shuffle_formula_herbs_list)):
    formula_herbs = shuffle_formula_herbs_list[i]
    herbs = formula_herbs.split()
    formula_herb_set = set()

    formula_dosages = shuffle_dosage_list[i]  # use dosage
    dosages = formula_dosages.split()

    for j in range(len(herbs)):
        herb = herbs[j]
        # if herb in formula_herb_set:
        #     continue

        use_dosage = float(dosages[j])
        dosage_range_list = dosage_range_dict[herbs[j]]
        min_dosage = dosage_range_list[0]
        max_dosage = dosage_range_list[1]
        relative_dosage = get_relative_dosage(min_dosage, max_dosage, use_dosage)
        # print(herb, min_dosage, max_dosage, use_dosage, relative_dosage)

        j = herb_id_map[herb]
        if i < train_size:
            row.append(i)
        else:
            row.append(i + vocab_size)
        col.append(train_size + j)
        weight.append(relative_dosage)
        formula_herb_set.add(herb)

node_size = train_size + vocab_size + test_size
adj = sp.csr_matrix(
    (weight, (row, col)), shape=(node_size, node_size))

# dump objects
with open("../data/ind.{}.x".format(dataset), 'wb') as f:
    pkl.dump(x, f)

with open("../data/ind.{}.y".format(dataset), 'wb') as f:
    pkl.dump(y, f)

with open("../data/ind.{}.tx".format(dataset), 'wb') as f:
    pkl.dump(tx, f)

with open("../data/ind.{}.ty".format(dataset), 'wb') as f:
    pkl.dump(ty, f)

with open("../data/ind.{}.allx".format(dataset), 'wb') as f:
    pkl.dump(allx, f)

with open("../data/ind.{}.ally".format(dataset), 'wb') as f:
    pkl.dump(ally, f)

with open("../data/ind.{}.adj".format(dataset), 'wb') as f:
    pkl.dump(adj, f)

print(f'formual-herb is built, please use train!')