# -*- coding: utf-8 -*-
# @Time : 2021/5/18 9:36
# @Author : JJun
# @Site : 
# @File : standardfile.py
# @Software: PyCharm

import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

'''
/data/dataset.txt indicates formula names, training/test split, formula labels. Each line is for a formula.
/data/corpus/dataset.txt contains raw herbs of each formula, each line is for the corresponding line in `/data/dataset.txt`
'''

from config import CONFIG
cfg = CONFIG()
dataset = cfg.dataset

# dataset_file = f'../data/origin/{dataset}.xlsx'
dataset_file = f'../data/origin/{dataset}.xlsx'

df = pd.read_excel(dataset_file, names=['num', 'name', 'labels', 'item', 'dosage'], dtype=str)

# shuffle
df = df.sample(frac=1.0).reset_index(drop=True)    #  数据打乱后行索引从0开始

print(df)

with open(f'../data/corpus/{dataset}.txt', 'w', encoding='utf8') as f:
    for line in df.item:
        f.write(''.join(line)+'\n')

with open(f'../data/corpus/{dataset}.clean.txt', 'w', encoding='utf8') as f:
    for line in df.item:
        f.write(''.join(line)+'\n')

with open(f'../data/corpus/{dataset}.dosage.txt', 'w', encoding='utf8') as f:
    for line in df.dosage:
        f.write(''.join(line)+'\n')

index_split = len(df) * 0.8

with open(f'../data/{dataset}.txt', 'w', encoding='utf8') as f:
    for index, row in df.iterrows():
        category = 'train' if index <= index_split else 'test'
        # single lable
        # f.write(f'{index}\t{category}\t{row[0].split()[sublabel]}\n')
        # multi_label
        name = '\t'.join(row[1].split())
        labels = '\t'.join(row[2].split())
        # num = '\t'.join(row[3].split())
        f.write(f"{name}\t{category}\t{labels}\n")

print(f'Dataset data/{dataset}.txt file is generated, please use herb_herb_graph!')