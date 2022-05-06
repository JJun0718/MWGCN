# -*- coding: utf-8 -*-
# @Time : 2022/3/4 16:55
# @Author : JJun
# @Site : 
# @File : check_num_dosage_herb.py
# @Software: PyCharm

import pandas as pd
import numpy as np

dataset_file = f'../data/origin/formulae.xlsx'

df = pd.read_excel(dataset_file, names=['num', 'name', 'labels', 'item', 'dosage'])
# print(df)
formula = df['num'].values.tolist()
herbs = df['item'].values.tolist()
dosage = df['dosage'].values.tolist()

need_change = []
for i in range(len(herbs)):
    list1 = herbs[i].split()
    list2 = dosage[i].split()
    if len(list1) != len(list2):
        print(formula[i])
        need_change.append(formula[i])

if len(need_change):
    print("A total of", len(need_change), "formulae need to be checked：\n", need_change)
else:
    print("No formulae need to be checked")