# -*- coding: utf-8 -*-
# @Time : 2022/3/15 16:29
# @Author : JJun
# @Site : 
# @File : plt_dim_2y.py
# @Software: PyCharm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.rc('font', family='Times New Roman', size='10')

fig, ax1 = plt.subplots()

# 柱形的宽度
width = 0.3

f1 = [0.626263897629536,0.610646478356275,0.609302367322579,0.620101741136983]  # similarity of action
hammingloss = [0.0928587989798284,0.0951773707396243,0.0985392997913285,0.0963366566195223]  # js diversity
labels = ['Herb-sum\nFormula-sum', 'Herb-sum\nFormula-mean', 'Herb-mean\nFormula-sum', 'Herb-mean\nFormula-mean']

# 柱形的间隔
x1_list = []
x2_list = []
for i in range(len(f1)):
    x1_list.append(i)
    x2_list.append(i + width)

# 绘制柱形图1
# b1 = ax1.bar(x1_list, f1,width=width,label='f1 score',color = sns.xkcd_rgb["pale red"], align='edge',)
b1 = ax1.bar(x1_list, f1,width=width,label='F1-score',color = '#5C7AAC', align='edge')
# 绘制柱形图2---双Y轴
ax2 = ax1.twinx()
# b2 = ax2.bar(x2_list, hammingloss,width=width,label='hamming loss',color = sns.xkcd_rgb["denim blue"],align='edge', tick_label = labels)
b2 = ax2.bar(x2_list, hammingloss,width=width,label='Hamming Loss',color = '#E2A2B3',align='edge', tick_label = labels)
# 坐标轴标签设置
# ax1.set_title('资助金额字段缺失分析-学科',fontsize = 14)
# ax1.set_xlabel('学科',fontsize=12)
ax1.set_ylabel('F1-score', fontsize=10)
ax2.set_ylabel('Hamming Loss', fontsize=10)

# x轴标签旋转
# ax1.set_xticklabels(ax1.get_xticklabels(),rotation = 25)

# 双Y轴标签颜色设置
# ax1.yaxis.label.set_color(b1[0].get_facecolor())
# ax2.yaxis.label.set_color(b2[0].get_facecolor())

# 双Y轴刻度颜色设置
# ax1.tick_params(axis = 'y', colors = b1[0].get_facecolor())
# ax2.tick_params(axis = 'y', colors = b2[0].get_facecolor())

#双y轴刻度设置
ax2.set_ylim(0.091, 0.099)
ax1.set_ylim(0.59,0.63)

# 图例设置
plt.legend(handles = [b1,b2])

# 网格设置
plt.grid()
# plt.savefig("../results/fussion.png")
plt.savefig('../results/fussion.jpeg', dpi=1000, format='jpeg')
plt.show()