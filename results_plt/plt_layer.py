# -*- coding: utf-8 -*-
# @Time : 2022/3/16 10:58
# @Author : JJun
# @Site : 
# @File : plt_layer.py
# @Software: PyCharm

import matplotlib.pyplot as plt
import seaborn as sns
plt.rc('font', family='Times New Roman', size='10')

fig, ax1 = plt.subplots()

# 柱形的宽度
width = 0.3

f1 = [0.547917711991972,0.626263897629536,0.571960875640428]
hammingloss = [0.104451657778808,0.0928587989798284,0.106538372362624]
labels = ['1', '2', '3']

# 柱形的间隔
x1_list = []
x2_list = []
for i in range(len(f1)):
    x1_list.append(i)
    x2_list.append(i + width)

# 绘制柱形图1
# b1 = ax1.bar(x1_list, f1,width=width,label='f1 score',color = sns.xkcd_rgb["pale red"], align='edge',)
b1 = ax1.bar(x1_list, f1,width=width,label='F1-score',color='#5C7AAC', align='edge',)

# 绘制柱形图2---双Y轴
ax2 = ax1.twinx()
# b2 = ax2.bar(x2_list, hammingloss,width=width,label='hamming loss',color = sns.xkcd_rgb["denim blue"],align='edge', tick_label = labels)
b2 = ax2.bar(x2_list, hammingloss,width=width,label='Hamming Loss',color='#E2A2B3',align='edge', tick_label = labels)

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
ax2.set_ylim(0.09, 0.11)
ax1.set_ylim(0.48,0.64)

# 图例设置
plt.legend(handles = [b1,b2], loc=1)
plt.subplots_adjust(left=0.1, right=0.88)
# 网格设置
plt.grid()

# plt.savefig("../results/gcn_layer.png")
plt.savefig('../results/gcn_layer.jpeg', dpi=1000, format='jpeg')
plt.show()