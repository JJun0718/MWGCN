# -*- coding: utf-8 -*-
# @Time : 2022/3/15 16:29
# @Author : JJun
# @Site : 
# @File : plt_dim_2y.py
# @Software: PyCharm

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('mathtext', default='regular')

time = [10,50,100,200,300,400,500]
F1 = [0.614252873563218,0.607379375591296,0.612587412587412,0.605666511843938,0.626263897629536,0.613023255813953,0.603488920320603]
HammingLoss = [0.0972640853234407,0.0962207280315325,0.0963366566195223,0.0984233712033387,0.0928587989798284,0.0964525852075121,0.0974959424994203]
plt.rc('font', family='Times New Roman', size='9')  # 设置字体样式、大小
fig = plt.figure()
ax = fig.add_subplot(111)

lns1 = ax.plot(time, F1, '-o', markersize=5, label = 'F1-score', color='#5C7AAC')
ax2 = ax.twinx()
lns2 = ax2.plot(time, HammingLoss, '-s', markersize=5, label = 'Hamming Loss', color='#E2A2B3')

# added these three lines
lns = lns1+lns2
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc=1)

ax.grid()
ax.set_xlabel("$\it{d}$")
ax.set_ylabel(r"F1-score")
ax2.set_ylabel(r"Hamming Loss")
ax2.set_ylim(0.090, 0.1012)
ax.set_ylim(0.60,0.628)
# plt.savefig('../results/dim.png') # 先show再save会保存空白图片

plt.show()