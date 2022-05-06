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

ebd = [100,200,300,400,500,600,700,800,900]
F1 = [0.5771484375,0.595704057279236,0.603719599427754,0.616033755274261,0.621458430097538,0.626263897629536,0.622018348623853,0.620564149226569,0.618789521228545]
HammingLoss = [0.100394157199165,0.0981915140273591,0.0963366566195223,0.0949455135636447,0.0944817992116856,0.0928587989798284,0.0955251565035937,0.0966844423834917,0.0978437282633897]
plt.rc('font', family='Times New Roman', size='9')  # 设置字体样式、大小
fig = plt.figure()
ax = fig.add_subplot(111)

lns1 = ax.plot(ebd, F1, '-o', markersize=5, label = 'F1-score', color='#5C7AAC')
ax2 = ax.twinx()
lns2 = ax2.plot(ebd, HammingLoss, '-s', markersize=5, label = 'Hamming Loss', color='#E2A2B3')

# added these three lines
lns = lns1+lns2
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc=1)

ax.grid()
ax.set_xlabel("GCN 1st Layer Embedding Dimensions")
ax.set_ylabel(r"F1-score")
ax2.set_ylabel(r"Hamming Loss")
ax2.set_ylim(0.090, 0.102)
ax.set_ylim(0.57,0.63)
# plt.savefig('../results/gcn_dim.png') # 先show再save会保存空白图片
plt.savefig('../results/gcn_dim.jpeg', dpi=1000, format='jpeg')
plt.show()