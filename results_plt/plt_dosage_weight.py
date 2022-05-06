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

ebd = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
F1 = [0.612948299953423,0.62368541380887,0.624826949700046,0.621072088724584,0.617249417249417,0.615029967727063,0.613159122725151,0.603024574669187,0.604543347241539]
HammingLoss = [0.0963366566195223,0.095409227915604,0.094249942035706,0.0950614421516345,0.0951773707396243,0.0968003709714815,0.0961047994435427,0.0973800139114305,0.0988870855552979]
plt.rc('font', family='Times New Roman', size='9')  # 设置字体样式、大小
fig = plt.figure()
ax = fig.add_subplot(111)

lns1 = ax.plot(ebd, F1, '-o', label = 'f-1 score')
ax2 = ax.twinx()
lns2 = ax2.plot(ebd, HammingLoss, '-rs', label = 'hamming loss')

# added these three lines
lns = lns1+lns2
labs = [l.get_label() for l in lns]


ax.grid()
ax.set_xlabel("gcn first-layer embedding dimensions")
ax.set_ylabel(r"f-1 Score")
ax2.set_ylabel(r"hamming loss")
ax2.set_ylim(0.090, 0.100)
ax.set_ylim(0.6,0.63)
ax.legend(lns, labs, loc=1)
plt.savefig('../results/dosage_weight.png') # 先show再save会保存空白图片
plt.show()