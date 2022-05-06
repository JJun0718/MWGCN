# -*- coding: utf-8 -*-
# @Time : 2022/4/4 17:12
# @Author : JJun
# @Site : 
# @File : plt_pq.py
# @Software: PyCharm

import matplotlib.pyplot as plt
from matplotlib import rc

rc('mathtext', default='regular')
f = plt.figure()
plt.rc('font', family='Times New Roman', size='9')

x = [0.8,0.9,1.0,1.1,1.2]
y1 = [0.620276497695852,0.612244897959183,0.610153702841173,0.610311193683232,0.610169491525423] # Y
y2 = [0.612206572769953,0.613531047265987,0.61646398503274,0.610169491525423,0.616100511865984] # Y
y3 = [0.610644257703081,0.611292580494633,0.614821591948764,0.6190036900369,0.60865475070555] # Y
y4 = [0.615455807496529,0.617079889807162,0.61332099907493,0.615525114155251,0.62185642432556] # Y
y5 = [0.626263897629536,0.616087751371115,0.617647058823529,0.611214953271028,0.609902822767237] # Y
# plt.title('')  # 折线图标题
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示汉字

plt.xlabel("$\it{q}$")
plt.ylabel("F1-score")

plt.plot(x, y1, linestyle='--', marker='<', markersize=5, label='$\it{p}$ = 0.8')  # 绘制折线图，添加数据点，设置点的大小
plt.plot(x, y2, linestyle='-.', marker='>', markersize=5, label='$\it{p}$ = 0.9')
plt.plot(x, y3, linestyle='--', marker='^', markersize=5, label='$\it{p}$ = 1.0')
plt.plot(x, y4, linestyle='-.', marker='x', markersize=5, label='$\it{p}$ = 1.1')
plt.plot(x, y5, linestyle='-', marker='o', markersize=5, label='$\it{p}$ = 1.2')
plt.grid()

# for a, b in zip(x, y1):
#     plt.text(a, b, b, ha='center', va='bottom', fontsize=10)  # 设置数据标签位置及大小
# for a, b in zip(x, y2):
#     plt.text(a, b, b, ha='center', va='bottom', fontsize=10)
# for a, b in zip(x, y3):
#     plt.text(a, b, b, ha='center', va='bottom', fontsize=10)
# for a, b in zip(x, y4):
#     plt.text(a, b, b, ha='center', va='bottom', fontsize=10)
# for a, b in zip(x, y5):
#     plt.text(a, b, b, ha='center', va='bottom', fontsize=10)

# plt.legend(['$\it{p}$=0.8', '$\it{p}$=0.9', '$\it{p}$=1.0', '$\it{p}$=1.1', '$\it{p}$=1.2'])  # 设置折线名称
plt.legend(loc=1) #Sshow legend  右上：1 左上：2 左下：3 右下;4
plt.show()  # 显示折线图

# f.savefig("../results/pq_F1-score.png", bbox_inches='tight')
f.savefig('../results/pq_F1-score.jpeg', dpi=1000, format='jpeg')


f = plt.figure()
plt.rc('font', family='Times New Roman', size='10')

x = [0.8,0.9,1.0,1.1,1.2]
y1 = [0.0955251565035937,0.0969162995594713,0.0970322281474611,0.0972640853234407,0.0959888708555529] # Y
y2 = [0.0957570136795733,0.0966844423834917,0.0950614421516345,0.0986552283793183,0.0956410850915835] # Y
y3 = [0.0966844423834917,0.0965685137955019,0.0976118710874101,0.0957570136795733,0.0964525852075121] # Y
y4 = [0.0963366566195223,0.0966844423834917,0.0969162995594713,0.0976118710874101,0.0958729422675631] # Y
y5 = [0.0928587989798284,0.0973800139114305,0.0964525852075121,0.0964525852075121,0.0977277996753999] # Y
# plt.title('')  # 折线图标题
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示汉字

plt.xlabel("$\it{q}$")
plt.ylabel("Hamming Loss")

plt.plot(x, y1, linestyle='--', marker='<', markersize=5, label='$\it{p}$ = 0.8')  # 绘制折线图，添加数据点，设置点的大小
plt.plot(x, y2, linestyle='-.', marker='>', markersize=5, label='$\it{p}$ = 0.9')
plt.plot(x, y3, linestyle='--', marker='^', markersize=5, label='$\it{p}$ = 1.0')
plt.plot(x, y4, linestyle='-.', marker='x', markersize=5, label='$\it{p}$ = 1.1')
plt.plot(x, y5, linestyle='-', marker='o', markersize=5, label='$\it{p}$ = 1.2')
plt.grid()

# for a, b in zip(x, y1):
#     plt.text(a, b, b, ha='center', va='bottom', fontsize=10)  # 设置数据标签位置及大小
# for a, b in zip(x, y2):
#     plt.text(a, b, b, ha='center', va='bottom', fontsize=10)
# for a, b in zip(x, y3):
#     plt.text(a, b, b, ha='center', va='bottom', fontsize=10)
# for a, b in zip(x, y4):
#     plt.text(a, b, b, ha='center', va='bottom', fontsize=10)
# for a, b in zip(x, y5):
#     plt.text(a, b, b, ha='center', va='bottom', fontsize=10)
font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 9,
}

# plt.legend(['$\it{p}$=0.8', '$\it{p}$=0.9', '$\it{p}$=1.0', '$\it{p}$=1.1', '$\it{p}$=1.2'])  # 设置折线名称
plt.legend(loc=4, prop = font1) #Sshow legend  右上：1 左上：2 左下：3 右下;4
plt.show()  # 显示折线图

# f.savefig("../results/pq_Hanmming-loss.png", bbox_inches='tight')
f.savefig('../results/pq_Hanmming-loss.jpeg', dpi=1000, format='jpeg')