# coding:utf-8 # 折线图
import numpy as np
from matplotlib import pyplot as plt

# 创建figure窗口，figsize设置窗口的大小
# plt.figure((default, 4))

# 设置横坐标轴的刻度(纵坐标不需要设置)
x_label = ['mono_comb', 'mono_fthr', 'bt_comb', 'bt_fthr', 'rule_comb', 'rule_fthr']
x_ticks = range(len(x_label))
plt.xticks(x_ticks, x_label)

# 创建数据
# DE small-scale dataset的模型变种
y1 = [20.89, 22.09, 19.41, 23.10, 21.86, 23.84]
# DE mixed dataset的模型变种
y2 = [20.85, 21.83, 19.54, 21.16, 20.25, 24.64]
# # EN small-scale dataset的模型变种
# y3 = [81.05, 79.69, 87.06, 87.54, 87.22, 88.62]
# # EN mixed dataset的模型变种
# y4 = [80.18, 77.92, 88.15, 88.71, 89.44, 89.90]

# 画4条曲线
# DE
plt.plot(x_ticks, y1, 's-', markerfacecolor='none', color='red', linewidth=0.8, label='Small-Scale')
plt.plot(x_ticks, y2, 's-', markerfacecolor='none', color='blue', linewidth=0.8, label='Mixed')
# plt.plot(x_ticks, y3, 's-', markerfacecolor='none', color='red', linewidth=0.8, label='Small-Scale')
# plt.plot(x_ticks, y4, 's-', markerfacecolor='none', color='blue', linewidth=0.8, label='Mixed')


# 设置坐标轴名称
plt.xlabel('Model Variants')
plt.ylabel('BLEU-4')

# 图标题, loc="best"自动选择放图例的合适位置
plt.legend(loc="best")

# 显示并保存图
path = r'C:\Users\pengr\Desktop\Prof.Chen研究任务\研究工作-手语翻译(数据增强+动态排序)\实验记录'
# 去除图片周围的白边
plt.savefig(path+r"\fig1_a.eps", bbox_inches='tight', pad_inches=0.0)
plt.show()