# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 18:24:40 2017

@author: admin
"""

import pandas as pd
import numpy as np
import random

# 读取创建数据表
data = pd.read_table('D:\\train.txt', sep='\t', header=None, names=['user', 'blog', 'time', 'forward', 'comment', 'like', 'content'], encoding='utf-8')

avg_data = data.groupby('user').mean()


count = 100000
data_size = len(data)

'''
i=0        
# 随机抽取测试样本，计算准确率
def test_precision():
    fenzi = np.zeros(3)
    fenmu = np.zeros(3)
    
    for i in range(count):
        
        cur = random.randint(0, data_size-1)
        actual = data.loc[cur]
        predict = avg_data.loc[actual.user]
        
        # 计算偏差
        de_forward = np.zeros(3)
        temp = abs(predict.forward * random.uniform(0.9, 1.1) - actual.forward) / (actual.forward + 5)
        de_forward += temp
        if '抽奖' in str(actual.content):
            i+=1
            de_forward[1] = abs(predict.forward * random.uniform(0.95, 1.15) - actual.forward) / (actual.forward + 5)
            de_forward[2] = abs(predict.forward * random.uniform(1.0, 1.2) - actual.forward) / (actual.forward + 5)
        de_comment = abs(predict.comment * random.uniform(0.9, 1.1) - actual.comment) / (actual.comment + 3)
        de_like = abs(predict.like * random.uniform(0.9, 1.1) - actual.like) / (actual.like + 3)
        
        precision_cur = 1 - 0.5 * de_forward - 0.25 * de_comment - 0.25 * de_like
        #sgn = 1 if precision_cur-0.8>0 else 0 #precison_cur - 0.8 > 0 ? 1:0
        sgn=precision_cur.copy()
        sgn[sgn>0.8] = 1
        sgn[sgn<=0.8] = 0
        count_cur = actual.forward + actual.comment + actual.like + 1
        if count_cur > 100:
            count_cur = 100
            
        fenzi += count_cur * sgn
        fenmu += count_cur
        
    return fenzi/fenmu
    
# print(fenzi/fenmu)

parray = np.zeros([10, 3])

print(test_precision())

'''
   
reader = open('D:\\test.txt', 'r', encoding='utf-8')
writer = open('D:\\test_pre.txt', 'w', encoding='utf-8')

i=0
line = reader.readline()

while line:
    params = line.split('\t')
    user = params[0]
    content = params[3]
    forward = 0
    comment = 0
    like = 0
    
    if user in avg_data.index:
        predict = avg_data.loc[user]
        if '抽奖' in content:
            forward = predict.forward * random.uniform(0.95, 1.15)
        else:
            forward = predict.forward * random.uniform(0.9, 1.1)
        comment = predict.comment * random.uniform(0.9, 1.1)
        like = predict.like * random.uniform(0.9, 1.1)
        
    
    writer.write('%s\t%s\t%d,%d,%d\n' % (params[0], params[1], forward, comment, like))
    line = reader.readline()
    i+=1
    #if i>10:
       #break;
        
reader.close()
writer.close()

print(i)


