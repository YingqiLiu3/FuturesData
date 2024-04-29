# _*_ coding:utf-8 _*_
import pandas as pd
from numpy import *
import jieba
import base64
import numpy as np

pic_url = "https://pic.cpolar.cn/"

def analysis():
    df = pd.read_csv("./Model/Analysis/Data/data.csv")
    df2 = df.copy()
    del df2['代码']
    del df2['名称']
    weight_list = np.load("./Model/Analysis/Data/Weight_List.npy").tolist()
    factor_score = np.load("./Model/Analysis/Data/Weight.npy")
    fa_t_score = np.dot(np.mat(df2), np.mat(factor_score))
    fa_t_score = np.array(fa_t_score)
    sum = []
    for i in range(0,len(fa_t_score)):
        num = 0
        data = []
        for j in range(0,len(fa_t_score[i])):
            data.append(fa_t_score[i][j]*weight_list[j])
            num+=fa_t_score[i][j]*weight_list[j]
        data.append(num)
        sum.append(data)
    return sum

def takeScore(elem):
    return elem[25][9]

def get_all():
    data = np.load("./Model/Analysis/Data/data_end.npy").tolist()
    ana = analysis()
    for i in range(0,len(data)):
        data[i].append(ana[i])
    data.sort(key=takeScore,reverse=True)
    for i in range(0,len(data)):
        for j in range(0,10):
            data[i][25][j] = str(data[i][25][j])
        data[i].append(i+1)
    return data

def get_data(code):
    data = get_all()
    for i in range(0,len(data)):
        if(data[i][0]==code):
            ans = data[i]
            ans.append(pic_url+"headmap.png")
            return data[i]

def cosine_similarity(sentence1: str, sentence2: str) -> float:
    """
    :param sentence1: s
    :param sentence2:
    :return: 两句文本的相识度
    """
    seg1 = [word for word in jieba.cut(sentence1)]
    seg2 = [word for word in jieba.cut(sentence2)]
    word_list = list(set([word for word in seg1 + seg2]))#建立词库
    word_count_vec_1 = []
    word_count_vec_2 = []
    for word in word_list:
        word_count_vec_1.append(seg1.count(word))#文本1统计在词典里出现词的次数
        word_count_vec_2.append(seg2.count(word))#文本2统计在词典里出现词的次数

    vec_1 = np.array(word_count_vec_1)
    vec_2 = np.array(word_count_vec_2)
    #余弦公式

    num = vec_1.dot(vec_2.T)
    denom = np.linalg.norm(vec_1) * np.linalg.norm(vec_2)
    cos = num / denom
    sim = 0.5 + 0.5 * cos
    return sim

def search_code(word):
    data = get_all()
    for i in range(0, len(data)):
        data[i].append(str(data[i][0]) + ' ' + str(data[i][1]))
    for i in range(0, len(data)):
        test = cosine_similarity(word, data[i][27])
        data[i][27] = str(test)
    answer = sorted(data, key=lambda s: float(s[27]), reverse=True)
    return answer