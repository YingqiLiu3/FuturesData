import pandas as pd
import numpy as np
import math as math
import numpy as np
from numpy import *
from scipy.stats import bartlett
from factor_analyzer import *
import numpy.linalg as nlg
from sklearn.cluster import KMeans
from matplotlib import cm
import matplotlib.pyplot as plt
def main():
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    df=pd.read_csv("./Data/data.csv")
    df2 = df.copy()
    del df2['代码']
    del df2['名称']
    # 皮尔森相关系数
    df2_corr = df2.corr()
    # 热力图
    cmap = cm.Blues
    fig = plt.figure()
    ax = fig.add_subplot(111)
    map = ax.imshow(df2_corr, interpolation='nearest', cmap=cmap, vmin=0, vmax=1)
    plt.title('correlation coefficient--headmap')
    ax.set_yticks(range(len(df2_corr.columns)))
    ax.set_yticklabels(df2_corr.columns)
    plt.colorbar(map)
    plt.savefig("D:/test/Huaqi/Qihuo/Pic/headmap.png")
    plt.show()
    # KMO测度
    def kmo(dataset_corr):
        corr_inv = np.linalg.inv(dataset_corr)
        nrow_inv_corr, ncol_inv_corr = dataset_corr.shape
        A = np.ones((nrow_inv_corr, ncol_inv_corr))
        for i in range(0, nrow_inv_corr, 1):
            for j in range(i, ncol_inv_corr, 1):
                A[i, j] = -(corr_inv[i, j]) / (math.sqrt(corr_inv[i, i] * corr_inv[j, j]))
                A[j, i] = A[i, j]
        dataset_corr = np.asarray(dataset_corr)
        kmo_num = np.sum(np.square(dataset_corr)) - np.sum(np.square(np.diagonal(A)))
        kmo_denom = kmo_num + np.sum(np.square(A)) - np.sum(np.square(np.diagonal(A)))
        kmo_value = kmo_num / kmo_denom
        return kmo_value
    # 巴特利特球形检验
    df2_corr1 = df2_corr.values
    # 求特征值和特征向量
    eig_value, eigvector = nlg.eig(df2_corr)  # 求矩阵R的全部特征值，构成向量
    eig = pd.DataFrame()
    eig['names'] = df2_corr.columns
    eig['eig_value'] = eig_value
    eig.sort_values('eig_value', ascending=False, inplace=True)
    eig1 = pd.DataFrame(eigvector)
    eig1.columns = df2_corr.columns
    eig1.index = df2_corr.columns
    # 求公因子个数m,使用前m个特征值的比重大于85%的标准，选出了公共因子是两个
    weight_list = []
    last = 0
    for m in range(1, 23):
        print("第" + str(m) + "个因子占比：" + str(eig['eig_value'][:m].sum() / eig['eig_value'].sum()))
        weight_list.append(eig['eig_value'][:m].sum() / eig['eig_value'].sum()-last)
        last=eig['eig_value'][:m].sum() / eig['eig_value'].sum()
        if eig['eig_value'][:m].sum() / eig['eig_value'].sum() >= 0.85:
            print("\n公因子个数:", m)
            break
    np.save("./Data/Weight_List.npy",np.array(weight_list))
    # 因子载荷阵
    A = np.mat(np.zeros((23, 9)))
    i = 0
    j = 0
    while i < 9:
        j = 0
        while j < 23:
            A[j:, i] = sqrt(eig_value[i]) * eigvector[j, i]
            j = j + 1
        i = i + 1
    a = pd.DataFrame(A)
    a.columns = ['factor1', 'factor2', 'factor3','factor4', 'factor5', 'factor6','factor7', 'factor8', 'factor9']
    a.index = df2_corr.columns
    fa = FactorAnalyzer(n_factors=3)
    fa.loadings_ = a
    # print(fa.loadings_)
    var = fa.get_factor_variance()  # 给出贡献率
    # 因子旋转
    rotator = Rotator()
    b = pd.DataFrame(rotator.fit_transform(fa.loadings_))
    b.columns = ['factor1', 'factor2', 'factor3','factor4', 'factor5', 'factor6','factor7', 'factor8', 'factor9']
    b.index = df2_corr.columns
    # 因子得分
    X1 = np.mat(df2_corr)
    X1 = nlg.inv(X1)
    b = np.mat(b)
    factor_score = np.dot(X1, b)
    print(factor_score)
    print(type(factor_score))
    np.save("./Data/Weight.npy",np.array(factor_score))
    factor_score = pd.DataFrame(factor_score)
    factor_score.columns = ['factor1', 'factor2', 'factor3','factor4', 'factor5', 'factor6','factor7', 'factor8', 'factor9']
    factor_score.index = df2_corr.columns
    fa_t_score = np.dot(np.mat(df2), np.mat(factor_score))
    print("\n九个因子得分：\n", pd.DataFrame(fa_t_score))
if __name__ == '__main__':
    main()