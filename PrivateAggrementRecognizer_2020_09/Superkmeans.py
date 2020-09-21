from sklearn.cluster import KMeans
import numpy as np
import math
import pandas as pd
import warnings
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.preprocessing import StandardScaler


def superkmeans(dataSet):
    #k,myinit,Q_dataSet,origin_dataSet,Q_dataSet_toOrigin,Q_dataSet_number = k_cluster(dataSet)
    #clf = KMeans(init=myinit, n_clusters=k, max_iter=3000)
    #clf = clf.fit(Q_dataSet)
    #dataSet = StandardScaler().fit_transform(dataSet)
    res = []
# 迭代不同的eps值
    for eps in np.arange(2000,4000,50):
    # 迭代不同的min_samples值
        for min_samples in range(2,10):
            dbscan = DBSCAN(eps = eps, min_samples = min_samples)
        # 模型拟合
            dbscan.fit(dataSet)
        # 统计各参数组合下的聚类个数（-1表示异常点）
            n_clusters = len([i for i in set(dbscan.labels_) if i != -1])
        # 异常点的个数
            outliners = np.sum(np.where(dbscan.labels_ == -1, 1,0))
        # 统计每个簇的样本个数
            stats = str(pd.Series([i for i in dbscan.labels_ if i != -1]).value_counts().values)
            res.append({'eps':eps,'min_samples':min_samples,'n_clusters':n_clusters,'outliners':outliners,'stats':stats})

# 将迭代后的结果存储到数据框中
    df = pd.DataFrame(res)
# 根据条件筛选合理的参数组合
    supdf = df.loc[df.n_clusters == 3, :]
    print(supdf)

    clf = DBSCAN(eps=2800, min_samples=5).fit(dataSet)
    #dataSet['label']=clf.labels_
    #labels = clf.labels_
    #raito = dataSet.loc[dataSet['labels']==-1].x.count()/dataSet.x.count() #labels=-1的个数除以总数，计算噪声点个数占总数的比例
    #print('噪声比:', format(raito, '.2%'))
    #n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0) # 获取分簇的数目
    #print('分簇的数目: %d' % n_clusters_)
    #print("轮廓系数: %0.3f" % metrics.silhouette_score(dataSet,labels)) #轮廓系数评价聚类的好坏

    return clf.labels_
