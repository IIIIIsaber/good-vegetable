from sklearn.cluster import KMeans
import numpy as np
import math
import pandas
import warnings
warnings.filterwarnings("ignore")
# 欧氏距离计算
def distEclud(x,y):
    return np.sqrt(np.sum((x-y)**2))
#平均距离
def mdistance(dataSet):
    m,n = np.shape(dataSet)

    #距离之和
    AllDist = 0.0
    num = 0

    for i in range(m):
        for j in range(m):
            if i ==j:
                continue
            AllDist += distEclud(dataSet[i,:],dataSet[j,:])
            num = num + 1

    MDist = AllDist / num
    return MDist
#标准差
def sdistance(dataSet,Mdist):
    m,n = np.shape(dataSet)
    #距离之和
    AllDist = 0.0
    num = 0
    for i in range(m):
        for j in range(m):
            if i ==j:
                continue
            AllDist += (  (Mdist - ( distEclud(dataSet[i,:],dataSet[j,:])  ))**2)
            num += 1

    SDist = AllDist / num
    SDist = math.sqrt(SDist)

    return SDist
#数据对象点的密度函数值为
def density(dataSet,SDist,i):
    m,n = np.shape(dataSet)
    alltemp = 0
    for j in range(m):
        if SDist < distEclud(dataSet[i,:],dataSet[j,:]):
            temp = 0
        else:
            temp = 1
        alltemp += temp

    return alltemp
#计算平均密度
def mdensity(dataSet, SDist):
    m,n = np.shape(dataSet)
    MDensity = 0.0
    for i in range(m):
        temp = density(dataSet,SDist,i)
        MDensity += temp
    MDensity = MDensity / m
    return MDensity
#计算密度标准差
def sdensity(dataSet,SDist,MDensity):
    m,n = np.shape(dataSet)
    #距离之和
    SDensity = 0.0

    for i in range(m):
        SDensity += ((MDensity - density(dataSet,SDist,i))**2 )

    SDensity = SDensity / m
    SDensity = math.sqrt(SDensity)
    return SDensity


'''
步骤：
1）根据公式（1）（2）（3）计算数据对象之间的欧式距离，样
本的平均距离和样本的标准差。
2）根据公式（4）（5）（6）计算样本点的密度函数值，平均密
度和密度标准差。
3）根据公式（7）将满足孤立点条件的点放入集合 G，其余
点放入集合Q。
4）在Q中执行（2），样本点的密度函数值存入集合D，将大
于密度标准差的密度函数值放入集合M。
5）找到M中密度函数最大值MAX在Q中对应的样本点xi
即为初始聚类中心。
6）将以初始聚类中心为圆心，样本标准差为半径的圆内所
有点的密度函数值赋为0。
7）重复（4）~（6）直到找到k个初始聚类中心。
'''
#确定最佳 k 值
def k_cluster(dataSet):
    origin_dataSet = dataSet


    m,n = np.shape(dataSet)

    k = 0
    #初始质心
    centroids = np.zeros((0,n), dtype = np.int)
    #样本的平均距离
    Mdist = 0.0
    #样本的标准差
    SDist = 0.0
    #样本的平均密度
    MDensity = 0.0
    #样本的密度标准差
    SDensity = 0.0
    len_M_density = 0
    #孤立点
    Guli_dataSet = np.zeros((0,n), dtype = np.int)
    #非孤立点
    Q_dataSet = np.zeros((0,n), dtype = np.int)
    #密度函数值
    D_density = np.zeros(0, dtype = np.int)
    #大于密度标准差的密度函数值
    M_density = np.zeros((0,1), dtype = np.int)
    #找出孤立点和非孤立点
    Mdist = mdistance(dataSet)

    SDist = sdistance(dataSet,Mdist)

    MDensity = mdensity(dataSet,SDist)

    SDensity = sdensity(dataSet,SDist,MDensity)

    #找出大于密度标准差的所有值,即孤立点

    # 再原始位置上判断孤立点位置，，0为非孤立点，1为孤立点
    Q_dataSet_toOrigin = np.zeros(m, dtype = np.int)
    Q_dataSet_number = 0
    for i in range(m):
        if density(dataSet,SDist,i) < SDensity*0.4:
            Guli_dataSet = np.append(Guli_dataSet,[dataSet[i,:]],axis=0)
            origin_dataSet[i,0]=100
        else:
            Q_dataSet = np.append(Q_dataSet,[dataSet[i,:]],axis=0)
            Q_dataSet_toOrigin[Q_dataSet_number]= i
            Q_dataSet_number+=1
            #新的数据集

    mm,nn = np.shape(Q_dataSet)


    #Q集合，即非孤立点的密度函数值存入集合D，
    Mdist = mdistance(Q_dataSet)

    SDist = sdistance(Q_dataSet,Mdist)
    MDensity = mdensity(Q_dataSet,SDist)
    SDensity = sdensity(Q_dataSet,SDist,MDensity)

    for i in range(mm):
        temp = density(Q_dataSet,SDist,i)
        D_density = np.append(D_density,np.array(temp))

    #k值最大为10
    for abc in range(5):
        #将大于密度标准差的密度函数值放入集合M[密度值，位置]
        M_density = np.zeros((0,2), dtype = np.int)
        for i in range(mm):
            temp = D_density[i]
            if temp > 1.6*SDensity and temp >0:
                M_density = np.append(M_density,np.array([[temp,i]]),axis=0)
        len_M_density = len(M_density)
        if len_M_density == 0:
            break
        #找到M中密度函数最大值MAX在Q中对应的样本点Xi即为初始聚类中心。
        maxnum_1 = 0
        max_1 = 0
        for i in range(len_M_density):
            if M_density[i,0] > max_1:
                max_1 = M_density[i,0]
                maxnum_1 = M_density[i,1]
        #将以初始聚类中心为圆心，样本标准差为半径的圆内所有点的密度函数值赋为0
        for i in range(mm):
            distance_temp = distEclud(Q_dataSet[maxnum_1,:],Q_dataSet[i,:])
            if distance_temp <= SDist:
                D_density[i] = 0
            D_density[maxnum_1] = 0

        k += 1
        centroids = np.append(centroids,[Q_dataSet[maxnum_1,:]] ,axis=0)

    return k,centroids,Q_dataSet,origin_dataSet,Q_dataSet_toOrigin,Q_dataSet_number

def superkmeans(dataSet):
    k,myinit,Q_dataSet,origin_dataSet,Q_dataSet_toOrigin,Q_dataSet_number = k_cluster(dataSet)
    clf = KMeans(init=myinit, n_clusters=k, max_iter=3000)
    clf = clf.fit(Q_dataSet)

    return clf.labels_
