import pandas as pd
import numpy as np
#导入 源数据 和 解析结果
import warnings
warnings.filterwarnings("ignore")

def datapreprocess(Towhere = None, source=None, output=None,):
    if output is None:

        source = pd.read_table(source, header=None)

        contains = []
        for i in range(source.__len__()):
            if (i + 1) % 3 == 0:
                s = np.array(source.iloc[i])
                contains.append(s[0])

        Res = []
        for contain in contains:
            temp = contain.split('|')
            res = []
            for i in range(temp.__len__() - 1):
                if (i >= 2 and i <= 785):
                    temp2 = temp[i]
                    # 16进制转换成10进制
                    number = int(temp2, 16)
                    res.append(number)
            if (res.__len__() < 784):
                while res.__len__() != 784:
                    res.append(0)
            if (res.__len__() != 784):
                print(res.__len__())
            Res.append(res)

        data = pd.DataFrame()

        for i in range(784):
            b = [x[i] for x in Res]
            data["pixel" + str(i)] = b

        data.to_csv(Towhere, sep=",", header=True, index=False)

    else:

        output = pd.read_csv(output, encoding='gbk')
        source = pd.read_table(source, header=None)

        protocol = output["Protocol"]  # 仅提取Protocol列的数据
        dict = {}  # 保存协议种类集合
        protocol_num = 0
        # 统计所有协议种类
        for i in protocol:
            if (dict.get(i) == None):
                dict[i] = protocol_num
                protocol_num += 1
        print(dict)

        label = np.zeros(protocol.__len__(), int)
        for i in range(protocol.__len__()):
            label[i] = dict[protocol.iloc[i]]

        contains = []
        for i in range(source.__len__()):
            if (i + 1) % 3 == 0:
                s = np.array(source.iloc[i])
                contains.append(s[0])
        print(contains.__len__())

        Res = []
        for contain in contains:
            temp = contain.split('|')
            res = []
            for i in range(temp.__len__() - 1):
                if (i >= 2 and i <= 785):
                    temp2 = temp[i]
                    # 16进制转换成10进制
                    number = int(temp2, 16)
                    res.append(number)
            if (res.__len__() < 784):
                while res.__len__() != 784:
                    res.append(0)
            if (res.__len__() != 784):
                print(res.__len__())
            Res.append(res)

        data = pd.DataFrame()
        label = pd.Series(label)
        data["label"] = label

        for i in range(784):
            b = [x[i] for x in Res]
            data["pixel" + str(i)] = b

        data.to_csv(Towhere, sep=",", header=True, index=False)

        # 统计每种协议的数量
        for i in range(protocol_num):
            count = list(data.label).count(i)
            print(i, count)



