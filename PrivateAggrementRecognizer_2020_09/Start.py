import torch
import pandas
import numpy
from DenseNet import DenseNet_I
from Train import train
from Test import test
from DataSet import MyDataSet
from torch.utils.data import DataLoader
from Process import  process
from InputDataSet import inputDataSet
from Superkmeans import superkmeans
from Pcapprocess import pcapprocess

from DBSQLite3 import init_db
from DBSQLite3 import insert_list
from DBSQLite3 import output_result

import warnings
warnings.filterwarnings("ignore")

#初始化数据库
init_db()

# 定义是否使用GPU
device = torch.device("cpu")
# *************************************各种参数的设置*************************************************************************************

batch_size = 128

# 最大训练次数
max_epoches = 50

# 初始学习率
learning_rate = 0.001

# 网络层数
net_number = 121

# 网络模型
net = DenseNet_I()
#net = Resnet34.ResNet(Resnet34.ResidualBlock, [3, 4, 6, 3]).to(device)
#net = Resnet50.ResNet(Resnet50.Bottleneck, [3, 4, 6, 3]).to(device)



############################################Start###################################################################
for i in range(10000):
    print("Input the number of what you want me to do.Sure?")
    youchoice = input("Preprocess Pcap:: ---------------------1 \ntrain the model:: ---------------------2\ntest the model:: ----------------------3\ndue with data of network:: ------------4\nquery result of data from database:: --5\nExit:: --------------------------------100\n")

    if youchoice == '1':
        print("You choose pcap process.\n")
        print("Now input the source and the Towhere. \n")
        source = input("source: \n")
        Towhere = input("To where? \n")

        #******************************************* 数据预处理：*********************************************************************************************
        print("Running!Please wait and donnot do anything!\n")
        try:
            pcapprocess(source=source, Towhere = Towhere)
        except:
            print("Error:  Donnot Get the file,maybe you input the wrong file name.")
        else:
            print("Pcap Preprocess Done!")

    elif youchoice == '2':
        print("You choose training the model.\n")
        print("Now input the train data and the verify data.You sure?")
        ##训练模型**************************************************************************************************************************
        traindata = input("train data: \n")
        verifydata = input("verify data: \n")
        try:
            print("Running!Please wait and donnot do anything!\n")
            ####################输入训练集的位置和文件名
            traindataset = MyDataSet(traindata)
            train_data = DataLoader(traindataset, batch_size=batch_size, shuffle=True)

            verifydataset = MyDataSet(verifydata)
            verify_data = DataLoader(verifydataset, batch_size=batch_size, shuffle=True)
            train(net_number, net, train_data, verify_data, max_epoches, learning_rate)
        except:
            print("Error:  Donnot Get the file,maybe you input the wrong file name.\n\n")
        else:
            print("Train Done!\n\n")


    elif youchoice == '3':
        print("You choose testing the model.\n")
        print("Now input the model you need and the test data. You sure?\n ")
        ##测试模型准确率*********************************************************************************************************************
        model = input("model: \n")
        testdata = input("test file: \n ")
        try:
            print("Running!Please wait and donnot do anything!\n")
            #####################输入需要使用的模型参数的文件位置和名
            net.load_state_dict(torch.load(model))

            ######################输入测试集的位置和名
            dataset = MyDataSet(testdata)
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            test_correct = test(net, data_loader)
            print("Accuracy:",test_correct*100,"%","\n\n")
        except:
            print("Error:  Donnot Get the file,maybe you input the wrong file name.\n\n")
        else:
            print("Testing Done!\n\n")



    elif youchoice == '4':
        print("You choose using the model to due with the data of network.\n")
        print("Now input the model you need and the data of network. You sure?\n ")
        #对网络流进行识别*********************************************************************************************************************
        # try:
        #############################输入使用的模型位置和名
        model = input("model: \n")
        ######################输入待分析的网络流数据集的位置和文件名
        data = input("data: \n")

        print("Running!Please wait and donnot do anything!\n")

        net.load_state_dict(torch.load(model, map_location=lambda storage,loc:storage))

        dataset = inputDataSet(data)
        data_loader = DataLoader(dataset, batch_size=batch_size)

        #########输入网络模型，，未知协议识别最高置信率阈值，，结果输出的位置和文件名
        label_list = process(net,data_loader, 0.95 )

        #***********************************************************************************************************************************

        unknow_data = pandas.DataFrame()

        ##############################输入未知协议的网络流数据集的文件位置和名
        inputData = pandas.read_csv(data, index_col=None)

        longlong = len(label_list)
        origin_to_new = numpy.zeros((longlong,2),dtype=int)
        num_unknow_data =0
        for i in range(longlong):
            if label_list[i] == 404:
                origin_to_new[num_unknow_data,0] = i
                num_unknow_data+=1
                unknow_data=unknow_data.append(inputData.iloc[i,:],ignore_index=True)
    
        if len(unknow_data)!=0:
            dataSet = numpy.array(unknow_data)

            ################输入分析结果的输出位置的文件名
            output = superkmeans(dataSet)

            for i in range(num_unknow_data):
                origin_to_new[i,1] = int(output[i])+100

        for y in range(num_unknow_data):
            label_list[origin_to_new[y,0]] = origin_to_new[y,1]

        #t把label代号转换具体的协议名称
        for label_i in range(len(label_list)):
            label_list[label_i] = int(label_list[label_i])
            if label_list[label_i] == 0:
                label_list[label_i] = 'TCP'
            elif label_list[label_i] == 1:
                label_list[label_i] = 'UDP'
            elif label_list[label_i] == 2:
                label_list[label_i] = 'DNS'
            elif label_list[label_i] == 3:
                label_list[label_i] = 'SMB'
            elif label_list[label_i] == 4:
                label_list[label_i] = 'MDNS'
            elif label_list[label_i] == 5:
                label_list[label_i] = 'SSH'
            elif label_list[label_i] == 6:
                label_list[label_i] = 'VNC'
            elif label_list[label_i] == 7:
                label_list[label_i] = 'STUN'
            elif label_list[label_i] == 8:
                label_list[label_i] = 'HTTP'
            elif label_list[label_i] == 9:
                label_list[label_i] = 'NBNS'
            elif label_list[label_i] == 404:
                label_list[label_i] = 'UNKNOWN'
            elif label_list[label_i] == 99:
                label_list[label_i] = 'NOISY'
            elif label_list[label_i] >= 100:
                label_list[label_i] = 'UNKNOWN_'+str(label_list[label_i]-100)

        print(label_list)

        #根据数据名生成表名
        table_name_1 = data.split('/')
        table_name_2 = table_name_1[-1].split('.')
        table_name = table_name_2[0]+'_'+table_name_2[1]

        #将数据插入数据库
        insert_list(label_list,table_name)

        output = pandas.DataFrame(label_list)
        output.to_csv("./output/result.csv", sep=",", header=True, index=False)

        # except:
        #     print("Error:  Donnot Get the file,maybe you input the wrong file name.\n\n")
        # else:
        #     print("The result has output to the file-----./output/1111111.csv.Please check!\n\n")

    elif youchoice == '5':
        print("You choose query result of data.\n")
        print("Now input the name of table to query. \n")
        table = input("table: \n")
        output_result(table)

    elif youchoice == '100':
        print("Goodbye! I wait for you always the time.")
        break

    else:
        print("Input Error! Please input again.")
