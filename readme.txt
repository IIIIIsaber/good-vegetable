用户手册
——Luckycat小组


一、产品概述
项目基于深度学习的网络流量的识别与分类系统的设计与实现，是在linux系统下使用pycharm开发的，主要目的是识别网络流量的种类并对其分类。

二、产品功能及使用说明
使用软件操作步骤：打开虚拟机，密码为123456，在命令行下cd至pycharm所在目录（/home/protocol/Downloads/pycharm-edu-2019.1/bin/），执行sudo pycharm.sh运行pycharm，打开左侧软件结构目录中的start.py并run，控制台输出如下表所示：
输入相应数字，可执行相应功能。
产品功能为识别并分类网络流量，主要分为六个部分：

	1.训练模型
	输入1，训练模型。
	例如，分别输入：
		1
		./data/train.csv
		./data/verify.csv

	2.测试模型
	输入2，测试模型。
	例如，分别输入：
		2
		./model/DenseNet121_45500.tar
		./data/test.csv

	3.抓取pcap包（要抓很多数据包的话，请做一些能够发送数据包的操作，比如访问网页。且请尽量将虚拟机先联网，不然基本上只有DNS协议的数据包产生）
	输入3，从自己电脑的端口抓取pcap包。
	例如，分别输入：
		3
		demo
		100

	4.处理pcap包
	输入4，将上一步的.pcap文件转为.csv文件。
	例如，分别输入：
		4
		./pcaps/demo.pcap

	5.识别处理后的pcap包
	输入5，识别上述的.csv文件。
	例如，分别输入：
		5
		./model/DenseNet121_45500.tar
		./data/demo.csv

	6.数据库结果查询
	输入6，将处理后的结果从数据库调取并可视化展示。
	例如，分别输入：
		6


数据集里面各个标签代表的含义
0:TCP
1:UDP
2:DNS
3:SMB
4:MDNS
5:SSH
6:VNC
7:STUN
8:HTTP
9:NBNS
99:噪点
404:未经分类的未知协议
>=100&<=400:已经被分类的未知协议
