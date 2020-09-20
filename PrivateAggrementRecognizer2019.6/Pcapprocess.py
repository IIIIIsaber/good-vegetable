#!/usr/bin/env python
#coding=utf-8
#读取pcap文件，解析相应的信息

import struct

def pcapprocess(Towhere = None, source=None):
  MaxPixel = 784

  fpcap = open(source,'rb')
  fcsv = open(Towhere,'w')

  string_data = fpcap.read()

  #把列名信息写入result.csv
  fcsv.write("label")
  for key in range(2500):
      fcsv.write(",pixel"+ str(key)) 
  fcsv.write("\n")

  #pcap文件的数据包数量与内容
  packet_num = 0
  packet_data = []

  i =24

  while(i<len(string_data)):
    
    #数据包头包长字段bytes     
    lens = string_data[i+12:i+16]
    #数据包包长的int表示
    packet_len = struct.unpack('I',lens)[0]

    #print(packet_len)
    #把数据包的内容进行处理，转成0~255的数，并没有的进行补全

    temp = ''                                            
    count = 0
    for j in range(i+16,i+16+packet_len):
      count = count + 1
      if(count > MaxPixel) :
        break
      temp = temp + ',' + str(struct.unpack('B',string_data[j:j+1])[0])
    for j in range(count, MaxPixel):
      temp = temp + ',' + '0'
    packet_data.append(temp)                              #packet_data为存每行像素值的列表
    i = i+ packet_len+16
    packet_num+=1
      
  #把pacp文件里的数据包信息写入result.csv
  for i in range(packet_num):
      fcsv.write(packet_data[i]+'\n')

  #fcsv.write('一共有'+str(packet_num)+"包数据"+'\n')

  fcsv.close()
  fpcap.close()

