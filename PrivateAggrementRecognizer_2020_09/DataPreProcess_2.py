import pandas

output = pandas.read_csv("./data/train.csv",encoding='gbk')

# del output['label']
output = output.iloc[0:100,:]
del output['label']
output.to_csv("data_100_3.csv", sep=",", header=True, index=False)