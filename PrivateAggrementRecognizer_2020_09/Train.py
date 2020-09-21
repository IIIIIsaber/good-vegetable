import torch
import Test

import warnings
from Scheduler import getscheduler
from LossFunction import getlossFunction
import time

def train(net_number, net, train_loader,verifyname, max_epoches,learning_rate=0.001):
    warnings.filterwarnings("ignore")


    # 定义是否使用GPU
    device = torch.device("cpu")
    with open("./Log/iteration_log.txt", "a") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        f.write('\n')
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, betas=(0.9, 0.999))
        lossFunction = getlossFunction()
        scheduler = getscheduler(optimizer)
        iteration = 1
        for epoch in range(max_epoches):
            print("epoch: {:d}".format(epoch + 1))
            for data in train_loader:
                scheduler.step()
                img, label = data
                input = img.view(img.size(0), 1, 28, 28)
                input = input.float()
                input = input.to(device)
                label = label.to(device)
                length = len(train_loader)
                # 将梯度置零
                optimizer.zero_grad()
                # 前向传播-计算误差-反向传播-优化
                net.train()
                outputs = net(input)
                loss = lossFunction(outputs, label)
                loss.backward()
                optimizer.step()

                iter_loss = loss.item()
                _, predlabel = torch.max(outputs.data, 1)
                iter_total = label.size(0)
                correct = (predlabel == label).sum()

                f.write('Epoch:%03d   iter:%05d | Loss:%.03f | Acc:%.3f'
                        % (epoch + 1, iteration, float(iter_loss), float(correct) / iter_total))
                f.write('\n')
                f.flush()

                if iteration % 25 == 0:
                    Test.test(net, verifyname ,epoch + 1, iteration)
                if iteration % 100 == 0:
                    # save the net
                    if net_number == 121:
                        torch.save(net.state_dict(), './model/DenseNet121_%02d.tar' % (iteration))
                    if net_number == 34:
                        torch.save(net.state_dict(), './model/ResNet34_%02d.tar' % (iteration))
                    if net_number == 50:
                        torch.save(net.state_dict(), './model/ResNet50_%02d.tar' % (iteration))
                iteration += 1




