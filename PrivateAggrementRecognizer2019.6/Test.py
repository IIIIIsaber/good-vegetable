import torch

import warnings
import time
warnings.filterwarnings("ignore")

def test( net, test_loader, epoch = 0 , iteration = 0):
# 定义是否使用GPU
     with open("./Log/ACC.txt", "a") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        f.write('\n')
        device = torch.device("cpu")
        batch_size = 128

        num_test = 0
        num_correct = 0

        with torch.no_grad():
            for data in test_loader:
                img, label = data
                img = img.view(img.size(0), 1, 28, 28)
                img = img.float()
                net.eval()
                img = img.to(device)
                label = label.to(device)
                out = net(img)
                _, pred = torch.max(out.data, 1)
                num_correct += (pred == label).sum()
                num_test = num_test + label.size(0)

        final_correct = float(num_correct.item()) / float(num_test)

        f.write("Epoch= %02d,Iter= %03d,Accuracy= %.4f" % (epoch , iteration, final_correct))
        f.write('\n')
        f.flush()

     return final_correct

