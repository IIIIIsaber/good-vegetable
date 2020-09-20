import torch.utils.data as data
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

class MyDataSet(data.Dataset):
    def __init__(self, location):
        data_train = pd.read_csv(location, index_col=None)
        label = data_train["label"]
        data_train.drop("label", axis=1, inplace=True)
        imgs = data_train.values
        self.image = imgs
        self.label = label


    def __getitem__(self, index):
        image, target = self.image[index], self.label[index]

        return image, target

    def __len__(self):
        return len(self.image)