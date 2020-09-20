import torch.utils.data as data
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

class inputDataSet(data.Dataset):
    def __init__(self, location):
        data_train = pd.read_csv(location, index_col=None)
        imgs = data_train.values
        self.image = imgs

    def __getitem__(self, index):
        image= self.image[index]
        return image

    def __len__(self):
        return len(self.image)