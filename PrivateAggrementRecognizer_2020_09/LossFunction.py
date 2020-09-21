import torch
import warnings
warnings.filterwarnings("ignore")

def getlossFunction():
     return torch.nn.CrossEntropyLoss()