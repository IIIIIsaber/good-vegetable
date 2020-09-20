import torch.optim.lr_scheduler as lr_scheduler
import warnings
warnings.filterwarnings("ignore")

def getscheduler(optimizer, time=None):
    if time is None:
        time = [5, 8, 10, 12]
    scheduler = lr_scheduler.MultiStepLR(optimizer, time, 0.3)
    return scheduler
