import os
import random
import numpy as np
import torch

def set_seed(seed=42):
    """
    设置随机种子以确保可重复性
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_lr(optimizer):
    """
    获取当前学习率
    """
    for param_group in optimizer.param_groups:
        return param_group['lr'] 