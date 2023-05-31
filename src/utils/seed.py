import random
import torch
import numpy as np
import torch.backends.cudnn as cudnn


def seed_everything(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
