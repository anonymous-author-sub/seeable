import torch
from torch import nn
import torchvision
from torch.nn import functional as F
from efficientnet_pytorch import EfficientNet
from typing import Any, Callable, Dict, List, Optional, Tuple
from loguru import logger

class SbiEff4(nn.Module):

    def __init__(self, path:Optional[str]=None, weights_path:Optional[str]=None):
        super(SbiEff4, self).__init__()

        if weights_path is not None:
            logger.info("Loading pretrained EfficientNet-b4")
            self.net=EfficientNet.from_pretrained("efficientnet-b4", weights_path=weights_path, advprop=True, num_classes=2)
        else:
            logger.info("Loading un-trained EfficientNet-b4")
            self.net=EfficientNet.from_name("efficientnet-b4", num_classes=2)
        
        # load checkpoint path
        if path is not None:
            logger.info(f"Loading checkpoint from {path}")
            checkpoint = torch.load(path)
            self.load_state_dict(checkpoint["model"])

    def forward(self,x):
        x=self.net(x)
        return x
    