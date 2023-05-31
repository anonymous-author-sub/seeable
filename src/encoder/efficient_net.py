from torch import nn
import efficientnet_pytorch


class EfficientNet(nn.Module):
    METAS = {
        0: {
            "params": 5.3,
            "top-1": 76.3,
        },
        1: {
            "params": 7.8,
            "top-1": 78.8,
        },
        2: {
            "params": 9.2,
            "top-1": 79.8,
        },
        3: {
            "params": 12,
            "top-1": 81.1,
        },
        4: {
            "params": 19,
            "top-1": 82.6,
        },
        5: {
            "params": 30,
            "top-1": 83.3,
        },
        6: {
            "params": 43,
            "top-1": 84.0,
        },
        7: {
            "params": 66,
            "top-1": 84.4,
        },
    }

    def __init__(self, size: int = 4, pretrained=True):
        super().__init__()
        assert (
            isinstance(size, int) and size in EfficientNet.METAS.keys()
        ), f"size must be an int in f{list(EfficientNet.METAS.keys())} got {size}"

        self.name = f"efficientnet-b{size}"
        self.meta = EfficientNet.METAS[size]
        self.net = efficientnet_pytorch.EfficientNet.from_pretrained(
            self.name,
            advprop=True,
            num_classes=2,
            include_top=False,
            in_channels=1,
        )

    def forward(self, x):
        x = self.net(x)
        # model.extract_features(img)
        return x
