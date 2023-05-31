from .deep_fake_mask import ComponentsMasking, DFLMasking, ExtendedMasking
from .grid import GridMasking
from .meshgrid import MeshgridMasking
from .sladd import SladdMasking
from .whole import WholeMasking


def from_name(name: str, **kwargs):
    if name == "grid":
        return GridMasking(**kwargs)
    elif name == "sladd":
        return SladdMasking(**kwargs)
    elif name == "meshgrid":
        return MeshgridMasking(**kwargs)
    elif name == "whole":
        return WholeMasking(**kwargs)
    elif name == "dfl":
        return DFLMasking(**kwargs)
    elif name == "components":
        return ComponentsMasking(**kwargs)
    elif name == "extended":
        return ExtendedMasking(**kwargs)
    else:
        raise NotImplementedError
