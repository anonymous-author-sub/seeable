from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union
from albumentations.core.transforms_interface import BasicTransform


class BaseTransform(BasicTransform):
    """Base Transform using Albumentations
    It provides the following features:

    - setup: called before the transform is applied to get additional information
    """

    AUGS = "augs"
    AUG = "aug"

    def setup(self, aug=None, **kwargs) -> dict:
        """fill the aug param with additional information"""
        return

    def __call__(self, *args, force_apply: bool = False, **kwargs) -> Dict[str, Any]:
        augs: dict = kwargs.setdefault(BaseTransform.AUGS, dict())
        aug = {
            "force_apply": force_apply,
            "always_apply": self.always_apply,
            "index": len(kwargs[BaseTransform.AUGS]),
            "is_applied": False,
        }
        augs[self.__class__.__name__] = aug
        self.setup(aug=aug, **kwargs)
        return super().__call__(*args, force_apply=force_apply, **kwargs)

    def update_params(self, params: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """update the params with the information from the setup:
        width, height, aug and augs"""
        augs = kwargs[BaseTransform.AUGS]
        aug = augs[self.__class__.__name__]
        aug["is_applied"] = True

        # update params that will be passed to apply (targets)
        params = super().update_params(params, **kwargs)
        params.update(
            {
                BaseTransform.AUGS: augs,
                BaseTransform.AUG: aug,
                "width": params["cols"],
                "height": params["rows"],
            }
        )
        return params

    @property
    def targets(self) -> Dict[str, Callable]:
        return {}

    @property
    def target_dependence(self) -> Dict:
        return {}
