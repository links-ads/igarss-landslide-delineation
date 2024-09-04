import albumentations as A
from albumentations.core.pydantic import SymmetricRangeType
from albumentations.core.types import ScaleFloatType
from pydantic import Field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union, cast
import random
import numpy as np



class RandomBrightnessContrast(A.ImageOnlyTransform):

    """Randomly change brightness and contrast of the input image. This is
    equivalent to img = img * alpha + beta * beta_factor, where alpha
    controls brightness, and beta controls contrast.

    Args:
        brightness_limit: factor range for changing brightness.
            If limit is a single float, the range will be (-limit, limit). Default: (-0.2, 0.2).
        contrast_limit: factor range for changing contrast.
            If limit is a single float, the range will be (-limit, limit). Default: (-0.2, 0.2).
        contrast_factor: adjust contrast by this factor.
        p: probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32

    """

    class InitSchema(A.core.transforms_interface.BaseTransformInitSchema):
        brightness_limit: SymmetricRangeType = (-0.5, 0.5)
        contrast_limit: SymmetricRangeType = (-0.5, 0.5)
        contrast_factor: float = Field(default=True, description="Contrast factor")

    def __init__(
        self,
        brightness_limit: ScaleFloatType = (-0.2, 0.2),
        contrast_limit: ScaleFloatType = (-0.2, 0.2),
        contrast_factor: float = 1000.,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply=always_apply, p=p)
        self.brightness_limit = cast(Tuple[float, float], brightness_limit)
        self.contrast_limit = cast(Tuple[float, float], contrast_limit)
        self.contrast_factor = contrast_factor

    def apply(self, img: np.ndarray, alpha: float, beta: float, **params: Any) -> np.ndarray:
        img = img.astype("float32")

        if alpha != 1:
            img *= alpha    # adjust brightness
        if beta != 0:
            img += beta * self.contrast_factor   # adjust contrast
        return img

    def get_params(self) -> Dict[str, float]:
        return {
            "alpha": 1.0 + random.uniform(self.brightness_limit[0], self.brightness_limit[1]),
            "beta": 0.0 + random.uniform(self.contrast_limit[0], self.contrast_limit[1]),
        }

    def get_transform_init_args_names(self) -> Tuple[str, str, str]:
        return ("brightness_limit", "contrast_limit", "contrast_factor")