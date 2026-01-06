from __future__ import annotations
import torch
from typing import Callable, Iterable, List, Any, Optional
import torchvision.transforms as T

def convert_to_rgb_tensor(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 4:  # BCHW
        c = x.shape[1]
        if c == 1:
            return x.expand(-1, 3, -1, -1)
        if c == 3:
            return x
        if c == 4:
            return x[:, :3]
        raise ValueError(f"Can't convert BCHW tensor with C={c} to RGB")
    raise ValueError(f"Expected BCHW, got shape {tuple(x.shape)}")


def build_tensor_preprocess(transforms_list: Iterable[Any]) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Builds a tensor-native preprocess that mirrors an existing torchvision/open_clip
    preprocess pipeline, using the provided transform list order at runtime.

    Only special-cases:
      - PIL-only `_convert_to_rgb` function
      - `ToTensor()`
    """
    ops: List[Callable[[torch.Tensor], torch.Tensor]] = []

    for t in transforms_list:
        # 1) Replace PIL-only rgb conversion
        if callable(t) and getattr(t, "__name__", None) == "_convert_to_rgb":
            ops.append(convert_to_rgb_tensor)
            continue

        # 2) Skip ToTensor() (PIL/ndarray -> tensor)
        if isinstance(t, T.ToTensor):
            continue

        # 3) Otherwise, try applying the transform directly (keeps its params + order)
        def _apply(x: torch.Tensor, tt=t) -> torch.Tensor:
            try:
                return tt(x)
            except Exception as e:
                raise TypeError(
                    f"Transform {tt!r} failed on tensor input with shape {tuple(x.shape)}. "
                ) from e

        ops.append(_apply)

    def preprocess(x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        for op in ops:
            x = op(x)
        return x

    return preprocess