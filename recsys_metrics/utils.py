from typing import Optional, Tuple

import torch
from torch import Tensor


def _check_topk_validity(preds: Tensor, k: Optional[int] = None) -> int:
    _max_k = preds.shape[-1]
    if k is None:
        k = _max_k

    if not (isinstance(k, int) and k > 0):
        raise ValueError(f'`k` has to be a positive integer or None')

    k = min(_max_k, k)
    return k


def _check_ranking_inputs(
    preds: Tensor, 
    target: Tensor, 
    k: Optional[int] = None, 
    batched: Optional[bool] = True
) -> Tuple[Tensor, Tensor, int]:
    """Adapted from https://github.com/PyTorchLightning/metrics/blob/93cb842f24d15804dd2e7677ca7fc6631b234773/torchmetrics/utilities/checks.py"""
    if preds.device != target.device:
        raise ValueError('`preds` and `target` are must on the same device')

    if preds.shape != target.shape:
        raise ValueError('`preds` and `target` must have the same shape')

    if not preds.numel() or not preds.size():  # `preds` and `target` already have same shape when executing this line
        raise ValueError('`preds` and `target` must be non-empty and non-scalar tensors')

    if target.dtype not in (torch.bool, torch.long, torch.int) and not target.is_floating_point():
        raise ValueError('`target` must be a tensor of booleans, integers or floats')

    if not preds.is_floating_point():
        raise ValueError('`preds` must be a tensor of floats')

    if target.max() > 1 or target.min() < 0:
        raise ValueError(f'`target` must contain binary values')

    preds = preds.float()
    target = target.long()
    if batched and preds.ndim == 1:
        preds = preds[None, ...]
        target = target[None, ...]
    if preds.ndim != 2:
        raise ValueError(f'`preds` and `target` must be either one or two dimensional tensors')

    k = _check_topk_validity(preds, k=k)
    return preds, target, k


def _check_beyond_accuracy_inputs(
    preds: Tensor, 
    indexes: Tensor, 
    k: Optional[int] = None
) -> Tuple[Tensor, Tensor, int]:
    if preds.device != indexes.device:
        raise ValueError('`preds` and `indexes` are must on the same device')

    if preds.shape != indexes.shape:
        raise ValueError('`preds` and `indexes` must have the same shape')

    if not preds.numel() or not preds.size():  # `preds` and `indexes` already have same shape when executing this line
        raise ValueError('`preds` and `indexes` must be non-empty and non-scalar tensors')

    if indexes.dtype is not torch.long:
        raise ValueError('`indexes` must be a tensor of long integers')

    if not preds.is_floating_point():
        raise ValueError('`preds` must be a tensor of floats')

    preds = preds.float()
    indexes = indexes.long()
    if preds.ndim != 2:
        raise ValueError('`preds` and `indexes` must be two dimensional tensors')

    k = _check_topk_validity(preds, k=k)
    return preds, indexes, k


def _check_preds_popularities_inputs(
    preds: Tensor,
    popularities: Tensor,
    k: Optional[int] = None
) -> Tuple[Tensor, Tensor, int]:
    if preds.device != popularities.device:
        raise ValueError('`preds` and `popularities` are must on the same device')
    
    if preds.shape != popularities.shape:
        raise ValueError('`preds` and `popularities` must have the same shape')

    if not preds.numel() or not preds.size():  # `preds` and `popularities` already have same shape when executing this line
        raise ValueError('`preds` and `popularities` must be non-empty and non-scalar tensors')

    if not preds.is_floating_point():
        raise ValueError('`preds` must be a tensor of floats')

    if not popularities.is_floating_point():
        raise ValueError('`popularities` must be a tensor of floats')

    if popularities.min() < 0 or popularities.max() > 1:
        raise ValueError('`popularities` must have values between 0 and 1')

    preds = preds.float()
    popularities = popularities.float()
    if preds.ndim != 2:
        raise ValueError('`preds` and `popularities` must be two dimensional tensors')

    k = _check_topk_validity(preds, k=k)
    return preds, popularities, k


def _reduce_tensor(tensor: Tensor, reduction: str) -> Tensor:
    if reduction == 'mean':
        return tensor.mean(0)
    elif reduction == 'sum':
        return tensor.sum(0)
    elif reduction == 'none':
        return tensor
    else:
        raise ValueError(f'`reduction` must be one of `mean`, `sum`, or `none`')


def div_no_nan(a: Tensor, b: Tensor, na_value: Optional[float] = 0.) -> Tensor:
    return (a / b).nan_to_num_(nan=na_value, posinf=na_value, neginf=na_value)
