from dataclasses import dataclass, field
from typing import Dict, Optional

import torch

from optim import omp_incremental_cholesky_with_fallback


def to_nonnegative_activations(activations: torch.Tensor) -> torch.Tensor:
    positive_activations = activations.clamp(min=0.0)
    negative_activations = activations.clamp(max=0.0)
    return torch.cat([positive_activations, negative_activations], dim=-1)


def to_unbounded_activations(activations: torch.Tensor) -> torch.Tensor:
    half_dim = activations.size(-1) // 2
    positive_activations = activations[..., :half_dim]
    negative_activations = activations[..., half_dim:]
    return positive_activations + negative_activations


class ITO_SAE:
    def __init__(self, atoms, l0=8, cfg=None):
        self.atoms = atoms
        self.l0 = l0
        self.cfg = cfg

    def encode(self, x):
        shape = x.size()
        x = x.view(-1, shape[-1])
        activations = omp_incremental_cholesky_with_fallback(self.atoms, x, self.l0)
        activations = to_nonnegative_activations(activations)
        return activations.view(*shape[:-1], -1)

    def decode(self, activations):
        original_activations = to_unbounded_activations(activations)
        original_device = original_activations.device
        original_activations = original_activations.to(self.atoms.device)
        return torch.matmul(original_activations, self.atoms).to(original_device)

    @property
    def W_dec(self):
        return torch.cat([self.atoms, -self.atoms], dim=0)

    def __call__(self, x):
        acts = self.encode(x)
        return self.decode(acts)

    @property
    def device(self):
        return self.atoms.device

    @property
    def dtype(self):
        return self.atoms.dtype

    def to(self, device=None, dtype=None):
        if device:
            self.atoms = self.atoms.to(device)
        if dtype:
            self.atoms = self.atoms.to(dtype)
        return self


@dataclass
class ITO_SAEConfig:
    model_name: str
    d_in: int
    d_sae: int
    hook_layer: int
    hook_name: str
    dtype: str

    context_size: int = None
    dtype: str = ""
    device: str = ""

    model_from_pretrained_kwargs: Dict = field(default_factory=dict)
    hook_head_index: Optional[int] = None
    prepend_bos: bool = True
    normalize_activations: str = "none"
    dataset_trust_remote_code: bool = True
    seqpos_slice: tuple = (None,)
