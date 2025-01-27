from dataclasses import dataclass, field
from typing import Dict, Optional

import torch


def omp(D, x, n_nonzero_coefs):
    batch_size = x.size(0)
    num_dict_atoms = D.size(0)

    residuals = x.clone()
    coefficients = torch.zeros(batch_size, num_dict_atoms, device=x.device)

    for _ in range(n_nonzero_coefs):
        correlations = torch.matmul(residuals, D.T)
        best_atoms = torch.argmax(torch.abs(correlations), dim=1)
        coefficients_for_atoms = correlations[torch.arange(batch_size), best_atoms]
        coefficients[torch.arange(batch_size), best_atoms] += coefficients_for_atoms
        residuals -= coefficients_for_atoms.unsqueeze(1) * D[best_atoms]

    return coefficients


class ITO_SAE:
    def __init__(self, atoms, l0=8, cfg=None):
        self.atoms = atoms
        self.l0 = l0
        self.cfg = cfg

    def encode(self, x):
        shape = x.size()
        x = x.view(-1, shape[-1])
        activations = omp(self.atoms, x, self.l0)
        return activations.view(*shape[:-1], -1)

    def decode(self, activations):
        return torch.matmul(activations, self.atoms)

    @property
    def W_dec(self):
        return self.atoms

    def __call__(self, x):
        acts = self.encode(x)
        return self.decode(acts)

    @property
    def W_enc(self):
        # necessary for running core evals with sae bench
        return torch.zeros(
            (self.atoms.size(1), self.atoms.size(0)), device=self.atoms.device
        )

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

    def normalize_decoder(self):
        norms = torch.norm(self.atoms, dim=1)
        self.atoms /= norms[:, None]
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
