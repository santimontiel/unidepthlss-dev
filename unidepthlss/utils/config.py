"""Config-layer helpers: OmegaConf resolvers and seeding."""

from __future__ import annotations

import random

import numpy as np
import torch
from omegaconf import OmegaConf


def register_new_resolvers() -> None:
    """Register the resolvers the config tree uses beyond OmegaConf's built-ins.

    Guarded with `replace=True` so a second call (a multi-grid eval re-composing config, a
    notebook re-running a cell) does not raise on an already-registered name.
    """
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    OmegaConf.register_new_resolver("int", int, replace=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
