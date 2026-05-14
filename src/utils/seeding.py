"""Global seeding utility – call seed_everything(N) at the top of every script."""

import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import transformers
        transformers.set_seed(seed)
    except ImportError:
        pass
