import random
import numpy as np
import torch
from typing import Optional


def set_seed(seed: Optional[int] = 42, deterministic: bool = True) -> None:
    """
    Sets random seed for reproducibility.

    Args:
        seed (int, optional): Seed value. Defaults to 42.
        deterministic (bool): If True, sets PyTorch to deterministic mode. Defaults to True.
    """
    if seed is None:
        return

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
