import torch
from typing import Optional, Union


def get_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    """
    Returns a torch.device object.

    Args:
        device (str or torch.device, optional): device name or object. If None, automatically chooses.

    Returns:
        torch.device: 'cuda' if available else 'cpu', or the provided device.
    """
    if device is not None:
        return torch.device(device) if isinstance(device, str) else device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
