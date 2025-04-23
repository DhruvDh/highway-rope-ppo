# utils/reproducibility.py
import random
import numpy as np
import torch

# Constants for reproducibility
SEED = 42


def set_random_seeds(seed=SEED, exact_reproducibility=False):
    """Set random seeds for reproducibility across all relevant libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Only enforce deterministic behavior if exact reproducibility needed
    if exact_reproducibility:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # Allow cuDNN to benchmark and select fastest algorithms
        torch.backends.cudnn.benchmark = True


def get_device():
    """Determine and return the best available compute device."""
    if torch.cuda.is_available():
        return torch.device("cuda"), f"GPU: {torch.cuda.get_device_name(0)}"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps"), "Apple Silicon GPU"
    else:
        return torch.device("cpu"), "CPU"
