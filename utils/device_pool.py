# utils/device_pool.py
import torch
import queue
import contextlib
import os
import threading
import logging

logger = logging.getLogger(__name__)


class DevicePool:
    """Manages a pool of available torch devices (CUDA GPUs or CPU)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._q = queue.Queue()
        # Determine available CUDA devices, default to [None] for CPU
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            self.devices = list(range(num_gpus))
            logger.info(f"DevicePool: Found {num_gpus} CUDA devices.")
        else:
            self.devices = [None]  # Represents CPU
            logger.info("DevicePool: No CUDA devices found, using CPU.")
        # Populate the queue
        for d_idx in self.devices:
            self._q.put(d_idx)

    @contextlib.contextmanager
    def acquire(self) -> torch.device:
        """Acquires a device (GPU index or None for CPU)."""
        device_idx = None
        acquired_device = None
        original_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        try:
            with self._lock:
                if self._q.empty():
                    raise RuntimeError("DevicePool is empty, cannot acquire device.")
                device_idx = self._q.get()
            # Set environment for this device
            if device_idx is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(device_idx)
                acquired_device = torch.device("cuda:0")
                logger.debug(f"Acquired GPU {device_idx} as cuda:0")
            else:
                acquired_device = torch.device("cpu")
                logger.debug("Acquired CPU device")
            yield acquired_device
        finally:
            # Restore environment
            if device_idx is not None:
                if original_visible is not None:
                    os.environ["CUDA_VISIBLE_DEVICES"] = original_visible
                else:
                    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            # Return to pool
            with self._lock:
                self._q.put(device_idx)
            logger.debug(f"Returned device {device_idx} to pool")
