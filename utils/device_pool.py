# utils/device_pool.py
import torch
import itertools
import contextlib
import os
import threading
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class DevicePool:
    """Round‑robin GPU picker that *allows* many experiments per GPU."""

    def __init__(self, oversub_factor: int | None = None):
        # Initialize available devices
        # Devices list may contain GPU indices or None when using CPU
        self.devices: list[Optional[int]]
        if torch.cuda.is_available():
            self.devices = list(range(torch.cuda.device_count()))
            logger.info(f"DevicePool: Found {len(self.devices)} CUDA devices.")
        else:
            self.devices = [None]
            logger.info("DevicePool: CUDA not available – using CPU.")
        # Oversubscription factor for time-sharing devices
        self.oversub = oversub_factor or int(os.getenv("OVERSUB", "1"))
        self._counter = itertools.count()
        self._lock = threading.Lock()

    def __getstate__(self):
        # Remove unpicklable lock for pickling
        state = self.__dict__.copy()
        state.pop("_lock", None)
        return state

    def __setstate__(self, state):
        # Restore state and recreate lock
        self.__dict__.update(state)
        import threading

        self._lock = threading.Lock()

    @contextlib.contextmanager
    def acquire(self):
        """Yield a torch.device; oversubscription is allowed."""
        with self._lock:
            # Round-robin over devices with oversubscription
            total_slots = len(self.devices) * self.oversub
            idx = next(self._counter) % total_slots
            # Map slot to actual device
            if len(self.devices) == 1 and self.devices[0] is None:
                dev = None
            else:
                dev = self.devices[idx % len(self.devices)]
        orig = os.environ.get("CUDA_VISIBLE_DEVICES")
        try:
            if dev is None:
                device = torch.device("cpu")
                logger.debug("Acquired CPU device")
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(dev)
                device = torch.device("cuda:0")
                logger.debug(f"Acquired GPU {dev} as cuda:0")
            yield device
        finally:
            if dev is not None:
                if orig is None:
                    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                else:
                    os.environ["CUDA_VISIBLE_DEVICES"] = orig
            logger.debug(f"Released device {dev}")
