# utils/device_pool.py
import torch, itertools, contextlib, os, threading, logging

logger = logging.getLogger(__name__)


class DevicePool:
    """Round‑robin GPU picker that *allows* many experiments per GPU."""

    def __init__(self):
        if torch.cuda.is_available():
            self.devices = list(range(torch.cuda.device_count()))
            logger.info(f"DevicePool: Found {len(self.devices)} CUDA devices.")
        else:
            self.devices = [None]
            logger.info("DevicePool: CUDA not available – using CPU.")
        self._counter = itertools.count()
        self._lock = threading.Lock()

    @contextlib.contextmanager
    def acquire(self):
        """Yield a torch.device; oversubscription is allowed."""
        with self._lock:
            idx = next(self._counter) % len(self.devices)
            dev = self.devices[idx]
        orig = os.environ.get("CUDA_VISIBLE_DEVICES")
        try:
            if dev is None:
                device = torch.device("cpu")
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(dev)
                device = torch.device("cuda:0")
            yield device
        finally:
            if dev is not None:
                if orig is None:
                    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                else:
                    os.environ["CUDA_VISIBLE_DEVICES"] = orig
