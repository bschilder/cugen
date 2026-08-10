"""Sampled peak-GPU-memory measurement, shared by every benchmark script.

Why not pool.total_bytes() after the run: cugen.ld calls free_all_blocks()
inside its own tile loop, so the pool shrinks mid-run and its final size is not
a high-water mark. The symptom is unmistakable once you look for it -- "peak"
at p=20,000 reading LOWER than at p=2,000. Sample used_bytes() on a thread.

This lives in one place on purpose: the first version of this fix was applied
to one benchmark and not the other, and the un-fixed copy silently reported
bad numbers.
"""
import threading
import time

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:                                            # noqa: BLE001
    cp = None
    HAS_CUPY = False


class PeakSampler:
    """Context manager recording max live device allocation during a block."""

    def __init__(self, interval=0.002):
        self.interval = interval
        self.peak = 0
        self._stop = False
        self._t = None

    def __enter__(self):
        if HAS_CUPY:
            cp.get_default_memory_pool().free_all_blocks()
            self._t = threading.Thread(target=self._run, daemon=True)
            self._t.start()
        return self

    def _run(self):
        pool = cp.get_default_memory_pool()
        while not self._stop:
            used = pool.used_bytes()
            if used > self.peak:
                self.peak = used
            time.sleep(self.interval)

    def __exit__(self, *exc):
        self._stop = True
        if HAS_CUPY:
            if self._t is not None:
                self._t.join(timeout=1.0)
            self.peak = max(self.peak, cp.get_default_memory_pool().used_bytes())

    @property
    def peak_gib(self):
        return round(self.peak / 2**30, 4)
