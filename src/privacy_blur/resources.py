"""Resource planning and an OOM-aware admission limit for subprocess workers."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import sys
import threading


class GPUOutOfMemory(RuntimeError):
    pass


class AdaptiveLimiter:
    def __init__(self, limit):
        self.limit = max(1, limit)
        self.active = 0
        self.condition = threading.Condition()
        self.cancelled = threading.Event()

    @contextmanager
    def slot(self):
        with self.condition:
            while self.active >= self.limit:
                if self.cancelled.is_set():
                    raise KeyboardInterrupt()
                self.condition.wait(.25)
            if self.cancelled.is_set():
                raise KeyboardInterrupt()
            self.active += 1
        try:
            yield
        finally:
            with self.condition:
                self.active -= 1
                self.condition.notify_all()

    def reduce(self):
        with self.condition:
            self.limit = max(1, self.limit // 2)
            self.condition.notify_all()
        print(f'[memory] Reducing concurrent workers to {self.limit}', flush=True)


def worker_budget(free_mb, total_mb, worker_mb, cpu_count, ram_mb, threads):
    reserve = max(1024, total_mb * .05)
    return max(1, min(int(max(0, free_mb - reserve) // max(1, worker_mb)),
                      max(1, cpu_count // threads), max(1, int(ram_mb // 2048))))


def available_ram_mb():
    try:
        for line in Path('/proc/meminfo').read_text().splitlines():
            if line.startswith('MemAvailable:'):
                return int(line.split()[1]) / 1024
    except OSError:
        pass
    return 4096


def plan_workers(weights, imgsz, device, jobs='auto', threads=2):
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1)
    ram = available_ram_mb()
    requested = None if jobs == 'auto' else int(jobs)
    probe = {'cuda': False}
    if device in ('auto', 'cuda'):
        # A disposable process releases the calibration model and CUDA context.
        result = subprocess.run([sys.executable, '-m', 'privacy_blur.resources', str(weights), str(imgsz)],
                                capture_output=True, text=True, timeout=180)
        if result.returncode:
            raise RuntimeError(f'GPU calibration failed: {result.stderr[-2000:]}')
        probe = json.loads(result.stdout.strip().splitlines()[-1])
    if probe['cuda']:
        device = 'cuda'
        count = worker_budget(probe['free_mb'], probe['total_mb'], probe['worker_mb'], cpus, ram, threads)
    elif device == 'cuda':
        raise RuntimeError('CUDA was requested but is unavailable')
    else:
        device = 'mps' if device == 'mps' else 'cpu'
        count = 1 if device == 'mps' else max(1, min(4, cpus // threads, int(ram // 2048)))
    if requested is not None:
        count = min(count, requested)
    result = {'device': device, 'workers': count, 'threads_per_worker': threads,
              'ram_available_mb': round(ram), **probe}
    print('[resources] ' + json.dumps(result), flush=True)
    return result


def probe_cuda(weights, imgsz):
    import contextlib
    import numpy as np
    import torch
    if not torch.cuda.is_available():
        return {'cuda': False}
    from ultralytics import YOLO
    torch.set_num_threads(2)
    with contextlib.redirect_stdout(sys.stderr):
        model = YOLO(weights)
        sample = np.zeros((imgsz, imgsz, 3), np.uint8)
        for _ in range(3):
            model(sample, device=0, imgsz=imgsz, conf=.35, verbose=False)
        torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    reserved = torch.cuda.max_memory_reserved()
    # Cover allocator growth, CUDA context, and libraries not tracked by PyTorch.
    per_worker = max(1024, reserved / 2**20 * 1.25 + 768)
    return {'cuda': True, 'gpu': torch.cuda.get_device_name(0), 'free_mb': round(free / 2**20),
            'total_mb': round(total / 2**20), 'worker_mb': round(per_worker)}


if __name__ == '__main__':
    print(json.dumps(probe_cuda(sys.argv[1], int(sys.argv[2]))))
