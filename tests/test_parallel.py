import argparse
import importlib.util
import tempfile
import threading
import time
import unittest
from pathlib import Path


class ParallelTests(unittest.TestCase):
    def test_tracking_disabled_by_default(self):
        from privacy_blur.temporal import add_temporal_arguments
        parser = argparse.ArgumentParser()
        add_temporal_arguments(parser)
        self.assertEqual(parser.parse_args([]).track_gap, 0)

    def test_default_gaussian_strength(self):
        from privacy_blur.blur_ops import DEFAULT_BLUR_STRENGTH
        self.assertEqual(DEFAULT_BLUR_STRENGTH, 1.)

    def test_single_video_input(self):
        from privacy_blur.batch import run_batch
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); source=root/'one.mp4'; source.write_bytes(b'input')
            def process(source,output,config,log):
                output.write_bytes(b'output'); return {'frames':1}
            result=run_batch(source,root/'out',{},processor=process)
            self.assertEqual(result['completed'],1)
            self.assertTrue((root/'out/one.mp4.blurred.mp4').is_file())

    def test_two_jobs_really_overlap(self):
        from privacy_blur.batch import run_batch
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); source=root/'in'; source.mkdir()
            for name in ['a.mp4','b.mp4']:(source/name).write_bytes(b'input')
            barrier=threading.Barrier(2)
            def process(source,output,config,log):
                barrier.wait(timeout=3)
                output.write_bytes(b'output'); return {'frames':1}
            result=run_batch(source,root/'out',{},processor=process,jobs=2)
            self.assertEqual(result,{'completed':2,'skipped':0,'failed':0})

    def resources(self):
        self.assertIsNotNone(importlib.util.find_spec('privacy_blur.resources'))
        from privacy_blur import resources
        return resources

    def test_gpu_budget_reserves_headroom(self):
        plan=self.resources().worker_budget(free_mb=12000,total_mb=16000,worker_mb=2000,
                                            cpu_count=32,ram_mb=64000,threads=2)
        self.assertEqual(plan,5)

    def test_cpu_and_ram_cap_concurrency(self):
        self.assertEqual(self.resources().worker_budget(24000,24000,1000,4,3000,2),1)

    def test_oom_reduces_limiter(self):
        limiter=self.resources().AdaptiveLimiter(4)
        limiter.reduce()
        self.assertEqual(limiter.limit,2)
        limiter.reduce();limiter.reduce()
        self.assertEqual(limiter.limit,1)


class RetryTests(unittest.TestCase):
    def test_oom_retries_and_other_files_continue(self):
        from privacy_blur.batch import run_batch
        from privacy_blur.resources import GPUOutOfMemory
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); source=root/'in'; source.mkdir()
            for name in ['a.mp4','b.mp4']:(source/name).write_bytes(b'input')
            attempted=set()
            def process(source,output,config,log):
                if source.name=='a.mp4' and source.name not in attempted:
                    attempted.add(source.name)
                    raise GPUOutOfMemory('simulated CUDA out of memory')
                output.write_bytes(b'output'); return {'frames':1}
            result=run_batch(source,root/'out',{},processor=process,jobs=2)
            self.assertEqual(result,{'completed':2,'skipped':0,'failed':0})
            again=run_batch(source,root/'out',{},processor=process,jobs=1)
            self.assertEqual(again['skipped'],2)
