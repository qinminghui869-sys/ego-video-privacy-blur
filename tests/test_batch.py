import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.inputs = self.root / 'input'
        self.outputs = self.root / 'output'
        self.inputs.mkdir()
        for name in ['a.mp4', 'b.mp4']:
            (self.inputs / name).write_bytes(name.encode())
        self.calls = []

    def run_batch(self, **kwargs):
        self.assertIsNotNone(importlib.util.find_spec('privacy_blur.batch'), 'batch runner is missing')
        from privacy_blur.batch import run_batch
        return run_batch(self.inputs, self.outputs, {'strength': .5}, processor=kwargs.pop('processor', self.process), **kwargs)

    def process(self, source, output, config, log):
        self.calls.append(source.name)
        output.write_bytes(b'validated-output:' + source.read_bytes())
        return {'frames': 1}

    def test_failure_does_not_stop_next_file_and_is_retried(self):
        def fail_first(source, output, config, log):
            if source.name == 'a.mp4':
                output.write_bytes(b'partial')
                raise RuntimeError('decoder failure')
            return self.process(source, output, config, log)
        result = self.run_batch(processor=fail_first, retries=0)
        self.assertEqual(result, {'completed': 1, 'skipped': 0, 'failed': 1})
        self.assertFalse((self.outputs / 'a.mp4.blurred.mp4').exists())
        self.calls.clear()
        self.assertEqual(self.run_batch(retries=0), {'completed': 1, 'skipped': 1, 'failed': 0})
        self.assertEqual(self.calls, ['a.mp4'])

    def test_resume_verifies_content_and_rebuilds_changed_input(self):
        self.run_batch()
        self.calls.clear()
        self.assertEqual(self.run_batch()['skipped'], 2)
        self.assertEqual(self.calls, [])
        (self.inputs / 'a.mp4').write_bytes(b'new input')
        self.assertEqual(self.run_batch()['completed'], 1)
        self.assertEqual(self.calls, ['a.mp4'])

    def test_corrupt_output_is_rebuilt(self):
        self.run_batch()
        (self.outputs / 'a.mp4.blurred.mp4').write_bytes(b'corrupt')
        self.calls.clear()
        self.assertEqual(self.run_batch()['completed'], 1)
        self.assertEqual(self.calls, ['a.mp4'])

    def test_retry_and_nested_names(self):
        (self.inputs / 'sub').mkdir()
        (self.inputs / 'sub/a.mp4').write_bytes(b'other')
        attempts = []
        def transient(source, output, config, log):
            attempts.append(str(source))
            if len(attempts) == 1:
                raise RuntimeError('temporary failure')
            return self.process(source, output, config, log)
        self.assertEqual(self.run_batch(processor=transient, retries=1)['completed'], 3)
        self.assertEqual(len(attempts), 4)
        self.assertTrue((self.outputs / 'sub/a.mp4.blurred.mp4').exists())

    def test_interrupt_is_not_success_and_resumes(self):
        def interrupt(source, output, config, log):
            output.write_bytes(b'partial')
            raise KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            self.run_batch(processor=interrupt)
        self.assertFalse((self.outputs / 'a.mp4.blurred.mp4').exists())
        self.assertEqual(self.run_batch()['completed'], 2)

    def test_unmanaged_output_is_not_overwritten(self):
        self.outputs.mkdir()
        existing = self.outputs / 'a.mp4.blurred.mp4'
        existing.write_bytes(b'keep me')
        self.assertEqual(self.run_batch()['failed'], 1)
        self.assertEqual(existing.read_bytes(), b'keep me')

    def test_output_within_input_rejected(self):
        self.outputs = self.inputs / 'outputs'
        with self.assertRaises(ValueError):
            self.run_batch()

    def test_config_change_rebuilds(self):
        self.run_batch()
        from privacy_blur.batch import run_batch
        self.calls.clear()
        result = run_batch(self.inputs, self.outputs, {'strength': 1.}, processor=self.process)
        self.assertEqual(result['completed'], 2)

    def test_concurrent_run_is_rejected(self):
        from privacy_blur.batch import run_batch
        def nested(source, output, config, log):
            with self.assertRaises(RuntimeError):
                run_batch(self.inputs, self.outputs, {'strength': .5}, processor=self.process)
            return self.process(source, output, config, log)
        self.run_batch(processor=nested)


if __name__ == '__main__':
    unittest.main()
