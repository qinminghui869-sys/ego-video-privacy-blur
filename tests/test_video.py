import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest


class VideoTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.video = self.root / 'sample.mp4'
        subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'testsrc2=size=64x48:rate=10:duration=1',
                        '-an', '-c:v', 'libx264', '-threads', '1', str(self.video)], check=True)

    def worker(self):
        self.assertIsNotNone(importlib.util.find_spec('privacy_blur.worker'), 'video worker missing')
        from privacy_blur import worker
        return worker

    def test_full_decode_and_timing(self):
        result = self.worker().inspect_video(self.video)
        self.assertEqual(result['frames'], 10)
        self.assertEqual((result['width'], result['height']), (64, 48))
        self.assertAlmostEqual(result['fps'], 10)

    def test_corrupt_video_rejected(self):
        self.video.write_bytes(b'not video')
        with self.assertRaises(RuntimeError):
            self.worker().inspect_video(self.video)

    def test_variable_timing_rejected(self):
        irregular = self.root / 'vfr.mp4'
        subprocess.run(['ffmpeg', '-v', 'error', '-i', str(self.video), '-vf', "select='not(eq(n,4))'",
                        '-vsync', 'vfr', '-c:v', 'libx264', '-threads', '1', str(irregular)], check=True)
        with self.assertRaisesRegex(ValueError, 'Variable'):
            self.worker().inspect_video(irregular)


if __name__ == '__main__':
    unittest.main()
