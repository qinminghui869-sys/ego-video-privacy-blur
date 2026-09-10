import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np


class TemporalIntegrationTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.source = self.root / 'source.mp4'
        writer = cv2.VideoWriter(str(self.source), cv2.VideoWriter_fourcc(*'mp4v'), 10, (160,100))
        frame = np.zeros((100,160,3), np.uint8)
        frame[30:60,30:45] = 255
        for _ in range(8):
            writer.write(frame)
        writer.release()
        self.detections = [[], [], [(30,30,60,60)], [], [], [], [], []]

    def assert_output(self, path):
        cap = cv2.VideoCapture(str(path))
        count = 0
        first = None
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if first is None:
                first = frame
            count += 1
        cap.release()
        self.assertEqual(count, 8, 'lookahead must flush the video tail')
        self.assertGreater(int(first[40,50,0]), 20, 'future detection must mask opening frames')

    def test_worker_uses_temporal_recovery_and_flushes(self):
        from privacy_blur.worker import render
        from privacy_blur.batch import sha256
        weights = self.root / 'weights.pt'
        weights.write_bytes(b'test-model')
        config = dict(face_weights=str(weights), weights_sha256=sha256(weights), device='cpu',
                      conf=.35, imgsz=960, scale=1.35, blur_strength=.5, method='egoblur',
                      track_gap=3, track_scale=1.25)
        output = self.root / 'worker.mp4'
        with patch('privacy_blur.detectors.FaceDetectorYOLO') as model:
            model.return_value.side_effect = self.detections
            result = render(self.source, output, config)
        self.assertGreater(result.get('recovered_boxes', 0), 0)
        self.assert_output(output)

    def test_cli_uses_temporal_recovery_and_flushes(self):
        from privacy_blur.cli import main
        output = self.root / 'cli.mp4'
        argv = ['privacy-blur', '--input', str(self.source), '--output', str(output),
                '--face-detector', 'yolo', '--no-blur-plates', '--device', 'cpu',
                '--method', 'egoblur', '--track-gap', '3', '--track-scale', '1.25']
        with patch('sys.argv', argv), patch('privacy_blur.cli.FaceDetectorYOLO') as model:
            model.return_value.side_effect = self.detections
            main()
        self.assert_output(output)
