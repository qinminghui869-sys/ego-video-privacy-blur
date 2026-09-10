import importlib.util
import unittest
import numpy as np


class TemporalTests(unittest.TestCase):
    def resolve(self, boxes, gap=3, frames=None):
        self.assertIsNotNone(importlib.util.find_spec('privacy_blur.temporal'),
                             'temporal gap filling is missing')
        from privacy_blur.temporal import temporal_frames
        if frames is None:
            frames = [np.full((100, 160, 3), 50, np.uint8) for _ in boxes]
        detections = iter(boxes)
        return list(temporal_frames(iter(frames), lambda _: next(detections), gap))

    def test_fills_gap_from_both_directions(self):
        outputs = self.resolve([[], [(30, 30, 50, 50)], [], [], [(36, 30, 56, 50)], []])
        self.assertEqual(len(outputs), 6)
        self.assertTrue(all(regions for _, regions in outputs))
        self.assertTrue(outputs[0][1][0][1])  # recovered from a future frame
        self.assertFalse(outputs[1][1][0][1])  # actual detection

    def test_predictions_expire_without_new_detection(self):
        outputs = self.resolve([[(30, 30, 50, 50)]] + [[]] * 7, gap=2)
        self.assertTrue(outputs[2][1])
        self.assertTrue(all(not regions for _, regions in outputs[3:]))

    def test_disabled_preserves_detections_and_frame_order(self):
        frames = [np.full((100, 160, 3), i, np.uint8) for i in range(4)]
        boxes = [[], [(30, 30, 50, 50)], [], []]
        outputs = self.resolve(boxes, gap=0, frames=frames)
        self.assertEqual([int(frame[0, 0, 0]) for frame, _ in outputs], list(range(4)))
        self.assertEqual([len(regions) for _, regions in outputs], [0, 1, 0, 0])

    def test_scene_cut_blocks_forward_and_backward_propagation(self):
        frames = [np.full((100,160,3), v, np.uint8) for v in [0,0,255,255]]
        outputs = self.resolve([[(30,30,50,50)], [], [], [(90,30,110,50)]], frames=frames)
        self.assertTrue(all(box[0] < 60 for box, _ in outputs[1][1]))
        self.assertTrue(all(box[0] > 60 for box, _ in outputs[2][1]))

    def test_short_motion_is_followed(self):
        outputs = self.resolve([[(30,30,50,50)], [(34,30,54,50)], [], []], gap=2)
        self.assertGreater(outputs[2][1][0][0][0], 34)
        self.assertGreater(outputs[3][1][0][0][0], outputs[2][1][0][0][0])

    def test_multiple_people_do_not_collapse(self):
        outputs = self.resolve([[(10,30,30,50),(100,30,120,50)], []], gap=2)
        self.assertEqual(len(outputs[1][1]), 2)

    def test_empty_video_and_no_detections(self):
        self.assertEqual(self.resolve([]), [])
        self.assertTrue(all(not regions for _, regions in self.resolve([[],[],[]])))

    def test_supplemental_egoblur_covers_rectangle_corners(self):
        from privacy_blur.blur_ops import egoblur_inplace
        import inspect
        self.assertIn('ellipse', inspect.signature(egoblur_inplace).parameters)
        frame = np.zeros((100,160,3), np.uint8)
        frame[30:50,30:50] = 255
        frame[30,30] = 0
        egoblur_inplace(frame,30,30,50,50,ellipse=False)
        self.assertGreater(int(frame[30,30,0]), 0)
        self.assertEqual(int(frame[0,0,0]), 0)


if __name__ == '__main__':
    unittest.main()
