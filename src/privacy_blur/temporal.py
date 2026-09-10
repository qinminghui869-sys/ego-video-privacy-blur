"""Bounded, bidirectional detection-box tracking for short privacy-mask gaps."""
from collections import deque
from dataclasses import dataclass
import math

import cv2
import numpy as np

from .utils import iou, union_box


def add_temporal_arguments(parser):
    parser.add_argument('--track-gap', type=int, default=0,
                        help='Maximum past/future frames for face gap filling; 0 disables (default: 0)')
    parser.add_argument('--track-scale', type=float, default=1.25,
                        help='Extra expansion for recovered masks, multiplied by --scale (default: 1.25)')


def validate_temporal_arguments(parser, args):
    if args.track_gap < 0 or not math.isfinite(args.track_scale) or args.track_scale < 1:
        parser.error('Require track-gap >= 0 and finite track-scale >= 1')


@dataclass
class _Track:
    box: np.ndarray
    last_detection: np.ndarray
    velocity: np.ndarray
    age: int = 0


class _Tracker:
    def __init__(self, gap):
        self.gap = gap
        self.tracks = []

    def update(self, boxes):
        boxes = [np.asarray(b, dtype=float) for b in boxes]
        predicted = [t.box + t.velocity for t in self.tracks]
        # Greedy one-to-one spatial association, strongest overlap first.
        candidates = sorted(((iou(p, b), ti, bi) for ti, p in enumerate(predicted)
                             for bi, b in enumerate(boxes)), reverse=True)
        matched_tracks, matched_boxes = set(), set()
        updated = []
        for overlap, ti, bi in candidates:
            if overlap < .15:
                break
            if ti in matched_tracks or bi in matched_boxes:
                continue
            t, b = self.tracks[ti], boxes[bi]
            displacement = ((b[:2] + b[2:]) -
                            (t.last_detection[:2] + t.last_detection[2:])) / (2 * (t.age + 1))
            # Limit unstable extrapolation to a quarter of the box size per frame.
            limit = np.maximum(1., (b[2:] - b[:2]) * .25)
            displacement = np.clip(displacement, -limit, limit)
            updated.append(_Track(b, b, np.tile(displacement, 2)))
            matched_tracks.add(ti)
            matched_boxes.add(bi)
        for ti, t in enumerate(self.tracks):
            if ti not in matched_tracks and t.age < self.gap:
                updated.append(_Track(predicted[ti], t.last_detection, t.velocity, t.age + 1))
        for bi, b in enumerate(boxes):
            if bi not in matched_boxes:
                updated.append(_Track(b, b, np.zeros(4)))
        self.tracks = updated
        return [(t.box, t.age > 0) for t in updated]


def _regions(detections, forward, backward, width, height):
    recovered = []
    for box, inferred in forward + backward:
        if not inferred or any(iou(box, actual) >= .15 for actual in detections):
            continue
        # Both directions may predict the same missing face. Cover their union.
        for i, existing in enumerate(recovered):
            if iou(box, existing) >= .15:
                recovered[i] = union_box(box, existing)
                break
        else:
            recovered.append(box)
    result = []
    for box, inferred in ([(b, False) for b in detections] + [(b, True) for b in recovered]):
        x1, y1, x2, y2 = box
        clipped = (max(0, math.floor(x1)), max(0, math.floor(y1)),
                   min(width, math.ceil(x2)), min(height, math.ceil(y2)))
        if clipped[2] > clipped[0] and clipped[3] > clipped[1]:
            result.append((clipped, inferred))
    return result


def temporal_frames(frames, detect, gap=0):
    """Yield (original frame, [(box, recovered)]) once per input frame, in order.

    Retains at most gap+1 full frames. Backward evidence never refreshes forward
    track age. Scene-cut resets use a conservative thumbnail-difference heuristic.
    This is box-motion tracking, not identity recognition or optical flow.
    """
    if gap < 0:
        raise ValueError('gap must be nonnegative')
    if gap == 0:
        for frame in frames:
            yield frame, [(b, False) for b in detect(frame)]
        return
    pending = deque()
    forward = _Tracker(gap)
    previous = None

    def emit():
        backward = _Tracker(gap)
        reverse_regions = []
        for _, boxes in reversed(pending):
            reverse_regions = backward.update(boxes)
        frame, boxes = pending.popleft()
        return frame, _regions(boxes, forward.update(boxes), reverse_regions,
                               frame.shape[1], frame.shape[0])

    for frame in frames:
        thumb = cv2.resize(frame, (64, 36), interpolation=cv2.INTER_AREA).astype(np.float32)
        cut = previous is not None and float(np.abs(thumb - previous).mean()) > 65
        if cut:
            while pending:
                yield emit()
            forward = _Tracker(gap)
        previous = thumb
        pending.append((frame, detect(frame)))
        if len(pending) > gap:
            yield emit()
    while pending:
        yield emit()
