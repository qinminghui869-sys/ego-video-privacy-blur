import cv2
import numpy as np

DEFAULT_BLUR_STRENGTH = 1.0  # Stronger smoothing; not an anonymity guarantee.

def gaussian_inplace(frame, x1,y1,x2,y2, strength_divisor=DEFAULT_BLUR_STRENGTH):
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0: return
    kw = max(3, int((x2-x1)/strength_divisor) | 1)
    kh = max(3, int((y2-y1)/strength_divisor) | 1)
    frame[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (kw, kh), 0)

def pixelate_inplace(frame, x1,y1,x2,y2, blocks: int = 12):
    """Overwrite the ROI with synthetic mosaic tiles independent of its pixels."""
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0: return
    h, w = roi.shape[:2]
    fx = max(1, w // blocks)
    fy = max(1, h // blocks)
    # A fixed seed avoids random flicker; no source colors enter the pattern.
    rng = np.random.default_rng(0)
    palette = np.array([64, 96, 128, 160, 192], dtype=roi.dtype)
    small = rng.choice(palette, size=(max(1,h//fy), max(1,w//fx)))
    if roi.ndim == 3:
        small = np.repeat(small[:, :, None], roi.shape[2], axis=2)
    frame[y1:y2, x1:x2] = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def egoblur_inplace(frame, x1,y1,x2,y2, ellipse=True):
    """Apply EgoBlur-style large-kernel mean filtering inside an ellipse.

    Set ellipse=False for full rectangular coverage of recovered regions.
    Colors still depend on source pixels; this is not synthetic replacement.
    """
    height, width = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    if x2 <= x1 or y2 <= y1: return
    roi = frame[y1:y2, x1:x2]
    # Preserve EgoBlur's kernel order: (frame height / 2, frame width / 2).
    kernel = (max(1, height // 2), max(1, width // 2))
    blurred = cv2.blur(roi, kernel)
    if not ellipse:
        roi[:] = blurred
        return
    h, w = roi.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, ((w // 2, h // 2), (w, h), 0), 255, -1)
    roi[mask > 0] = blurred[mask > 0]
