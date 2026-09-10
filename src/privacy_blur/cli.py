import argparse, cv2, sys
from .detectors import PlateDetectorYOLO, FaceDetectorHaar, FaceDetectorYOLO
from .blur_ops import DEFAULT_BLUR_STRENGTH, gaussian_inplace, pixelate_inplace, egoblur_inplace
from .utils import choose_device, expand_box, nms_merge
from .temporal import temporal_frames, add_temporal_arguments, validate_temporal_arguments

def main():
    ap = argparse.ArgumentParser("Blur license plates and/or faces in video")
    ap.add_argument("--input", required=True, help="video path or camera index, e.g. 0")
    ap.add_argument("--output", default="out_blurred.mp4", help="output video path")
    ap.add_argument("--device", default="auto", choices=["auto","cpu","cuda","mps"], help="inference device")
    ap.add_argument("--imgsz", type=int, default=960, help="YOLO inference size")
    ap.add_argument("--conf", type=float, default=0.35, help="YOLO confidence threshold")
    ap.add_argument("--scale", type=float, default=1.35, help="bbox expansion factor")
    ap.add_argument("--show", action="store_true", help="preview window")
    ap.add_argument("--blur-plates", action="store_true", default=True, help="blur license plates")
    ap.add_argument("--no-blur-plates", dest="blur_plates", action="store_false")
    ap.add_argument("--blur-faces", action="store_true", default=True, help="blur faces")
    ap.add_argument("--no-blur-faces", dest="blur_faces", action="store_false")
    ap.add_argument("--face-detector", choices=["haar","yolo"], default="yolo", help="face backend")
    ap.add_argument("--face-yolo-weights", default="models/yolov8n-face.pt", help="YOLO face weights (file or hub id)")
    ap.add_argument("--method", choices=["gaussian","pixelate","egoblur"], default="gaussian", help="blur method")
    ap.add_argument("--plate-weights", default="models/license_plate_detector.pt", help="Path to YOLO license plate model (.pt)")
    ap.add_argument("--blur-strength", type=float, default=DEFAULT_BLUR_STRENGTH, help="Lower = stronger blur (bbox divisor; default: 1). Does not guarantee anonymization.")
    add_temporal_arguments(ap)
    args = ap.parse_args()
    validate_temporal_arguments(ap, args)

    device = choose_device(args.device)
    print(f"[INFO] Using device: {device}")

    cap = cv2.VideoCapture(int(args.input) if args.input.isdigit() else args.input)
    if not cap.isOpened():
        sys.exit(f"Could not open input: {args.input}")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, fps, (W, H))

    plate_detector = PlateDetectorYOLO(args.plate_weights, device, args.conf, args.imgsz) \
                 if args.blur_plates else None

    if args.blur_faces:
        face_detector = FaceDetectorHaar() if args.face_detector=="haar" else FaceDetectorYOLO(
            args.face_yolo_weights, device, args.conf, args.imgsz
        )
    else:
        face_detector = None

    def input_frames():
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame

    def detect_faces(frame):
        return nms_merge(face_detector(frame), iou_thresh=0.5) if face_detector else []

    try:
        for frame, regions in temporal_frames(input_frames(), detect_faces, args.track_gap):
            if plate_detector:
                regions += [(box, False) for box in nms_merge(plate_detector(frame), iou_thresh=0.5)]
            for box, recovered in regions:
                scale = args.scale * (args.track_scale if recovered else 1.)
                x1,y1,x2,y2 = expand_box(*box, scale, W, H)
                if args.method == "gaussian":
                    gaussian_inplace(frame, x1,y1,x2,y2, args.blur_strength)
                elif args.method == "egoblur":
                    egoblur_inplace(frame, x1,y1,x2,y2, ellipse=not recovered)
                else:
                    pixelate_inplace(frame, x1,y1,x2,y2)

            writer.write(frame)
            if args.show:
                cv2.imshow("privacy-blur", frame)
                if cv2.waitKey(1) & 0xFF == 27: break
    finally:
        cap.release()
        writer.release()
        if args.show: cv2.destroyAllWindows()

    print(f"[OK] Saved: {args.output}")
