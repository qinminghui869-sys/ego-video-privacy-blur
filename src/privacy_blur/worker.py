"""One video per process, with full input/output decode validation."""
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile


def inspect_video(path):
    metadata = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                               '-show_streams',
                               '-of', 'json', str(path)], capture_output=True, text=True)
    if metadata.returncode or metadata.stderr.strip():
        raise RuntimeError(f'Cannot probe video: {metadata.stderr.strip()}')
    streams = json.loads(metadata.stdout).get('streams', [])
    if not streams:
        raise RuntimeError('No video stream')
    info = streams[0]
    numerator, denominator = map(int, info['r_frame_rate'].split('/'))
    if denominator == 0 or numerator <= 0:
        raise ValueError('Invalid frame rate')
    fps = numerator / denominator
    if int(info.get('tags', {}).get('rotate', 0)) % 360 or any(
            float(item.get('rotation', 0)) % 360 for item in info.get('side_data_list', [])):
        raise ValueError('Rotation metadata is unsupported; normalize orientation first')
    width, height = info['width'], info['height']
    if width % 2 or height % 2:
        raise ValueError('H.264 output requires even frame dimensions')
    command = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_frames',
               '-show_entries', 'frame=best_effort_timestamp_time', '-of', 'default=noprint_wrappers=1', str(path)]
    frames = 0
    previous = None
    step = 1 / fps
    with tempfile.TemporaryFile(mode='w+') as errors:
        probe = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors, text=True)
        try:
            for line in probe.stdout:
                if not line.startswith('best_effort_timestamp_time='):
                    continue
                value = line.strip().split('=', 1)[1]
                timestamp = float(value)
                if not math.isfinite(timestamp):
                    raise ValueError('Invalid frame timestamp')
                if previous is not None and abs(timestamp - previous - step) > max(.0001, step * .01):
                    raise ValueError('Variable frame timing is unsupported; use a timestamp-preserving pipeline')
                previous = timestamp
                frames += 1
            code = probe.wait()
            errors.seek(0)
            message = errors.read()
            if code or message.strip() or not frames:
                raise RuntimeError(f'Video decode failed: {message.strip()}')
        finally:
            if probe.poll() is None:
                probe.kill()
                probe.wait()
            probe.stdout.close()
    return {'width': width, 'height': height, 'frames': frames, 'fps': fps,
            'rate': f'{numerator}/{denominator}'}


def render(source, output, config):
    import cv2
    import numpy as np
    from .batch import sha256
    from .blur_ops import gaussian_inplace, pixelate_inplace, egoblur_inplace
    from .detectors import FaceDetectorYOLO
    from .utils import choose_device, expand_box, nms_merge
    from .temporal import temporal_frames

    source_info = inspect_video(source)
    if sha256(config['face_weights']) != config['weights_sha256']:
        raise RuntimeError('Model weights changed after batch started')
    threads = config.get('threads', 2)
    cv2.setNumThreads(threads)
    import torch
    torch.set_num_threads(threads)
    detector = FaceDetectorYOLO(config['face_weights'], choose_device(config['device']),
                                config['conf'], config['imgsz'])
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError('Cannot open source video')
    width, height = source_info['width'], source_info['height']
    encoder = None
    frames = boxes_count = recovered_count = 0

    def input_frames():
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame.shape != (height, width, 3):
                raise RuntimeError('Unexpected decoded frame dimensions')
            yield frame

    def detect_faces(frame):
        return nms_merge(detector(frame), iou_thresh=.5)

    try:
        encoder = subprocess.Popen(['ffmpeg', '-v', 'error', '-n', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
                                    '-s', f'{width}x{height}', '-r', source_info['rate'], '-i', 'pipe:0',
                                    '-an', '-map_metadata', '-1', '-c:v', 'libx264', '-preset', 'fast',
                                    '-threads', str(threads), '-crf', '18', '-pix_fmt', 'yuv420p',
                                    '-movflags', '+faststart', str(output)], stdin=subprocess.PIPE)
        for frame, regions in temporal_frames(input_frames(), detect_faces, config.get('track_gap', 0)):
            for box, recovered in regions:
                scale = config['scale'] * (config.get('track_scale', 1.25) if recovered else 1.)
                expanded = expand_box(*box, scale, width, height)
                if config['method'] == 'gaussian':
                    gaussian_inplace(frame, *expanded, config['blur_strength'])
                elif config['method'] == 'egoblur':
                    egoblur_inplace(frame, *expanded, ellipse=not recovered)
                else:
                    pixelate_inplace(frame, *expanded)
            encoder.stdin.write(np.ascontiguousarray(frame).tobytes())
            frames += 1
            boxes_count += sum(not recovered for _, recovered in regions)
            recovered_count += sum(recovered for _, recovered in regions)
            if frames % 300 == 0:
                print(f'{frames}/{source_info["frames"]} frames', flush=True)
        encoder.stdin.close()
        if encoder.wait() != 0:
            raise RuntimeError('FFmpeg encoding failed')
    finally:
        cap.release()
        if encoder is not None:
            if encoder.poll() is None:
                encoder.kill()
            if not encoder.stdin.closed:
                encoder.stdin.close()
            encoder.wait()
    if frames != source_info['frames']:
        raise RuntimeError(f'Frame count mismatch: {frames} != {source_info["frames"]}')
    validated = inspect_video(output)
    if validated != source_info:
        raise RuntimeError(f'Output geometry or timing mismatch: {validated} != {source_info}')
    return {'status': 'complete', **validated, 'boxes': boxes_count, 'recovered_boxes': recovered_count, 'audio': False}


def main():
    from .batch import write_json
    request = json.loads(Path(sys.argv[1]).read_text())
    output = Path(request['output'])
    try:
        result = render(Path(request['source']), output, request['config'])
    except RuntimeError as exc:
        if 'out of memory' in str(exc).lower() and 'cuda' in str(exc).lower():
            print(str(exc), file=sys.stderr)
            sys.exit(75)
        raise
    write_json(output.with_suffix('.report.json'), result)


if __name__ == '__main__':
    main()
