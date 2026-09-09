# video-privacy-blur

[中文](README_ZH.md)

Detect and obscure faces in ego videos with YOLO and OpenCV. The batch command processes folders recursively, isolates failures, and resumes completed work.

## Installation

Python 3.9+, FFmpeg/ffprobe with H.264 encoding, and Linux or macOS are required for batch processing.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Place a compatible YOLO face detection model in `models/`. Model weights are not included. Use face-specific weights, such as a compatible `yolov8n-face.pt`; general object-detection weights are not a substitute. Choose a PyTorch installation that supports your GPU when using CUDA.

## Batch processing

```bash
privacy-blur-batch \
  --input-dir /data/ego/videos \
  --output-dir /data/ego/blurred \
  --face-weights models/yolov8n-face.pt \
  --device cuda
```

Input and output directories must not overlap. Supported extensions: `.mp4`, `.mov`, `.mkv`, `.avi`, `.m4v`. Files retain their relative directories: `session/left.mp4` becomes `session/left.mp4.blurred.mp4`.

| Option | Default | Description |
| --- | --- | --- |
| `--device` | `auto` | `auto`, `cpu`, `cuda`, or `mps` |
| `--conf` | `0.35` | Detection confidence; minimum supported value is `0.25` |
| `--imgsz` | `960` | Detection input size |
| `--scale` | `1.35` | Detection box expansion factor |
| `--method` | `gaussian` | `gaussian` or `pixelate` |
| `--blur-strength` | `0.5` | Gaussian kernel divisor; smaller values increase blur |
| `--retries` | `1` | Additional attempts for each failed file |
| `--timeout` | unlimited | Maximum seconds per attempt, including validation |

### Recovery and validation

Run the same command again to resume. Completed files are skipped only when input content, configuration, model, source code, runtime package versions, and output checksum match. Failed or interrupted files restart from the beginning. Changing settings rebuilds affected outputs.

Each video runs in a separate process. A failure does not stop the remaining files. Outputs are written to temporary files, fully decoded and checked for frame count, dimensions, and frame rate, then atomically moved into place. Unmanaged existing files are never overwritten. One batch may write to an output directory at a time.

State is stored under the output directory:

- `.batch/jobs/`: per-file status, checksums, settings, and processing results.
- `.batch/logs/`: worker logs.
- `.batch/summary.json`: counts and failures from the latest completed run.
- `.batch/previous/`: the most recent superseded output for each rebuilt file.

Keep `.batch/` for recovery. Superseded outputs and diagnostic files consume disk space and may be removed when no longer needed; keep job records for skipping successful work. Exit codes: `0` all successful, `1` file failures, `2` setup error, `130` keyboard interruption.

Batch output is H.264 MP4 without audio or copied source metadata. Frame dimensions and constant frame intervals are preserved; timestamps start at zero. Variable frame timing, rotation metadata, and odd frame dimensions are rejected rather than silently transformed. Absolute timestamps and sensor synchronization metadata must be managed separately.

## Single video

The original single-video command remains available. Recovery and full output validation apply only to the batch command.

```bash
privacy-blur --input input.mp4 --output blurred.mp4 \
  --no-blur-plates --face-detector yolo \
  --face-yolo-weights models/yolov8n-face.pt
```

## Tests

```bash
python -m unittest discover -s tests -v
```

Tests use synthetic videos and do not download model weights. Models, input/output videos, environments, logs, and local experiments are excluded from Git.

Blur only affects detected regions. Missed faces and identifying details elsewhere remain possible; output validation checks file integrity, not anonymity or detection coverage.

## License and provenance

The original project is [MengWoods/video-privacy-blur](https://github.com/MengWoods/video-privacy-blur). See [LICENSE](LICENSE) and [NOTICE.md](NOTICE.md). Dependencies and model weights have their own licenses.
