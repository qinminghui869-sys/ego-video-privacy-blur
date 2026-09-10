# ego-video-privacy-blur

[中文](README_ZH.md)

Detect and obscure faces in ego videos with YOLO and OpenCV. The batch command processes folders recursively, isolates failures, and resumes completed work.

## Installation

Python 3.9+, FFmpeg/ffprobe with H.264 encoding, and Linux or macOS are required for batch processing.

```bash
git clone https://github.com/qinminghui869-sys/ego-video-privacy-blur.git
cd ego-video-privacy-blur
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Place a compatible YOLO face detection model in `models/`. Model weights are not included. Use face-specific weights, such as a compatible `yolov8n-face.pt`; general object-detection weights are not a substitute. Choose a PyTorch installation that supports your GPU when using CUDA.

## One-argument processing

```bash
# One video; one process, no splitting.
bash scripts/process_ego_face_blur_data.sh /data/ego/videos/left.mp4

# Recursively process a directory with automatic file-level concurrency.
bash scripts/process_ego_face_blur_data.sh /data/ego/videos

# Inspect the plan (includes a short GPU warmup, but writes no videos).
bash scripts/process_ego_face_blur_data.sh /data/ego/videos --dry-run
```

The script resolves its own project directory, uses `.venv/bin/python` when available, and does not require environment activation or a particular working directory. It defaults to `models/yolov8n-face.pt` and the project's `outputs/` directory. Override with `--face-weights` and `--output-dir`; use `PRIVACY_BLUR_PYTHON` to select another Python executable.

Default processing is **YOLOv8 face detection + Gaussian blur**, `conf=0.35`, `imgsz=960`, box expansion `1.35`, merge IoU `0.5`, and **Gaussian divisor `1`**. These retain the upstream YOLO path's detection settings while changing its Gaussian divisor from `3` to `1`. Tracking and temporal gap filling are **disabled** (`--track-gap 0`). This face-processing script does not detect plates. See the [upstream CLI](https://github.com/MengWoods/video-privacy-blur/blob/main/src/privacy_blur/cli.py).

### Automatic parallelism

`--jobs auto` warms up the actual model in a disposable GPU process, estimates per-worker VRAM, and selects a worker limit from currently free VRAM, CPU cores, and available RAM. It leaves at least 1 GiB or 5% of GPU memory as headroom. It uses the first visible CUDA GPU; select a different GPU with `CUDA_VISIBLE_DEVICES`. Existing GPU jobs are not stopped. Without CUDA, `auto` falls back to at most four CPU workers; explicit MPS uses one worker.

Each video runs in a separate process. Only different videos run concurrently; a single video is never split. `--jobs 2` caps concurrency at two (resource limits may reduce it further), and `--threads 2` limits CPU/OpenCV/PyTorch/encoder threads per worker. GPU out-of-memory failures halve the admission limit and retry within `--retries`. Memory estimates are approximate: 100% VRAM occupancy is not guaranteed or a throughput goal, since decoding, Gaussian blur, and encoding also consume CPU time.

Each worker has an independent log. Output validation, checkpoints, and Ctrl-C process-group cleanup apply to all batch jobs. Changing only `--jobs` does not invalidate successful outputs. Outputs keep relative subdirectories and use names such as `left.mp4.blurred.mp4`; the legacy `--input-dir` entry remains supported.

## Batch processing

```bash
privacy-blur-batch \
  --input-dir /data/ego/videos \
  --output-dir /data/ego/blurred \
  --face-weights models/yolov8n-face.pt \
  --device cuda \
  --method gaussian
```

For directory input, input and output directories must not overlap. Supported extensions: `.mp4`, `.mov`, `.mkv`, `.avi`, `.m4v`. Files retain their relative directories: `session/left.mp4` becomes `session/left.mp4.blurred.mp4`.

| Option | Default | Description |
| --- | --- | --- |
| `--device` | `auto` | `auto`, `cpu`, `cuda`, or `mps` |
| `--conf` | `0.35` | Detection confidence; minimum supported value is `0.25` |
| `--imgsz` | `960` | Detection input size |
| `--scale` | `1.35` | Detection box expansion factor |
| `--method` | `gaussian` | Gaussian blur (`gaussian`), synthetic mosaic (`pixelate`), or large-kernel mean blur (`egoblur`) |
| `--blur-strength` | `1` | Gaussian kernel divisor; smaller values increase blur |
| `--track-gap` | `0` | Past/future face recovery horizon in frames; `0` disables |
| `--track-scale` | `1.25` | Extra expansion for recovered masks, multiplied by `--scale` |
| `--jobs` | `auto` | Automatic concurrency or a positive upper limit |
| `--threads` | `2` | CPU threads per worker |
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

## Obscuring methods

The default is Gaussian blur (`gaussian`), as shown in the examples. Omitting `--method` also selects Gaussian blur. Pass `--method pixelate` for synthetic mosaic or `--method egoblur` for large-kernel mean filtering.

- `pixelate` completely replaces each detected, expanded region with gray tiles from a preset palette. Tile colors do not sample or average source pixels. A fixed random seed makes the pattern repeatable for the same region dimensions; changing box dimensions can still change the pattern.
- `egoblur` uses EgoBlur-style large-kernel mean filtering with an elliptical mask. The kernel is `(H//2, W//2)` from the full frame dimensions (each at least 1), not the face dimensions. Only pixels inside the ellipse are replaced; rectangle corners retain the source image. This changes obscuring only, retaining the configured detector. Colors still depend on the source, so this is not an irreversible replacement. `--scale` controls box expansion; `--blur-strength` does not affect this mode.
- `gaussian` blurs the original pixels. `--blur-strength` controls only this method and does not affect the synthetic mosaic.

The synthetic mosaic removes original pixel values inside the covered region, but does not guarantee that a person cannot be identified. Missed detections, exposed edges, mask geometry, clothing, and surrounding context can still reveal information. Review detection coverage across frames before sharing a video.

## Short-term tracking and gap filling

Face tracking and gap filling are disabled by default in both commands (`--track-gap 0`). To opt in for any obscuring method, pass `--track-gap 15 --track-scale 1.25`. Detection still runs on each original frame with the configured model and confidence threshold. Spatial box association and bounded constant-velocity prediction connect nearby detections; a lookahead buffer also propagates future detections backward to fill brief misses, including at the start of a video. This is detection-box tracking, not optical flow or identity recognition. Plate detection in the single-video command remains frame-by-frame.

When enabled at 30 FPS, 15 frames allow up to 0.5 seconds of evidence in either direction and add about 0.5 seconds of live-preview latency. At most 16 full frames are queued (roughly 100 MB at 1920×1088 RGB, plus processing overhead). All tail frames are flushed at end of input. `--track-gap 0` restores independent per-frame processing.

Recovered boxes receive an additional 1.25× expansion: with `--scale 1.35`, the total is 1.6875×. They use full rectangular masks, including with `egoblur`; directly detected boxes retain the selected method's normal mask. Predictions expire without a new detection. A thumbnail-difference heuristic resets tracking at obvious scene cuts, but does not detect every cut.

This reduces short intermittent misses, at the cost of extra background masking. It cannot recover faces never detected within the window; fast motion, crossings, long occlusion, and subtle scene changes can still cause errors. Review the output. Batch reports distinguish direct `boxes` from `recovered_boxes`, and tracking configuration participates in resume validation.

Enable gap filling explicitly:

```bash
bash scripts/process_ego_face_blur_data.sh /data/ego/videos \
  --track-gap 15 --track-scale 1.25 --output-dir outputs/tracked
```

## Single video

Use `privacy-blur` for a single video with an explicit output filename or an interactive preview:

```bash
privacy-blur --input input.mp4 --output blurred.mp4 \
  --no-blur-plates --face-detector yolo \
  --face-yolo-weights models/yolov8n-face.pt \
  --method gaussian
```

The single-video command writes MP4 without audio and overwrites an existing output at the same path. For checkpoints and full output validation, use the processing script with a video path instead. Plate detection requires a separate compatible model.

## Tests

```bash
python -m unittest discover -s tests -v
```

Tests use synthetic videos and do not download model weights. Models, input/output videos, environments, logs, and local experiments are excluded from Git.

Obscuring affects detected and temporally recovered regions. Missed faces and identifying details elsewhere remain possible; output validation checks file integrity, not anonymity or detection coverage.

## License and provenance

The original project is [MengWoods/video-privacy-blur](https://github.com/MengWoods/video-privacy-blur). See [LICENSE](LICENSE) and [NOTICE.md](NOTICE.md). Dependencies and model weights have their own licenses.
