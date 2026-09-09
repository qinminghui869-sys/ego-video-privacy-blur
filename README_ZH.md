# video-privacy-blur

[English](README.md)

使用 YOLO 和 OpenCV 检测、遮挡 ego 视频中的人脸。批处理支持递归目录、失败隔离和文件级恢复。

## 安装

批处理需要 Python 3.9+、支持 H.264 编码的 FFmpeg/ffprobe，以及 Linux 或 macOS。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

将兼容的 YOLO 人脸检测权重放入 `models/`。仓库不包含权重。应使用兼容的 `yolov8n-face.pt` 等人脸专用模型，不能用通用目标检测权重替代。使用 CUDA 时，请安装与 GPU 环境匹配的 PyTorch。

## 批处理

```bash
privacy-blur-batch \
  --input-dir /data/ego/videos \
  --output-dir /data/ego/blurred \
  --face-weights models/yolov8n-face.pt \
  --device cuda
```

输入、输出目录不能相同或相互包含。支持 `.mp4`、`.mov`、`.mkv`、`.avi`、`.m4v`。保留相对目录，例如 `session/left.mp4` 输出为 `session/left.mp4.blurred.mp4`。

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--device` | `auto` | `auto`、`cpu`、`cuda` 或 `mps` |
| `--conf` | `0.35` | 检测置信度，支持的最小值为 `0.25` |
| `--imgsz` | `960` | 检测输入尺寸 |
| `--scale` | `1.35` | 检测框扩展倍数 |
| `--method` | `gaussian` | 高斯模糊 `gaussian` 或像素化 `pixelate` |
| `--blur-strength` | `0.5` | 高斯核除数，越小越模糊 |
| `--retries` | `1` | 每个失败文件的额外重试次数 |
| `--timeout` | 不限 | 每次尝试的最长秒数，包含校验时间 |

### 恢复与校验

重新执行相同命令即可恢复。只有输入内容、配置、模型、源码、运行依赖版本和输出校验值一致的成功文件才会跳过。失败或中断的文件从头处理；修改参数会重建受影响的结果。

每个视频使用独立进程，单文件失败后继续其他文件。输出先写临时文件，完整解码并核对帧数、尺寸和帧率后，再原子替换到最终位置。不会覆盖没有任务记录的已有文件。同一输出目录同时只允许一个批处理任务。

输出目录中的运行记录：

- `.batch/jobs/`：各文件状态、校验值、参数和处理结果。
- `.batch/logs/`：处理日志。
- `.batch/summary.json`：最近一次完整运行的统计和失败列表。
- `.batch/previous/`：每个重建文件最近一次被替换的旧输出。

恢复时需要保留 `.batch/`。旧输出和诊断文件会占用磁盘，不再需要时可清理；跳过已成功任务仍需保留任务记录。退出码：`0` 全部成功，`1` 存在文件失败，`2` 配置错误，`130` 键盘中断。

批处理输出为 H.264 MP4，不含音轨，不复制源元数据。保留图像尺寸和恒定帧间隔，时间戳从零开始。变帧率、旋转元数据和奇数尺寸的视频会报错，不会静默转换。绝对时间戳及传感器同步信息需另外管理。

## 单个视频

保留原有单视频命令。恢复和完整输出校验仅适用于批处理命令。

```bash
privacy-blur --input input.mp4 --output blurred.mp4 \
  --no-blur-plates --face-detector yolo \
  --face-yolo-weights models/yolov8n-face.pt
```

## 测试

```bash
python -m unittest discover -s tests -v
```

测试使用合成视频，不下载模型权重。模型、输入输出视频、环境、日志及本地实验文件均排除在 Git 之外。

遮挡仅作用于检测到的区域，仍可能存在漏脸及其他身份线索。输出校验验证文件完整性，不代表检测覆盖率或匿名化通过验收。

## 许可与来源

基于 [MengWoods/video-privacy-blur](https://github.com/MengWoods/video-privacy-blur)。见 [LICENSE](LICENSE) 和 [NOTICE.md](NOTICE.md)。依赖库与模型权重遵循各自的许可。
