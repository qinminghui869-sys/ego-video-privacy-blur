# ego-video-privacy-blur

[English](README.md)

使用 YOLO 和 OpenCV 检测、遮挡 ego 视频中的人脸。批处理支持递归目录、失败隔离和文件级恢复。

## 安装

批处理需要 Python 3.9+、支持 H.264 编码的 FFmpeg/ffprobe，以及 Linux 或 macOS。

```bash
git clone https://github.com/qinminghui869-sys/ego-video-privacy-blur.git
cd ego-video-privacy-blur
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

将兼容的 YOLO 人脸检测权重放入 `models/`。仓库不包含权重。应使用兼容的 `yolov8n-face.pt` 等人脸专用模型，不能用通用目标检测权重替代。使用 CUDA 时，请安装与 GPU 环境匹配的 PyTorch。

## 只传入视频或目录

```bash
# 单个视频：一个进程完整处理，不分段。
bash scripts/process_ego_face_blur_data.sh /data/ego/videos/left.mp4

# 目录：递归查找视频，按文件自动并行。
bash scripts/process_ego_face_blur_data.sh /data/ego/videos

# 只查看计划（会短暂预热 GPU，不生成视频）。
bash scripts/process_ego_face_blur_data.sh /data/ego/videos --dry-run
```

脚本自动定位项目目录，优先使用 `.venv/bin/python`，无需激活环境，也不依赖当前工作目录。默认读取 `models/yolov8n-face.pt`，输出到项目的 `outputs/`。可以通过 `--face-weights`、`--output-dir` 覆盖路径，通过 `PRIVACY_BLUR_PYTHON` 指定 Python。

默认配置为 **YOLOv8 人脸检测 + 高斯模糊**：`conf=0.35`、`imgsz=960`、检测框扩展 `1.35`、合并 IoU `0.5`，**高斯模糊强度除数为 `1`**。保留上游 YOLO 路径的检测配置，将原高斯除数从 `3` 改为 `1`。短时跟踪和前后帧补漏默认**关闭**（`--track-gap 0`）。此人脸处理脚本不检测车牌。参见[上游 CLI 源码](https://github.com/MengWoods/video-privacy-blur/blob/main/src/privacy_blur/cli.py)。

### 自动多进程并行

`--jobs auto` 使用实际模型在独立 GPU 进程中预热、估算每个进程的显存需求，再结合当前空闲显存、CPU 核数和可用内存计算并发上限。保留至少 1 GiB 或显存总量 5% 的余量。使用第一张可见 CUDA GPU，可通过 `CUDA_VISIBLE_DEVICES` 选择其他 GPU；不会停止已有 GPU 任务。无 CUDA 时，`auto` 回退到最多 4 个 CPU 进程；显式选择 MPS 时使用单进程。

每个视频使用独立进程，**仅多个视频按文件并行，单视频不分段**。`--jobs 2` 将并发上限限制为 2（资源限制可能进一步降低），`--threads 2` 限制每个进程的 CPU/OpenCV/PyTorch/编码线程数。发生 CUDA 显存不足时，将并发准入上限减半，并在 `--retries` 范围内重试。显存用量为估算值，不能保证占满 100%；解码、高斯模糊和编码也消耗 CPU，显存占用率本身不等于处理速度。

每个进程使用独立日志，所有批处理任务均支持输出校验、断点恢复和 Ctrl-C 进程组清理。仅修改 `--jobs` 不会让已完成的输出失效。输出保留相对目录，例如 `left.mp4.blurred.mp4`；原 `--input-dir` 用法继续兼容。

## 批处理

```bash
privacy-blur-batch \
  --input-dir /data/ego/videos \
  --output-dir /data/ego/blurred \
  --face-weights models/yolov8n-face.pt \
  --device cuda \
  --method gaussian
```

目录输入时，输入、输出目录不能相同或相互包含。支持 `.mp4`、`.mov`、`.mkv`、`.avi`、`.m4v`。保留相对目录，例如 `session/left.mp4` 输出为 `session/left.mp4.blurred.mp4`。

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--device` | `auto` | `auto`、`cpu`、`cuda` 或 `mps` |
| `--conf` | `0.35` | 检测置信度，支持的最小值为 `0.25` |
| `--imgsz` | `960` | 检测输入尺寸 |
| `--scale` | `1.35` | 检测框扩展倍数 |
| `--method` | `gaussian` | 高斯模糊 `gaussian`、合成马赛克 `pixelate` 或大核均值滤波 `egoblur` |
| `--blur-strength` | `1` | 高斯核除数，越小越模糊 |
| `--track-gap` | `0` | 向前、向后补漏的最大帧数，`0` 关闭 |
| `--track-scale` | `1.25` | 补漏框额外扩展倍数，与 `--scale` 相乘 |
| `--jobs` | `auto` | 自动并发或指定正整数上限 |
| `--threads` | `2` | 每个进程的 CPU 线程数 |
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

## 遮挡方式

默认采用高斯模糊（`gaussian`），下方示例也使用此方式。不传 `--method` 时同样使用高斯模糊；如需合成马赛克或大核均值滤波，可分别指定 `--method pixelate` 或 `--method egoblur`。

- `pixelate`：用预设灰色调色板生成色块，完全替换检测并扩展后的区域。色块颜色不采样原始像素，也不计算原区域平均值。固定随机种子使相同区域尺寸产生相同图案；检测框尺寸变化时，图案仍可能变化。
- `egoblur`：复用 EgoBlur 的大核均值滤波与椭圆遮罩，滤波核按整帧尺寸设为 `(H//2, W//2)`（各维最小为 1），不按人脸区域尺寸计算。仅椭圆内部被替换，矩形四角保持原图。该选项只改变打码方式，人脸检测仍使用当前配置的模型；颜色仍依赖原图，不提供不可逆保证。`--scale` 控制检测框扩展，`--blur-strength` 不影响此模式。
- `gaussian`：对原始像素做高斯模糊。`--blur-strength` 仅控制此模式，不影响合成马赛克。

合成马赛克移除了覆盖区域内的原始像素值，但不保证人物无法被识别。漏检、未覆盖的边缘、遮罩形状、衣着及周围环境仍可能泄露信息。分享视频前，应检查各帧的检测覆盖情况。

## 短时跟踪与前后帧补漏

单视频和批处理默认关闭短时跟踪与人脸补漏（`--track-gap 0`）。如需手动启用，可指定 `--track-gap 15 --track-scale 1.25`，适用于三种打码方式。每一原始帧仍使用配置的模型和置信度执行检测；通过检测框空间关联和受限匀速预测连接相邻检测，并缓存后续帧，利用未来检测向前补上短暂漏检，包括视频开头。这是检测框运动跟踪，不是光流或身份识别。单视频命令中的车牌检测仍按帧独立处理。

手动启用后，30 FPS 下的 15 帧对应前后各最多 0.5 秒的证据窗口，实时预览会增加约 0.5 秒延迟。队列最多保留 16 个完整帧（1920×1088 RGB 约 100 MB，另有处理开销），视频结束时会输出所有剩余帧。设置 `--track-gap 0` 可恢复逐帧独立处理。

补漏框额外扩大 1.25 倍：当 `--scale 1.35` 时，总扩展倍数为 1.6875。补漏区域采用完整矩形遮挡，包括 `egoblur` 模式；直接检测到的区域保持所选方式的常规遮罩。没有新检测时，预测会到期失效。通过缩略图差异启发式检测明显切镜并清空跟踪，但不能保证识别所有切镜。

该机制减少短时漏检，代价是可能多遮挡部分背景。窗口内从未检出的人脸无法补回；快速运动、人物交叉、长时间遮挡和不明显的场景切换仍可能出错，需要检查成片。批处理报告分别记录直接检测框 `boxes` 和补漏框 `recovered_boxes`，跟踪参数也参与恢复校验。

手动启用补漏：

```bash
bash scripts/process_ego_face_blur_data.sh /data/ego/videos \
  --track-gap 15 --track-scale 1.25 --output-dir outputs/tracked
```

## 单个视频

需要指定输出文件名或交互预览时，可以使用 `privacy-blur`：

```bash
privacy-blur --input input.mp4 --output blurred.mp4 \
  --no-blur-plates --face-detector yolo \
  --face-yolo-weights models/yolov8n-face.pt \
  --method gaussian
```

单视频命令输出不含音频的 MP4，并会覆盖同路径的已有输出。如需断点恢复和完整输出校验，请使用处理脚本传入视频路径。车牌检测需要另备兼容的模型。

## 测试

```bash
python -m unittest discover -s tests -v
```

测试使用合成视频，不下载模型权重。模型、输入输出视频、环境、日志及本地实验文件均排除在 Git 之外。

遮挡作用于直接检测及跨帧补漏区域，仍可能存在漏脸及其他身份线索。输出校验验证文件完整性，不代表检测覆盖率或匿名化通过验收。

## 许可与来源

基于 [MengWoods/video-privacy-blur](https://github.com/MengWoods/video-privacy-blur)。见 [LICENSE](LICENSE) 和 [NOTICE.md](NOTICE.md)。依赖库与模型权重遵循各自的许可。
