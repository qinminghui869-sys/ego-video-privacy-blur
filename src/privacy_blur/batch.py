"""Resumable, file-isolated video processing."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
import signal
from pathlib import Path
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from .resources import AdaptiveLimiter, GPUOutOfMemory

VIDEO_SUFFIXES = {'.mp4', '.mov', '.mkv', '.avi', '.m4v'}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


@contextmanager
def directory_lock(path):
    import fcntl
    with path.open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another batch is using this output directory') from exc
        try:
            yield stream.fileno()
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def process_file(source, output, config, log, lock_fd=None, cancelled=None):
    request = output.with_suffix('.request.json')
    report = output.with_suffix('.report.json')
    report.unlink(missing_ok=True)
    write_json(request, {'source': str(source), 'output': str(output), 'config': config})
    command = [sys.executable, '-m', 'privacy_blur.worker', str(request)]
    with log.open('a', encoding='utf-8') as stream:
        stream.write(f'\nAttempt started {time.strftime("%Y-%m-%dT%H:%M:%S%z")}\n')
        stream.flush()
        env = os.environ.copy()
        env.update(OMP_NUM_THREADS=str(config.get('threads', 2)),
                   MKL_NUM_THREADS=str(config.get('threads', 2)), OPENBLAS_NUM_THREADS=str(config.get('threads', 2)))
        worker = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
                                  pass_fds=() if lock_fd is None else (lock_fd,))
        try:
            deadline = time.monotonic() + config['timeout'] if config.get('timeout') else None
            while True:
                if cancelled is not None and cancelled.is_set():
                    raise KeyboardInterrupt()
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError('Video processing timed out')
                try:
                    code = worker.wait(timeout=.25)
                    break
                except subprocess.TimeoutExpired:
                    pass
        except BaseException:
            try:
                os.killpg(worker.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                worker.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(worker.pid, signal.SIGKILL)
                worker.wait()
            raise
    if code == 75:
        raise GPUOutOfMemory(f'CUDA out of memory; see {log}')
    if code:
        raise RuntimeError(f'Worker exited with code {code}; see {log}')
    result = json.loads(report.read_text())
    if result.get('status') != 'complete' or result.get('frames', 0) <= 0:
        raise RuntimeError('Worker did not produce a valid completion report')
    return result


def run_batch(input_dir, output_dir, config, retries=1, processor=None, jobs=1):
    """Resume matching successes; retry failed/interrupted files from frame zero."""
    source_path = Path(input_dir)
    if source_path.is_symlink():
        raise ValueError('Symbolic-link inputs are not supported')
    source_path = source_path.resolve(strict=True)
    single = source_path.is_file()
    source_root = source_path.parent if single else source_path
    destination = Path(output_dir).resolve()
    if not source_root.is_dir():
        raise ValueError('Input must be a directory')
    if not single and (destination == source_root or source_root in destination.parents or destination in source_root.parents):
        raise ValueError('Input and output directories must not overlap')
    if retries < 0 or jobs < 1:
        raise ValueError('Retries must be non-negative and jobs must be positive')
    videos = ([source_path] if source_path.suffix.lower() in VIDEO_SUFFIXES else []) if single else sorted(
        p for p in source_root.rglob('*') if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES)
    if not videos:
        raise ValueError('No supported videos found')
    if any(p.is_symlink() or source_root not in p.resolve().parents for p in videos):
        raise ValueError('Symbolic-link inputs are not supported')
    destination.mkdir(parents=True, exist_ok=True)
    state = destination / '.batch'
    state.mkdir(exist_ok=True)
    for name in ['jobs', 'logs', 'work', 'previous']:
        (state / name).mkdir(exist_ok=True)
    code_hash = hashlib.sha256()
    for module in sorted(Path(__file__).parent.glob('*.py')):
        code_hash.update(module.name.encode())
        code_hash.update(module.read_bytes())
    signature = hashlib.sha256(json.dumps({'config': config, 'code': code_hash.hexdigest()}, sort_keys=True).encode()).hexdigest()
    limiter = AdaptiveLimiter(jobs)

    def run_one(source, lock_fd):
        counts = {'completed': 0, 'skipped': 0, 'failed': 0}
        failures = []
        relative = source.relative_to(source_root)
        key = hashlib.sha256(str(relative).encode()).hexdigest()
        record_path = state / 'jobs' / (key + '.json')
        output = destination / relative.parent / (relative.name + '.blurred.mp4')
        if destination not in output.resolve().parents or output.is_symlink() or output.resolve() == source:
            raise ValueError(f'Output path escapes directory or is a symlink: {output}')
        temporary = state / 'work' / (key + '.mp4')
        log = state / 'logs' / (key + '.log')
        record = {}
        try:
            if record_path.exists():
                record = json.loads(record_path.read_text())
                if not isinstance(record, dict):
                    raise ValueError('Invalid job record')
            input_hash = sha256(source)
            owned = record.get('source') == str(source) and record.get('output') == str(output)
            if (owned and record.get('status') == 'complete' and record.get('input_sha256') == input_hash
                    and record.get('signature') == signature and output.is_file()
                    and record.get('output_sha256') == sha256(output)):
                counts['skipped'] += 1
                print(f'[skip] {relative}', flush=True)
                return counts, failures
            if output.exists():
                if not owned:
                    raise FileExistsError(f'Refusing to replace unmanaged output: {output}')
                os.replace(output, state / 'previous' / (key + '.mp4'))
            record = {'source': str(source), 'output': str(output), 'input_sha256': input_hash,
                      'signature': signature, 'config': config, 'status': 'pending',
                      'attempts': record.get('attempts', 0), 'log': str(log)}
            for attempt in range(retries + 1):
                temporary.unlink(missing_ok=True)
                record.update(status='running', attempts=record['attempts'] + 1, error=None)
                write_json(record_path, record)
                print(f'[run {attempt + 1}/{retries + 1}] {relative}', flush=True)
                try:
                    with limiter.slot():
                        details = (processor(source, temporary, config, log) if processor else
                                   process_file(source, temporary, config, log, lock_fd=lock_fd, cancelled=limiter.cancelled))
                    if not temporary.is_file() or temporary.stat().st_size == 0:
                        raise RuntimeError('No output produced')
                    if sha256(source) != input_hash:
                        raise RuntimeError('Input changed while processing')
                    output_hash = sha256(temporary)
                    output.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(temporary, output)
                    record.update(status='complete', output_sha256=output_hash, details=details)
                    write_json(record_path, record)
                    counts['completed'] += 1
                    break
                except KeyboardInterrupt:
                    record.update(status='interrupted', error='Interrupted by user')
                    write_json(record_path, record)
                    raise
                except Exception as exc:
                    if isinstance(exc, GPUOutOfMemory):
                        limiter.reduce()
                    record.update(status='failed', error=f'{type(exc).__name__}: {exc}')
                    write_json(record_path, record)
                    if attempt == retries:
                        counts['failed'] += 1
                        failures.append({'source': str(source), 'error': record['error'], 'log': str(log)})
                        print(f'[failed] {relative}: {record["error"]}', flush=True)
                finally:
                    temporary.unlink(missing_ok=True)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            counts['failed'] += 1
            failures.append({'source': str(source), 'error': f'{type(exc).__name__}: {exc}'})
            print(f'[failed] {relative}: {exc}', flush=True)
        return counts, failures

    counts = {'completed': 0, 'skipped': 0, 'failed': 0}
    failures = []
    with directory_lock(state / 'lock') as lock_fd:
        executor = ThreadPoolExecutor(max_workers=min(jobs, len(videos)))
        futures = []
        try:
            for source in videos:
                futures.append(executor.submit(run_one, source, lock_fd))
            for future in as_completed(futures):
                partial, errors = future.result()
                for name, value in partial.items():
                    counts[name] += value
                failures.extend(errors)
        except BaseException:
            limiter.cancelled.set()
            for future in futures:
                future.cancel()
            raise
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
        write_json(state / 'summary.json', {'counts': counts, 'failures': failures})
    return counts


def positive_float(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError('Must be finite and positive')
    return number


def main():
    from .temporal import add_temporal_arguments, validate_temporal_arguments
    parser = argparse.ArgumentParser(description='Batch face blur with file-level recovery (Linux/macOS).')
    parser.add_argument('input', nargs='?', help='Video file or directory (recursive)')
    parser.add_argument('--input-dir', help='Legacy alias for input')
    project = Path(__file__).resolve().parents[2]
    parser.add_argument('--output-dir', default=str(project / 'outputs'))
    parser.add_argument('--face-weights', default=str(project / 'models/yolov8n-face.pt'), help='Local YOLOv8 face weights')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda', 'mps'])
    parser.add_argument('--conf', type=positive_float, default=.35)
    parser.add_argument('--imgsz', type=int, default=960)
    parser.add_argument('--scale', type=positive_float, default=1.35)
    parser.add_argument('--blur-strength', type=positive_float, default=1.)
    parser.add_argument('--method', choices=['gaussian', 'pixelate', 'egoblur'], default='gaussian')
    parser.add_argument('--retries', type=int, default=1, help='Additional attempts per failed file')
    parser.add_argument('--timeout', type=positive_float, help='Maximum seconds per attempt; default unlimited')
    parser.add_argument('--jobs', default='auto', help='auto or maximum concurrent video workers')
    parser.add_argument('--threads', type=int, default=2, help='CPU threads per worker')
    parser.add_argument('--dry-run', action='store_true', help='Show resource plan without processing videos')
    add_temporal_arguments(parser)
    args = parser.parse_args()
    validate_temporal_arguments(parser, args)
    if bool(args.input) == bool(args.input_dir):
        parser.error('Pass one video/directory, or --input-dir, not both')
    input_path = Path(args.input or args.input_dir).expanduser()
    if not input_path.exists():
        parser.error(f'Input not found: {input_path}')
    video_count = (int(input_path.suffix.lower() in VIDEO_SUFFIXES) if input_path.is_file() else
                   sum(1 for p in input_path.rglob('*') if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES))
    if not video_count:
        parser.error('No supported videos found')
    if args.threads < 1 or (args.jobs != 'auto' and (not args.jobs.isdigit() or int(args.jobs) < 1)):
        parser.error('Require jobs=auto or a positive integer, threads >= 1')
    if not .25 <= args.conf <= 1 or args.imgsz <= 0 or args.scale < 1 or args.retries < 0:
        parser.error('Require 0.25 <= conf <= 1, imgsz > 0, scale >= 1, retries >= 0')
    import shutil
    for binary in ['ffmpeg', 'ffprobe']:
        if not shutil.which(binary):
            parser.error(f'{binary} is required')
    weights = Path(args.face_weights).resolve()
    if not weights.is_file():
        parser.error(f'Weights not found: {weights}')
    from .resources import plan_workers
    try:
        plan = plan_workers(weights, args.imgsz, args.device, args.jobs, args.threads)
    except (ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        parser.exit(2, f'{exc}\n')
    plan['capacity'] = plan['workers']
    plan['workers'] = min(plan['workers'], video_count)
    print(f"[plan] {video_count} video(s), up to {plan['workers']} concurrent file worker(s); no splitting", flush=True)
    if args.dry_run:
        print(json.dumps({'input': str(input_path.resolve()), 'output_dir': args.output_dir,
                          'method': args.method, 'blur_strength': args.blur_strength,
                          'track_gap': args.track_gap, 'conf': args.conf, 'imgsz': args.imgsz,
                          'scale': args.scale, 'plan': plan}))
        return 0
    config = {'face_weights': str(weights), 'weights_sha256': sha256(weights), 'device': plan['device'], 'threads': args.threads,
              'conf': args.conf, 'imgsz': args.imgsz, 'scale': args.scale,
              'blur_strength': args.blur_strength, 'method': args.method, 'timeout': args.timeout,
              'track_gap': args.track_gap, 'track_scale': args.track_scale}
    # Runtime changes can affect detection or rendering and must invalidate cached success.
    import importlib.metadata
    config['versions'] = {name: importlib.metadata.version(name) for name in ['ultralytics', 'numpy', 'opencv-python', 'torch', 'torchvision']}
    config['python'] = sys.version
    config['ffmpeg'] = subprocess.check_output(['ffmpeg', '-version'], text=True).splitlines()[0]
    def interrupt(signum, frame):
        raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM, interrupt)
    try:
        result = run_batch(input_path, args.output_dir, config, retries=args.retries, jobs=plan['workers'])
    except KeyboardInterrupt:
        print('Interrupted. Re-run the same command to resume.', file=sys.stderr)
        return 130
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(2, f'{exc}\n')
    print(json.dumps(result))
    return 1 if result['failed'] else 0


if __name__ == '__main__':
    sys.exit(main())
