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


def process_file(source, output, config, log, lock_fd=None):
    request = output.with_suffix('.request.json')
    report = output.with_suffix('.report.json')
    report.unlink(missing_ok=True)
    write_json(request, {'source': str(source), 'output': str(output), 'config': config})
    command = [sys.executable, '-m', 'privacy_blur.worker', str(request)]
    with log.open('a', encoding='utf-8') as stream:
        stream.write(f'\nAttempt started {time.strftime("%Y-%m-%dT%H:%M:%S%z")}\n')
        stream.flush()
        worker = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
                                  pass_fds=() if lock_fd is None else (lock_fd,))
        try:
            code = worker.wait(timeout=config.get('timeout') or None)
        except BaseException:
            os.killpg(worker.pid, signal.SIGTERM)
            try:
                worker.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(worker.pid, signal.SIGKILL)
                worker.wait()
            raise
    if code:
        raise RuntimeError(f'Worker exited with code {code}; see {log}')
    result = json.loads(report.read_text())
    if result.get('status') != 'complete' or result.get('frames', 0) <= 0:
        raise RuntimeError('Worker did not produce a valid completion report')
    return result


def run_batch(input_dir, output_dir, config, retries=1, processor=None):
    """Resume matching successes; retry failed/interrupted files from frame zero."""
    source_root = Path(input_dir).resolve(strict=True)
    destination = Path(output_dir).resolve()
    if not source_root.is_dir():
        raise ValueError('Input must be a directory')
    if destination == source_root or source_root in destination.parents or destination in source_root.parents:
        raise ValueError('Input and output directories must not overlap')
    if retries < 0:
        raise ValueError('Retries must be non-negative')
    videos = sorted(p for p in source_root.rglob('*') if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES)
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
    counts = {'completed': 0, 'skipped': 0, 'failed': 0}
    failures = []
    with directory_lock(state / 'lock') as lock_fd:
        for source in videos:
            relative = source.relative_to(source_root)
            key = hashlib.sha256(str(relative).encode()).hexdigest()
            record_path = state / 'jobs' / (key + '.json')
            output = destination / relative.parent / (relative.name + '.blurred.mp4')
            if destination not in output.resolve().parents or output.is_symlink():
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
                    continue
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
                        details = (processor(source, temporary, config, log) if processor else
                                   process_file(source, temporary, config, log, lock_fd=lock_fd))
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
        write_json(state / 'summary.json', {'counts': counts, 'failures': failures})
    return counts


def positive_float(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError('Must be finite and positive')
    return number


def main():
    parser = argparse.ArgumentParser(description='Batch face blur with file-level recovery (Linux/macOS).')
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--face-weights', required=True, help='Local YOLO face weights')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda', 'mps'])
    parser.add_argument('--conf', type=positive_float, default=.35)
    parser.add_argument('--imgsz', type=int, default=960)
    parser.add_argument('--scale', type=positive_float, default=1.35)
    parser.add_argument('--blur-strength', type=positive_float, default=.5)
    parser.add_argument('--method', choices=['gaussian', 'pixelate'], default='gaussian')
    parser.add_argument('--retries', type=int, default=1, help='Additional attempts per failed file')
    parser.add_argument('--timeout', type=positive_float, help='Maximum seconds per attempt; default unlimited')
    args = parser.parse_args()
    if not .25 <= args.conf <= 1 or args.imgsz <= 0 or args.scale < 1 or args.retries < 0:
        parser.error('Require 0.25 <= conf <= 1, imgsz > 0, scale >= 1, retries >= 0')
    import shutil
    for binary in ['ffmpeg', 'ffprobe']:
        if not shutil.which(binary):
            parser.error(f'{binary} is required')
    weights = Path(args.face_weights).resolve()
    if not weights.is_file():
        parser.error(f'Weights not found: {weights}')
    config = {'face_weights': str(weights), 'weights_sha256': sha256(weights), 'device': args.device,
              'conf': args.conf, 'imgsz': args.imgsz, 'scale': args.scale,
              'blur_strength': args.blur_strength, 'method': args.method, 'timeout': args.timeout}
    # Runtime changes can affect detection or rendering and must invalidate cached success.
    import importlib.metadata
    config['versions'] = {name: importlib.metadata.version(name) for name in ['ultralytics', 'numpy', 'opencv-python', 'torch', 'torchvision']}
    config['python'] = sys.version
    config['ffmpeg'] = subprocess.check_output(['ffmpeg', '-version'], text=True).splitlines()[0]
    def interrupt(signum, frame):
        raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM, interrupt)
    try:
        result = run_batch(args.input_dir, args.output_dir, config, retries=args.retries)
    except KeyboardInterrupt:
        print('Interrupted. Re-run the same command to resume.', file=sys.stderr)
        return 130
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(2, f'{exc}\n')
    print(json.dumps(result))
    return 1 if result['failed'] else 0


if __name__ == '__main__':
    sys.exit(main())
