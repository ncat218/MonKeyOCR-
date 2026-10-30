"""Small standard-library helpers shared by controller and GPU workers."""
import hashlib
import json
import os
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def utc_stamp():
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')


def stop_signals():
    return (signal.SIGTERM, signal.SIGINT) + ((signal.SIGUSR1,) if hasattr(signal, 'SIGUSR1') else ())


def canonical(obj):
    return (json.dumps(obj, sort_keys=True, ensure_ascii=False, allow_nan=False,
                       separators=(',', ':')) + '\n').encode('utf-8')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def identity(obj):
    return hashlib.sha256(canonical(obj)).hexdigest()


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.writing')
    temp.write_bytes(canonical(obj))
    os.replace(temp, path)


def locked_json(path, obj):
    path = Path(path)
    if path.exists() and path.read_bytes() != canonical(obj):
        raise RuntimeError(f'Frozen artifact differs: {path}. Use a new run_name.')
    save_json(path, obj)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def jsonl(path):
    with Path(path).open(encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.writing')
    with temp.open('wb') as f:
        for record in records:
            f.write(canonical(record))
    os.replace(temp, path)


def resume_jsonl(path):
    """Only repair an interrupted final line; never discard a corrupt middle row."""
    path = Path(path)
    if not path.exists():
        return []
    good, offset = [], 0
    with path.open('rb') as f:
        data = f.read()
    lines = data.splitlines(keepends=True)
    for index, line in enumerate(lines):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise RuntimeError(f'Corrupt middle of prediction file: {path}')
            path.with_name(path.name + '.' + utc_stamp() + '.interrupted_tail').write_bytes(line)
            with path.open('r+b') as f:
                f.truncate(offset)
            return good
        good.append(record)
        offset += len(line)
    if data and not data.endswith(b'\n'):
        with path.open('ab') as f:
            f.write(b'\n')
    return good


class Processes:
    def __init__(self, log_dir, env=None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.env = dict(os.environ if env is None else env)
        self.active = None
        self.stop_requested = False

    def request_stop(self, signum, frame):
        self.stop_requested = True
        if self.active is not None and self.active.poll() is None:
            self.active.send_signal(signum)

    def run(self, command, label, cwd=None, env=None):
        if self.stop_requested:
            raise InterruptedError('Stop requested; run the same command to resume.')
        command = [str(x) for x in command]
        log = self.log_dir / f'{utc_stamp()}__{label}.log'
        print('START', label, '| log:', log, flush=True)
        started = time.monotonic()
        with log.open('w', encoding='utf-8') as sink:
            sink.write(json.dumps(dict(command=command, cwd=str(cwd)), ensure_ascii=False) + '\n')
            sink.flush()
            self.active = subprocess.Popen(command, cwd=cwd, env=env or self.env,
                                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                           text=True, errors='replace', bufsize=1)
            try:
                for line in self.active.stdout:
                    sink.write(line)
                    sink.flush()
                    print(line, end='', flush=True)
                code = self.active.wait()
            finally:
                # Keyboard interrupts must not leave a GPU worker training in the background.
                if self.active.poll() is None:
                    self.active.terminate()
                    try:
                        self.active.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        self.active.kill()
                        self.active.wait()
                self.active = None
        if code:
            raise RuntimeError(f'{label} exited {code}; inspect {log}. Partial outputs are not scored.')
        print('DONE', label, 'seconds:', round(time.monotonic() - started, 1), flush=True)
        return log


class StageState:
    def __init__(self, root, run_identity):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        locked_json(self.root / 'immutable_run_identity.json', run_identity)

    def execute(self, name, inputs, action):
        marker = self.root / f'{name}__stage_status.json'
        fingerprint = identity(inputs)
        if marker.exists():
            prior = read_json(marker)
            if prior['input_fingerprint'] != fingerprint:
                raise RuntimeError(f'{name}: inputs/code changed. Use a new run_name, do not mix experiments.')
            if prior['status'] == 'completed':
                for filename, digest in prior['artifacts'].items():
                    p = Path(filename)
                    if not p.is_file() or sha256(p) != digest:
                        raise RuntimeError(f'{name}: completed artifact missing/modified: {p}')
                print('RESUME verified completed stage:', name, flush=True)
                return prior['result']
        started = utc_stamp()
        base = dict(stage=name, input_fingerprint=fingerprint, started_utc=started, status='running')
        save_json(marker, base)
        try:
            result, artifacts = action()
            base.update(status='completed', finished_utc=utc_stamp(), result=result,
                        artifacts={str(Path(p).resolve()): sha256(p) for p in artifacts})
            save_json(marker, base)
            return result
        except BaseException as exc:
            base.update(status='interrupted_or_failed', error=f'{type(exc).__name__}: {exc}')
            save_json(marker, base)
            raise
