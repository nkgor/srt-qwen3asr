# -*- coding: utf-8 -*-
"""asrkit.runtime — 環境(venv)づくり・ワーカープロセス・vLLMサーバー・GPU/Colab まわり

ノートブックのカーネル側で動く。モデルごとに依存がぶつかるので、エンジンは
それぞれ専用の venv の中の「ワーカー」プロセスで動かす(カーネルは汚さない=再起動いらず)。
"""
from __future__ import annotations

import hashlib
import json
import os
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import core

PKG_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_PARENT = os.path.dirname(PKG_DIR)
BASE_DIR = os.environ.get("ASR_V3_HOME", "/content/asr_v3")
ENV_ROOT = os.path.join(BASE_DIR, "envs")
LOG_DIR = os.path.join(BASE_DIR, "logs")
WORK_DIR = os.path.join(BASE_DIR, "work")


def _mkdirs() -> None:
    for d in (BASE_DIR, ENV_ROOT, LOG_DIR, WORK_DIR):
        os.makedirs(d, exist_ok=True)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# =====================================================================
# GPU / Colab
# =====================================================================


@dataclass
class GPUInfo:
    name: str = ""
    mem_gb: float = 0.0
    cc: Tuple[int, int] = (0, 0)
    driver: str = ""
    cuda: str = ""  # ドライバが対応する CUDA バージョン(nvidia-smi 表示)

    @property
    def ok(self) -> bool:
        return bool(self.name)

    @property
    def ampere_plus(self) -> bool:
        return self.cc[0] >= 8


def gpu_info() -> GPUInfo:
    try:
        p = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,compute_cap,driver_version", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30,
        )
        line = (p.stdout or "").strip().splitlines()[0]
        name, mem, cc, drv = [x.strip() for x in line.split(",")[:4]]
        maj, mnr = (cc.split(".") + ["0"])[:2]
        info = GPUInfo(name, float(mem) / 1024.0, (int(maj), int(mnr)), drv)
        q = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=30).stdout
        import re

        m = re.search(r"CUDA Version:\s*([\d.]+)", q or "")
        info.cuda = m.group(1) if m else ""
        return info
    except Exception:
        return GPUInfo()


def auto_batch_size(g: GPUInfo, model_gb: float = 4.0, per_item_gb: float = 0.5, cap: int = 128) -> int:
    """VRAM からざっくりバッチサイズを決める(30秒クリップ想定)。T4≒8 / L4≒16 / A100-40G≒32 / 80G≒64"""
    if not g.ok:
        return 1
    free = max(1.0, g.mem_gb * 0.8 - model_gb)
    bs = int(free / per_item_gb)
    for p in (128, 64, 32, 16, 8, 4, 2, 1):
        if bs >= p:
            return min(p, cap)
    return 1


def in_colab() -> bool:
    try:
        import google.colab  # noqa: F401

        return True
    except Exception:
        return False


def get_secret(name: str) -> Optional[str]:
    """Colab のシークレット → 環境変数 の順で探す"""
    try:
        from google.colab import userdata  # type: ignore

        v = userdata.get(name)
        if v:
            return v
    except Exception:
        pass
    return os.environ.get(name) or None


# =====================================================================
# venv の用意
# =====================================================================


@dataclass
class EnvSpec:
    name: str
    packages: List[str]
    check: str = "print('ok')"  # インストール確認用の python コード(最後の行を表示)
    isolated: bool = False  # True: システムの torch を使わない完全分離(vLLM など torch を固定するもの)
    python: str = "3.12"  # isolated のときの Python
    pip_args: List[str] = field(default_factory=list)
    post: List[List[str]] = field(default_factory=list)  # 追加の pip コマンド(引数リスト)
    note: str = ""

    def digest(self) -> str:
        d = asdict(self)
        d.pop("note", None)
        return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]


def env_dir(name: str) -> str:
    return os.path.join(ENV_ROOT, name)


def env_python(name: str) -> str:
    return os.path.join(env_dir(name), "bin", "python")


def _uv_version(uv: str) -> Tuple[int, ...]:
    try:
        out = subprocess.run([uv, "--version"], capture_output=True, text=True, timeout=30).stdout
        return tuple(int(x) for x in out.split()[1].split(".")[:3])
    except Exception:
        return (0,)


def _uv() -> str:
    """uv を返す(無いか古ければ入れる。--torch-backend などを使うので 0.8 以上)"""
    uv = shutil.which("uv")
    if uv and _uv_version(uv) >= (0, 8, 0):
        return uv
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "uv"], check=True)
    cand = os.path.join(os.path.dirname(sys.executable), "uv")
    return cand if os.path.exists(cand) else (shutil.which("uv") or "uv")


def _run_logged(cmd: List[str], logf, env: Optional[Dict[str, str]] = None, timeout: Optional[float] = None) -> int:
    logf.write(f"\n$ {' '.join(cmd)}\n")
    logf.flush()
    p = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env, timeout=timeout)
    return p.returncode


@dataclass
class EnvStatus:
    name: str
    ok: bool
    seconds: float = 0.0
    info: str = ""
    log: str = ""
    skipped: bool = False


def ensure_env(spec: EnvSpec, force: bool = False) -> EnvStatus:
    """venv を作って packages を入れる。前回と同じ内容なら確認だけしてスキップ"""
    _mkdirs()
    d = env_dir(spec.name)
    marker = os.path.join(d, ".asr_v3_env.json")
    logp = os.path.join(LOG_DIR, f"env_{spec.name}.log")
    t0 = time.time()
    if not force and os.path.exists(marker):
        try:
            if json.load(open(marker)).get("digest") == spec.digest():
                ok, info = check_env(spec)
                if ok:
                    return EnvStatus(spec.name, True, time.time() - t0, info, logp, skipped=True)
        except Exception:
            pass
    if os.path.exists(d):
        shutil.rmtree(d, ignore_errors=True)
    uv = _uv()
    env = dict(os.environ)
    env.setdefault("UV_LINK_MODE", "copy")
    with open(logp, "a", encoding="utf-8") as logf:
        logf.write(f"\n===== {time.ctime()} {spec.name} =====\n")
        if spec.isolated:
            rc = _run_logged([uv, "venv", "--seed", "-p", spec.python, d], logf, env)
            if rc == 0:
                rc = _run_logged([uv, "pip", "install", "--python", env_python(spec.name), *spec.pip_args, *spec.packages], logf, env)
        else:
            # システムの torch 等をそのまま使う venv(pip はシステムにある物を「入っている」とみなしてくれる)
            rc = _run_logged([uv, "venv", "--seed", "--system-site-packages", "-p", sys.executable, d], logf, env)
            if rc == 0:
                rc = _run_logged([env_python(spec.name), "-m", "pip", "install", "--progress-bar", "off",
                                  *spec.pip_args, *spec.packages], logf, env)
        for extra in spec.post:
            if rc != 0:
                break
            # post は失敗しても致命的でない(flash-attn など)
            _run_logged([env_python(spec.name), "-m", "pip", "install", "--progress-bar", "off", *extra], logf, env)
    if rc != 0:
        return EnvStatus(spec.name, False, time.time() - t0, _tail(logp), logp)
    ok, info = check_env(spec)
    if ok:
        with open(marker, "w") as f:
            json.dump({"digest": spec.digest(), "time": time.time(), "info": info}, f)
    return EnvStatus(spec.name, ok, time.time() - t0, info if ok else info + "\n" + _tail(logp), logp)


def check_env(spec: EnvSpec) -> Tuple[bool, str]:
    py = env_python(spec.name)
    if not os.path.exists(py):
        return False, "python がありません"
    p = subprocess.run([py, "-c", spec.check], capture_output=True, text=True, timeout=600)
    out = (p.stdout or "").strip().splitlines()
    if p.returncode != 0:
        return False, (p.stderr or "")[-1500:]
    return True, out[-1] if out else "ok"


def _tail(path: str, n: int = 40) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return "".join(deque(f, maxlen=n))
    except Exception:
        return ""


def ensure_envs(specs: Sequence[EnvSpec], parallel: int = 3, force: bool = False) -> List[EnvStatus]:
    """複数の venv を並列に用意(経過を表示)"""
    specs = list(specs)
    if not specs:
        return []
    res: Dict[str, EnvStatus] = {}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max(1, parallel)) as ex:
        futs = {ex.submit(ensure_env, s, force): s for s in specs}
        pending = set(futs)
        last = 0.0
        while pending:
            done = [f for f in pending if f.done()]
            for f in done:
                s = futs[f]
                pending.discard(f)
                try:
                    st = f.result()
                except Exception as e:  # pragma: no cover
                    st = EnvStatus(s.name, False, 0, repr(e))
                res[s.name] = st
                mark = "✅" if st.ok else "❌"
                how = "(前回のまま)" if st.skipped else f"({core.fmt_dur(st.seconds)})"
                log(f"{mark} 環境 {s.name} {how} {st.info.splitlines()[0] if st.ok and st.info else ''}")
                if not st.ok:
                    print(st.info[-3000:])
            if pending and time.time() - last > 20:
                last = time.time()
                names = ", ".join(futs[f].name for f in pending)
                log(f"… インストール中: {names} (経過 {core.fmt_dur(time.time() - t0)})")
            time.sleep(0.5)
    return [res[s.name] for s in specs]


# =====================================================================
# ワーカー(venv の中で asrkit.worker を動かし、JSON 1行ずつでやりとり)
# =====================================================================


class WorkerError(RuntimeError):
    pass


class Worker:
    def __init__(self, env_name: str, extra_env: Optional[Dict[str, str]] = None, echo: bool = False):
        _mkdirs()
        self.env_name = env_name
        py = env_python(env_name)
        if not os.path.exists(py):
            raise WorkerError(f"環境 {env_name} がありません。セットアップのセルで入れてください。")
        env = dict(os.environ)
        env.update(extra_env or {})
        env["PYTHONPATH"] = PKG_PARENT + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        env["PYTHONUNBUFFERED"] = "1"
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        env.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
        self.log_path = os.path.join(LOG_DIR, f"worker_{env_name}.log")
        self._log = open(self.log_path, "a", encoding="utf-8")
        self._log.write(f"\n===== start {time.ctime()} =====\n")
        self.proc = subprocess.Popen(
            [py, "-u", "-m", "asrkit.worker"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace", bufsize=1, env=env,
        )
        self.q: "queue.Queue[Optional[Dict[str, Any]]]" = queue.Queue()
        self.err: deque = deque(maxlen=400)
        self.echo = echo
        self._id = 0
        self._lock = threading.Lock()
        threading.Thread(target=self._read_out, daemon=True).start()
        threading.Thread(target=self._read_err, daemon=True).start()
        self.loaded: Dict[str, Dict[str, Any]] = {}

    def _read_out(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                self.q.put(json.loads(line))
            except Exception:
                self.err.append("[stdout] " + line)
        self.q.put(None)

    def _read_err(self) -> None:
        assert self.proc.stderr is not None
        for line in self.proc.stderr:
            self.err.append(line.rstrip("\n"))
            try:
                self._log.write(line)
                self._log.flush()
            except Exception:
                pass
            if self.echo:
                print(line, end="")

    def alive(self) -> bool:
        return self.proc.poll() is None

    def tail(self, n: int = 60) -> str:
        return "\n".join(list(self.err)[-n:])

    def death_hint(self) -> str:
        rc = self.proc.poll()
        if rc in (-9, 137):
            return ("(強制終了されました。メモリ不足(OOM)の可能性が高いです。"
                    "ランタイムをハイメモリにするか、⑧で VRAM を解放してから、読み込むモデルを減らしてください)")
        if rc in (-11, 139):
            return "(セグメンテーション違反で落ちました。ライブラリの組み合わせの問題の可能性があります)"
        return f"(終了コード {rc})"

    def call(self, cmd: str, *, on_event: Optional[Callable[[Dict[str, Any]], None]] = None,
             timeout: Optional[float] = None, **payload: Any) -> Dict[str, Any]:
        with self._lock:
            self._id += 1
            rid = self._id
            msg = {"id": rid, "cmd": cmd, **payload}
            if not self.alive():
                raise WorkerError(f"ワーカー({self.env_name})は終了しています\n{self.tail()}")
            assert self.proc.stdin is not None
            self.proc.stdin.write(json.dumps(msg, ensure_ascii=False) + "\n")
            self.proc.stdin.flush()
            t0 = time.time()
            try:
                while True:
                    try:
                        m = self.q.get(timeout=1.0)
                    except queue.Empty:
                        if not self.alive():
                            raise WorkerError(f"ワーカー({self.env_name})が落ちました {self.death_hint()}\n{self.tail()}")
                        if timeout and time.time() - t0 > timeout:
                            raise WorkerError(f"ワーカー({self.env_name})がタイムアウトしました")
                        continue
                    if m is None:
                        try:
                            self.proc.wait(timeout=5)
                        except Exception:
                            pass
                        raise WorkerError(f"ワーカー({self.env_name})が落ちました {self.death_hint()}\n{self.tail()}")
                    if m.get("id") != rid:
                        continue
                    if m.get("event"):
                        if on_event:
                            on_event(m)
                        continue
                    if m.get("ok"):
                        return m
                    err = m.get("error", "?")
                    if err == "cancelled":
                        raise KeyboardInterrupt
                    raise WorkerError(f"{err}\n{m.get('traceback', '')}\n--- ワーカーのログ(末尾) ---\n{self.tail(30)}")
            except KeyboardInterrupt:
                self._cancel(rid)
                raise

    def _cancel(self, rid: int) -> None:
        """セルの停止ボタン → ワーカーにも SIGINT を送って今の処理だけ止める(モデルは残す)"""
        if not self.alive():
            return
        try:
            self.proc.send_signal(signal.SIGINT)
        except Exception:
            return
        t0 = time.time()
        while time.time() - t0 < 20:
            try:
                m = self.q.get(timeout=1.0)
            except queue.Empty:
                continue
            if m is None or (m.get("id") == rid and not m.get("event")):
                return
        log(f"⚠️ ワーカー({self.env_name})が止まらないので終了させます")
        self.close(force=True)

    def close(self, force: bool = False) -> None:
        if self.alive() and not force:
            try:
                assert self.proc.stdin is not None
                self.proc.stdin.write(json.dumps({"id": 0, "cmd": "exit"}) + "\n")
                self.proc.stdin.flush()
                self.proc.wait(timeout=20)
            except Exception:
                pass
        if self.alive():
            self.proc.kill()
            try:
                self.proc.wait(timeout=10)
            except Exception:
                pass
        try:
            self._log.close()
        except Exception:
            pass


# =====================================================================
# vLLM サーバー(OpenAI 互換の /v1/audio/transcriptions を叩く)
# =====================================================================


class VLLMServer:
    def __init__(self, env_name: str, model: str, *, port: int = 8791, gpu_mem: float = 0.5,
                 max_model_len: Optional[int] = None, extra_args: Sequence[str] = (),
                 extra_env: Optional[Dict[str, str]] = None):
        self.env_name, self.model, self.port = env_name, model, port
        self.gpu_mem, self.max_model_len = gpu_mem, max_model_len
        self.extra_args = list(extra_args)
        self.extra_env = dict(extra_env or {})
        self.proc: Optional[subprocess.Popen] = None
        self.log_path = os.path.join(LOG_DIR, f"vllm_{model.replace('/', '__')}.log")
        self.base = f"http://127.0.0.1:{port}"

    @property
    def key(self) -> str:
        return json.dumps([self.model, self.gpu_mem, self.max_model_len, self.extra_args])

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self, timeout: float = 2400) -> None:
        import urllib.request

        _mkdirs()
        exe = os.path.join(env_dir(self.env_name), "bin", "vllm")
        if not os.path.exists(exe):
            raise WorkerError("vLLM の環境がありません。セットアップのセルで vLLM にチェックを入れてください。")
        args = [exe, "serve", self.model, "--host", "127.0.0.1", "--port", str(self.port),
                "--served-model-name", "asr", "--gpu-memory-utilization", f"{self.gpu_mem:.2f}"]
        if self.max_model_len:
            args += ["--max-model-len", str(self.max_model_len)]
        args += self.extra_args
        env = dict(os.environ)
        env.update(self.extra_env)
        env.setdefault("VLLM_LOGGING_LEVEL", "INFO")
        logf = open(self.log_path, "a", encoding="utf-8")
        logf.write(f"\n===== {time.ctime()} {' '.join(args)} =====\n")
        logf.flush()
        # 親(カーネル)が死んだら道連れにする prctl をしてから exec するラッパー経由で起動
        wrap = ("import ctypes,os,signal,sys\n"
                "try: ctypes.CDLL('libc.so.6').prctl(1, signal.SIGTERM)\n"
                "except Exception: pass\n"
                "os.execvp(sys.argv[1], sys.argv[1:])")
        self.proc = subprocess.Popen([sys.executable, "-c", wrap, *args], stdout=logf, stderr=subprocess.STDOUT, env=env)
        t0, last = time.time(), 0.0
        while True:
            if not self.alive():
                raise WorkerError(f"vLLM サーバーが起動に失敗しました。\n{_tail(self.log_path, 60)}")
            try:
                with urllib.request.urlopen(self.base + "/health", timeout=5) as r:
                    if r.status == 200:
                        log(f"✅ vLLM サーバー起動 ({core.fmt_dur(time.time() - t0)}): {self.model}")
                        return
            except Exception:
                pass
            if time.time() - t0 > timeout:
                self.stop()
                raise WorkerError(f"vLLM サーバーの起動がタイムアウトしました\n{_tail(self.log_path, 40)}")
            if time.time() - last > 30:
                last = time.time()
                lines = [l for l in _tail(self.log_path, 5).splitlines() if l.strip()]
                log(f"… vLLM 起動待ち {core.fmt_dur(time.time() - t0)}: {lines[-1][-160:] if lines else ''}")
            time.sleep(2)

    def _one(self, audio: np.ndarray, ctx: str, language: Optional[str], temperature: float) -> Dict[str, Any]:
        import requests

        data = {"model": "asr", "response_format": "json", "temperature": str(temperature)}
        if language:
            data["language"] = language
        if ctx:
            data["prompt"] = ctx
        body = core.wav_bytes(audio)
        err: Optional[Exception] = None
        for i in range(4):
            try:
                r = requests.post(self.base + "/v1/audio/transcriptions", data=data,
                                  files={"file": ("clip.wav", body, "audio/wav")}, timeout=900)
                if r.status_code >= 400:
                    raise RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")
                return {"text": (r.json().get("text") or "").strip(), "language": language or ""}
            except Exception as e:  # 一時的な失敗は少し待って再送
                err = e
                if not self.alive():
                    break
                time.sleep(1.5 * (i + 1))
        raise WorkerError(f"vLLM への送信に失敗: {err}\n{_tail(self.log_path, 20)}")

    def transcribe_fn(self, language: Optional[str], concurrency: int = 64, temperature: float = 0.0) -> core.TranscribeFn:
        def fn(items: List[Tuple[np.ndarray, str]]) -> List[Dict[str, Any]]:
            with ThreadPoolExecutor(max_workers=max(1, min(concurrency, len(items)))) as ex:
                futs = [ex.submit(self._one, a, c, language, temperature) for a, c in items]
                return [f.result() for f in futs]

        return fn

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=30)
            except Exception:
                self.proc.kill()
        self.proc = None


# =====================================================================
# ハンドルの管理(モデルは使い回す / 切り替え時は解放)
# =====================================================================


class Handles:
    """ワーカーと vLLM サーバーをまとめて持つ。カーネルに1つだけ置いて使い回す"""

    def __init__(self) -> None:
        self.workers: Dict[str, Worker] = {}
        self.vllm: Optional[VLLMServer] = None

    def worker(self, env_name: str, extra_env: Optional[Dict[str, str]] = None) -> Worker:
        w = self.workers.get(env_name)
        if w is not None and w.alive():
            return w
        if w is not None:
            log(f"⚠️ ワーカー({env_name})が落ちていたので起動しなおします。末尾ログ:\n{w.tail(15)}")
        w = Worker(env_name, extra_env)
        self.workers[env_name] = w
        return w

    def ensure_loaded(self, env_name: str, key: str, kind: str, options: Dict[str, Any],
                      extra_env: Optional[Dict[str, str]] = None, exclusive_group: Optional[str] = None) -> Dict[str, Any]:
        """ワーカーにエンジンを読み込ませる(同じ設定なら何もしない)。
        exclusive_group が同じエンジンは同時に1つだけ(ASR モデルの切り替えで前のを解放)"""
        w = self.worker(env_name, extra_env)
        cur = w.loaded.get(key)
        if cur is not None and cur.get("options") == options and cur.get("kind") == kind:
            return cur.get("info", {})
        if exclusive_group:
            for k, v in list(w.loaded.items()):
                if v.get("group") == exclusive_group and k != key:
                    self.unload(env_name, k)
        if cur is not None:
            self.unload(env_name, key)
        t0 = time.time()
        log(f"モデル読み込み中: {options.get('model', kind)} ({env_name})")
        r = w.call("load", key=key, kind=kind, options=options)
        info = r.get("info", {})
        w.loaded[key] = {"kind": kind, "options": options, "info": info, "group": exclusive_group}
        log(f"✅ 読み込み完了 ({core.fmt_dur(time.time() - t0)}) {info.get('summary', '')}")
        return info

    def unload(self, env_name: str, key: str) -> None:
        w = self.workers.get(env_name)
        if w is None or not w.alive():
            return
        if key in w.loaded:
            try:
                w.call("unload", key=key)
            except Exception:
                pass
            w.loaded.pop(key, None)

    def vllm_server(self, env_name: str, model: str, **kw: Any) -> VLLMServer:
        s = VLLMServer(env_name, model, **kw)
        if self.vllm is not None and self.vllm.alive() and self.vllm.key == s.key:
            return self.vllm
        if self.vllm is not None:
            self.vllm.stop()
        s.start()
        self.vllm = s
        return s

    def stop_vllm(self) -> None:
        if self.vllm is not None:
            self.vllm.stop()
            self.vllm = None

    def close_all(self) -> None:
        self.stop_vllm()
        for w in list(self.workers.values()):
            w.close()
        self.workers.clear()

    def status(self) -> List[str]:
        out = []
        for name, w in self.workers.items():
            out.append(f"{name}: {'稼働中' if w.alive() else '停止'} / " + ", ".join(v['options'].get('model', k) for k, v in w.loaded.items()))
        if self.vllm is not None:
            out.append(f"vLLM: {'稼働中' if self.vllm.alive() else '停止'} / {self.vllm.model}")
        return out
