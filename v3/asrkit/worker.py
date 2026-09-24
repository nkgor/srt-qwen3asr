# -*- coding: utf-8 -*-
"""asrkit.worker — 各 venv の中で動くワーカー。stdin/stdout で JSON を1行ずつやりとりする

  → {"id": 1, "cmd": "load", "key": "asr", "kind": "qwen", "options": {...}}
  ← {"id": 1, "ok": true, "info": {...}}
  → {"id": 2, "cmd": "transcribe", "key": "asr", "wav": "...", "clips": [...], ...}
  ← {"id": 2, "event": "progress", "stage": "asr", "done": 8, "total": 120, "results": [...]}
  ← {"id": 2, "ok": true, "results": [...]}

stdout はこのやりとり専用。ライブラリのログや print は全部 stderr に流す。
"""
from __future__ import annotations

import json
import os
import signal
import sys
import time
import traceback
from typing import Any, Dict


def _setup_io():
    proto = os.fdopen(os.dup(1), "w", encoding="utf-8", buffering=1)
    os.dup2(2, 1)  # 以降の print は stderr へ
    sys.stdout = sys.stderr
    return proto


def _pdeathsig() -> None:
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").prctl(1, signal.SIGTERM)
    except Exception:
        pass


def main() -> None:
    proto = _setup_io()
    _pdeathsig()
    from . import core, engines

    objs: Dict[str, Any] = {}

    def send(obj: Dict[str, Any]) -> None:
        proto.write(json.dumps(obj, ensure_ascii=False) + "\n")
        proto.flush()

    wav_cache: Dict[str, core.Wav16] = {}

    def wav(path: str) -> core.Wav16:
        w = wav_cache.get(path)
        if w is None:
            wav_cache.clear()
            w = wav_cache[path] = core.Wav16(path)
        return w

    while True:
        try:
            line = sys.stdin.readline()
        except KeyboardInterrupt:
            continue
        if not line:
            break  # 親が閉じた
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except Exception:
            continue
        rid, cmd = msg.get("id"), msg.get("cmd")
        try:
            if cmd == "exit":
                send({"id": rid, "ok": True})
                break
            elif cmd == "ping":
                info: Dict[str, Any] = {"python": sys.version.split()[0], "loaded": list(objs)}
                try:
                    import torch

                    info.update(torch=torch.__version__, cuda=torch.cuda.is_available(),
                                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
                except Exception as e:
                    info["torch_error"] = repr(e)
                send({"id": rid, "ok": True, "info": info})
            elif cmd == "load":
                key = msg["key"]
                if key in objs:
                    try:
                        objs.pop(key).close()
                    except Exception:
                        pass
                    engines.free_cuda()
                t0 = time.time()
                obj = engines.make_engine(msg["kind"], dict(msg.get("options") or {}))
                objs[key] = obj
                info = obj.info() if hasattr(obj, "info") else {}
                info["load_sec"] = round(time.time() - t0, 1)
                send({"id": rid, "ok": True, "info": info})
            elif cmd == "unload":
                obj = objs.pop(msg["key"], None)
                if obj is not None:
                    obj.close()
                engines.free_cuda()
                send({"id": rid, "ok": True})
            elif cmd == "transcribe":
                eng = objs[msg["key"]]
                w = wav(msg["wav"])
                clips = [core.Clip.from_dict(d) for d in msg["clips"]]
                done = {int(d["clip"]["id"]): core.ClipResult.from_dict(d) for d in (msg.get("done") or [])}
                language = msg.get("language") or None
                pol = core.RetryPolicy(**(msg.get("policy") or {}))
                pol.punctuates = bool(getattr(eng, "punctuates", True)) and pol.punctuates
                db = core.frame_db(w) if pol.enabled and pol.split else None

                def on_progress(stage, n, total, part):
                    send({"id": rid, "event": "progress", "stage": stage, "done": n, "total": total,
                          "results": [r.to_dict() for r in part]})

                t0 = time.time()
                res = core.run_asr(
                    clips, w.get, lambda items: eng.transcribe(items, language),
                    context=msg.get("context") or "", ctx_terms=msg.get("ctx_terms") or [],
                    ctx_label=msg.get("ctx_label") or "", policy=pol,
                    batch_size=int(msg.get("batch_size") or getattr(eng, "batch_size", 8)),
                    db=db, on_progress=on_progress, done=done,
                )
                send({"id": rid, "ok": True, "results": [r.to_dict() for r in res],
                      "seconds": round(time.time() - t0, 2), "gpu_peak_gb": engines.gpu_peak_gb()})
            elif cmd == "align":
                al = objs[msg["key"]]
                w = wav(msg["wav"])
                items = msg["items"]  # [{"id", "start", "end", "text", "language"}]
                out: Dict[str, Any] = {}
                bs = int(msg.get("batch_size") or 8)
                total = len(items)
                # 長さ順に並べてまとめて流す
                order = sorted(items, key=lambda d: d["end"] - d["start"], reverse=True)
                for i in range(0, len(order), bs * 4):
                    chunk = order[i: i + bs * 4]
                    toks = al.align([(w.get(d["start"], d["end"]), d["text"], d["language"]) for d in chunk])
                    for d, t in zip(chunk, toks):
                        out[str(d["id"])] = [[a, round(b, 3), round(c, 3)] for a, b, c in t]
                    send({"id": rid, "event": "progress", "stage": "align", "done": min(i + bs * 4, total), "total": total})
                send({"id": rid, "ok": True, "aligned": out, "gpu_peak_gb": engines.gpu_peak_gb()})
            elif cmd == "diarize":
                d = objs[msg["key"]]
                t0 = time.time()
                r = d.diarize(msg["wav"], int(msg.get("num_speakers") or 0), int(msg.get("min_speakers") or 0),
                              int(msg.get("max_speakers") or 0))
                r["seconds"] = round(time.time() - t0, 2)
                send({"id": rid, "ok": True, **r})
            else:
                raise ValueError(f"未知のコマンド: {cmd}")
        except KeyboardInterrupt:
            engines.free_cuda()
            send({"id": rid, "ok": False, "error": "cancelled"})
        except BaseException as e:  # noqa: BLE001
            engines.free_cuda()
            send({"id": rid, "ok": False, "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()})
            if isinstance(e, (SystemExit,)):
                break


if __name__ == "__main__":
    main()
