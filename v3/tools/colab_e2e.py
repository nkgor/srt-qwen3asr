# -*- coding: utf-8 -*-
"""Colab の GPU ランタイムの上で v3 ノートブックを通しでテストするスクリプト

Colab CLI から:  colab exec -s asr -f v3/tools/colab_e2e.py
(先に Qwen3-ASR_v3.ipynb と v3/tools/nb_run.py を /content にアップロードしておく。手順は v3/COLAB_TEST.md)

1. Common Voice 8 日本語テストセットから、いろいろな話者の発話をつないだ「会議っぽい音声」と正解文を作る
2. ノートブックのフォームに値を入れて、⓪〜⑧ を上から実行する(nb_run)
3. 結果を /content/e2e_out に集める(colab download で回収)
"""
import json
import os
import random
import subprocess
import sys
import wave

import numpy as np

E2E = os.environ.get("E2E_DIR", "/content/e2e")
OUT = os.environ.get("E2E_OUT", "/content/e2e_out")
NB = os.environ.get("E2E_NB", "/content/Qwen3-ASR_v3.ipynb")
MINUTES = float(os.environ.get("E2E_MINUTES", "6"))
ENGINES = os.environ.get("E2E_ENGINES", "qwen,pyannote,vllm,fw,nemo,hf").split(",")
COMPARE = os.environ.get("E2E_COMPARE", "qwen3-1.7b,qwen3-1.7b-ja,qwen3-0.6b,cohere-transcribe,whisper-large-v3-turbo,"
                                        "kotoba-whisper-v2,parakeet-ja,qwen3-1.7b-vllm").split(",")
os.makedirs(E2E, exist_ok=True)
os.makedirs(OUT, exist_ok=True)


def build_audio() -> str:
    """CV8 の発話を 0.3〜2 秒の間でつなぐ(ところどころ長めの沈黙)。正解文も書く"""
    wav = os.path.join(E2E, "meeting_cv8.wav")
    if os.path.exists(wav):
        return wav
    pq_path = os.path.join(E2E, "cv8_test.parquet")
    if not os.path.exists(pq_path):
        from huggingface_hub import hf_hub_download

        pq_path = hf_hub_download("japanese-asr/ja_asr.common_voice_8_0", "data/test-00000-of-00001.parquet",
                                  repo_type="dataset", local_dir=E2E)
    import pyarrow.parquet as pq

    rows = pq.read_table(pq_path).to_pylist()
    random.seed(7)
    random.shuffle(rows)
    sr, parts, refs, total = 16000, [], [], 0.0
    for i, r in enumerate(rows):
        p = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", "pipe:0", "-ac", "1", "-ar", "16000", "-f", "s16le",
                            "pipe:1"], input=r["audio"]["bytes"], capture_output=True)
        if p.returncode != 0:
            continue
        x = np.frombuffer(p.stdout, np.int16)
        gap = 7.0 if i % 25 == 24 else random.uniform(0.3, 2.0)
        parts += [x, np.zeros(int(gap * sr), np.int16)]
        refs.append(r["transcription"].strip().rstrip("."))
        total += len(x) / sr + gap
        if total > MINUTES * 60:
            break
    a = np.concatenate(parts)
    a = (a.astype(np.float32) + np.random.default_rng(0).normal(0, 50, len(a))).clip(-32768, 32767).astype(np.int16)
    with wave.open(wav, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(a.tobytes())
    open(os.path.join(E2E, "meeting_cv8.ref.txt"), "w", encoding="utf-8").write("".join(refs))
    print(f"テスト音声: {len(a) / sr / 60:.1f} 分, {len(refs)} 発話")
    return wav


def main() -> int:
    wav = build_audio()
    sys.path.insert(0, "/content")
    import nb_run

    cmp_flags = {}
    ov = {
        "v3setup": {"qwen": "qwen" in ENGINES, "pyannote": "pyannote" in ENGINES, "vllm": "vllm" in ENGINES,
                    "faster_whisper": "fw" in ENGINES, "nemo": "nemo" in ENGINES, "hf": "hf" in ENGINES,
                    "flash_attn": True, "mount_drive": False},
        "v3main": {"src": wav, "output_dir": OUT, "model": os.environ.get("E2E_MODEL", "Qwen3-ASR 1.7B（標準・おすすめ）"),
                   "context": "田中 山田 松井", "vtt": True, "csv": True, "plain": True},
        "v3diar": {"diarize": "pyannote" in ENGINES, "rttm": True},
        "v3adv": {},
        "v3cmp": {"start_sec": 0, "duration_sec": float(os.environ.get("E2E_CMP_SEC", "300")),
                  "reference": os.path.join(E2E, "meeting_cv8.ref.txt")},
        "v3min": {"mode": "プロンプトだけ作る"},
        "v3clean": {},
    }
    # 比較するモデル(プリセットのキー → cmp_ 変数)
    nb = json.load(open(NB, encoding="utf-8"))
    cmp_src = "".join(next(c for c in nb["cells"] if c.get("metadata", {}).get("id") == "v3cmp")["source"])
    import re

    for var, key in re.findall(r"\('(cmp_\w+)', '([^']+)'\)", cmp_src):
        cmp_flags[var] = key in COMPARE
    ov["v3cmp"].update(cmp_flags)
    if os.environ.get("E2E_SKIP"):
        ov["skip"] = os.environ["E2E_SKIP"].split(",")
    json.dump(ov, open(os.path.join(OUT, "overrides.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    rc = nb_run.run(NB, ov)
    # ログも回収しやすいように
    logs = os.path.join(os.environ.get("ASR_V3_HOME", "/content/asr_v3"), "logs")
    if os.path.isdir(logs):
        subprocess.run(["bash", "-lc", f"tar czf {OUT}/logs.tgz -C {logs} ."])
    subprocess.run(["bash", "-lc", f"cd {OUT} && tar czf /content/e2e_out.tgz ."])
    print("E2E_RC", rc)
    return rc


if __name__ == "__main__" or "get_ipython" in globals():
    main()
