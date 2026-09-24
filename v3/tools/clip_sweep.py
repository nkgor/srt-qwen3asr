# -*- coding: utf-8 -*-
"""モデルごとの「クリップの区切り方」(max_clip / max_gap)とエンジンの設定を、Colab の GPU で比べるスクリプト

colab_e2e.py を一度通したあと(環境・テスト音声・⓪ のライブラリがそろった状態)で:
  colab exec -s asr --timeout 5400 -f v3/tools/clip_sweep.py

環境変数:
  SWEEP_VARIANTS  比べる組み合わせ。「プリセット|max_clip|max_gap|オプション」を ; 区切り
                  (max_clip / max_gap は数字か auto。オプションは chunk_length=15,beam_size=5 のように。0 は「なし」)
  SWEEP_ONLY      このプリセットだけ(カンマ区切り)
  SWEEP_SEC       先頭から何秒を使うか(既定 300)
結果: /content/sweep_out/sweep.json と sweep.md(表)

GPU がなくても(遅いけれど)同じ比較ができる。CER は GPU とほぼ同じになる:
  E2E_DIR=/tmp/e2e SWEEP_OUT=/tmp/sweep_out ASR_V3_HOME=<envs のある場所> PYTHONPATH=v3:v3/tools python v3/tools/clip_sweep.py
"""
import dataclasses
import json
import os
import sys
import time

E2E = os.environ.get("E2E_DIR", "/content/e2e")
OUT = os.environ.get("SWEEP_OUT", "/content/sweep_out")
SEC = float(os.environ.get("SWEEP_SEC", "300"))
CONTEXT = os.environ.get("SWEEP_CONTEXT", "田中 山田 松井")  # e2e の ⑥ と同じ(正解文には出てこない語)

DEFAULT_VARIANTS = """
qwen3-1.7b|30|6|;qwen3-1.7b|15|2|;qwen3-1.7b|60|6|
whisper-large-v3-turbo|30|6|;whisper-large-v3-turbo|15|2|
kotoba-whisper-v2|30|6|chunk_length=0;kotoba-whisper-v2|15|6|chunk_length=0;kotoba-whisper-v2|auto|auto|;kotoba-whisper-v2|15|2|;kotoba-whisper-v2|10|2|
cohere-transcribe|30|6|;cohere-transcribe|20|6|;cohere-transcribe|15|6|;cohere-transcribe|15|2|;cohere-transcribe|10|2|;cohere-transcribe|30|2|
parakeet-ja|30|6|;parakeet-ja|15|6|;parakeet-ja|15|2|;parakeet-ja|60|6|
granite-speech-4.1|30|6|;granite-speech-4.1|15|6|;granite-speech-4.1|15|2|
vibevoice-asr|30|6|;vibevoice-asr|60|6|;vibevoice-asr|120|10|;vibevoice-asr|15|2|
"""


def _lib():
    lib = os.path.join(os.environ.get("ASR_V3_HOME", "/content/asr_v3"), "lib")
    if lib not in sys.path:
        sys.path.insert(0, lib)
    for p in ("/content", os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else ""):
        if p and p not in sys.path:
            sys.path.insert(0, p)


def parse_variants(spec: str):
    out = []
    for part in spec.replace("\n", ";").split(";"):
        part = part.strip()
        if not part:
            continue
        key, mc, mg, opts = (part.split("|") + ["", "", ""])[:4]
        o = {}
        for kv in filter(None, (x.strip() for x in opts.split(","))):
            k, v = kv.split("=", 1)
            v = v.strip()
            o[k.strip()] = None if v in ("0", "none", "None") else (int(v) if v.isdigit() else v)
        out.append((key.strip(), mc.strip() or "auto", mg.strip() or "auto", o))
    return out


def edit_counts(ref: str, hyp: str):
    """置換・脱落・挿入の数(rapidfuzz があれば)"""
    try:
        from rapidfuzz.distance import Levenshtein

        ops = Levenshtein.editops(ref, hyp)
        c = {"replace": 0, "delete": 0, "insert": 0}
        for op in ops:
            c[op.tag] += 1
        return c["replace"], c["delete"], c["insert"]
    except Exception:
        return None, None, None


def dropped_utts(utts, hyp_norm, core):
    """正解の発話のうち、文字起こしにほとんど出てこないもの(読み飛ばし)の数"""
    try:
        from rapidfuzz import fuzz
    except Exception:
        return None
    n = 0
    for u in utts:
        t = core.normalize_for_cer(u["text"])
        if len(t) >= 4 and fuzz.partial_ratio(t, hyp_norm) < 60:
            n += 1
    return n


def main() -> int:
    _lib()
    import colab_e2e
    from asrkit import core, pipeline, presets

    os.makedirs(OUT, exist_ok=True)
    wav = colab_e2e.build_audio()
    ref = open(colab_e2e.excerpt_ref(0.0, SEC), encoding="utf-8").read()
    utts = [u for u in json.load(open(os.path.join(E2E, "meeting_cv8.utts.json"), encoding="utf-8"))
            if (u["start"] + u["end"]) / 2 < SEC]
    ref_n = core.normalize_for_cer(ref)
    variants = parse_variants(os.environ.get("SWEEP_VARIANTS") or DEFAULT_VARIANTS)
    only = set(filter(None, os.environ.get("SWEEP_ONLY", "").split(",")))
    if only:
        variants = [v for v in variants if v[0] in only]
    sess = pipeline.Session()
    terms = core.parse_terms(CONTEXT)
    rows, cur_env = [], None
    warmed = set()
    t_all = time.time()
    for key, mc, mg, opts in variants:
        base = presets.find_preset(key)
        if base is None:
            print(f"!! プリセットがありません: {key}")
            continue
        tag = f"{mc}/{mg}" + ("".join(f",{k}={v}" for k, v in opts.items()) if opts else "")
        p = base
        if opts:
            o = dict(base.options)
            for k, v in opts.items():
                if v is None:
                    o.pop(k, None)
                else:
                    o[k] = v
            p = dataclasses.replace(base, key=f"{base.key}@{tag}", label=f"{base.label} [{tag}]", options=o)
            presets.PRESETS.append(p)
        if cur_env and cur_env != (base.env, base.key):  # 前のモデルを外す
            sess.h.unload(cur_env[0], "asr")
        cur_env = (base.env, base.key)
        st = pipeline.Settings(preset=p.key, language="Japanese", output_dir=OUT, formats=("txt", "json"),
                               context_terms=terms, cache=False, diarize=False, review=False,
                               max_clip=pipeline.auto_num(mc), max_gap=pipeline.auto_num(mg))
        safe = tag.replace("/", "-").replace(",", "_").replace("=", "")
        print(f"\n===== {base.key} [{tag}] =====", flush=True)
        try:
            if base.key not in warmed:  # 1 回目は CUDA の準備などで遅いので、短い区間で空回ししておく
                sess.transcribe_file(wav, st, excerpt=(0.0, 20.0), out_base=os.path.join(OUT, "warmup"), quiet=True)
                warmed.add(base.key)
            o = sess.transcribe_file(wav, st, excerpt=(0.0, SEC), out_base=os.path.join(OUT, f"{base.key}.{safe}"),
                                     quiet=True)
        except Exception as e:
            print(f"!! 失敗: {type(e).__name__}: {str(e)[:800]}", flush=True)
            rows.append({"preset": base.key, "tag": tag, "error": f"{type(e).__name__}: {str(e)[:300]}"})
            continue
        hyp_n = core.normalize_for_cer(o["text"])
        s_, d_, i_ = edit_counts(ref_n, hyp_n)
        asr = o["meta"]["asr"] or {}
        clips = [r.clip for r in o["results"] if r.clip.parent is None]  # リトライで割ったものは数えない
        row = {
            "preset": base.key, "tag": tag, "max_clip": o["meta"]["clips"]["max_clip"], "max_gap": o["meta"]["clips"]["max_gap"],
            "options": opts, "clips": len(clips), "longest": round(max((c.dur for c in clips), default=0), 1),
            "cer": round(core.cer(ref, o["text"]), 4), "sub": s_, "del": d_, "ins": i_,
            "drops": core.drop_runs(ref, o["text"]), "negations": core.negation_check(ref, o["text"]),
            "ref_chars": len(ref_n), "hyp_chars": len(hyp_n), "dropped_utts": dropped_utts(utts, hyp_n, core),
            "n_utts": len(utts), "asr_sec": asr.get("asr_sec"), "x": round(SEC / max(asr.get("asr_sec") or 1e-6, 1e-6), 1),
            "load_sec": asr.get("load_sec"), "gpu_peak_gb": asr.get("gpu_peak_gb"), "flags": o["flags"],
            "retries": o["retries"], "text_head": o["text"][:120],
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        json.dump(rows, open(os.path.join(OUT, "sweep.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    sess.free()
    # 表
    lines = [f"# クリップの区切り方の比較（先頭 {SEC:g} 秒・正解 {len(ref_n)} 字・{len(utts)} 発話・context「{CONTEXT}」）", "",
             f"GPU: {sess.gpu.name} / 所要 {core.fmt_dur(time.time() - t_all)}", "",
             "| モデル | 区切り(上限/無音) | クリップ | CER | 置換/脱落/挿入 | 文字数 | 読み飛ばし発話 | 抜け | 否定 | 推論 | 倍速 | VRAM峰 | 要確認 |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if "error" in r:
            lines.append(f"| {r['preset']} | {r['tag']} | - | 失敗 | {r['error'][:80]} | | | | | | | | |")
            continue
        lines.append(
            f"| {r['preset']} | {r['tag']} ({r['max_clip']:g}/{r['max_gap']:g}) | {r['clips']}（最長 {r['longest']}秒） | "
            f"**{r['cer'] * 100:.1f}%** | {r['sub']}/{r['del']}/{r['ins']} | {r['hyp_chars']}/{r['ref_chars']} | "
            f"{r['dropped_utts']}/{r['n_utts']} | {r['drops'][0]}か所/{r['drops'][1]}字 | {r['negations'][0]}/{r['negations'][1]}"
            f"{' +' + str(r['negations'][2]) if r['negations'][2] else ''} | {r['asr_sec']}秒 | x{r['x']:.0f} | "
            f"{r['gpu_peak_gb'] or '-'} | {r['flags']} |")
    md = "\n".join(lines) + "\n"
    open(os.path.join(OUT, "sweep.md"), "w", encoding="utf-8").write(md)
    print(md)
    return 0


if __name__ == "__main__" or "get_ipython" in globals():
    main()
