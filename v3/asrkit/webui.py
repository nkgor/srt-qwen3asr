# -*- coding: utf-8 -*-
"""asrkit.webui — Gradio のかんたん画面(音声をアップロードしたら文字起こし)

ノートブックの ⑨ から `webui.launch(SESS, ST)` で立ち上げる。
画面の処理は Session.transcribe_file をそのまま呼ぶだけなので、結果・キャッシュ・出力はノートブックと同じ。
gradio が無くても import できる(gradio は launch / build の中で読む)。
"""
from __future__ import annotations

import contextlib
import io
import os
import re
import shutil
import sys
import threading
import time
from typing import Any, Dict, List, Optional

from . import core, runtime
from .pipeline import Session, Settings, log, timing_breakdown
from .presets import PRESETS, Preset, find_preset

OUT_ROOT = os.path.join(runtime.BASE_DIR, "webui")
FORMATS = ("txt", "srt", "vtt", "json", "csv", "md", "plain")
LANGS = ["Japanese", "auto", "English", "Chinese", "Korean", "Cantonese", "French", "German", "Spanish"]


def _env_ready(p: Preset) -> bool:
    if p.engine == "vllm":
        return os.path.exists(os.path.join(runtime.env_dir("vllm"), "bin", "vllm"))
    return os.path.exists(runtime.env_python(p.env))


def available_presets() -> List[Preset]:
    """① で環境を入れたモデルだけ(画面のプルダウン用)"""
    return [p for p in PRESETS if _env_ready(p)]


def diar_ready() -> bool:
    return os.path.exists(runtime.env_python("pyannote"))


class _Tee(io.TextIOBase):
    """print を画面のログ欄にも流す(元の出力先にもそのまま書く)"""

    def __init__(self, orig: Any, buf: List[str], lock: threading.Lock) -> None:
        self.orig, self.buf, self.lock = orig, buf, lock

    def write(self, s: str) -> int:
        with self.lock:
            self.buf.append(s)
        try:
            self.orig.write(s)
        except Exception:
            pass
        return len(s)

    def flush(self) -> None:
        try:
            self.orig.flush()
        except Exception:
            pass


def _log_text(buf: List[str], lock: threading.Lock, n: int = 40) -> str:
    with lock:
        s = "".join(buf)
    s = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", s)
    lines = [l.split("\r")[-1] for l in s.split("\n")]  # tqdm の \r 上書きは最後だけ
    return "\n".join([l for l in lines if l.strip()][-n:])


def make_settings(base: Settings, *, model: str, language: str, context: str, diarize: bool,
                  num_speakers: int = 0, speaker_names: str = "", output_dir: str = "") -> Settings:
    """画面の入力 → Settings(ほかの細かい設定は ⑤ の ST を引き継ぐ)"""
    p = find_preset(model)
    if p is None:
        raise ValueError(f"モデルが見つかりません: {model}")
    return base.replace(
        preset=p.key, language=language or "Japanese", context_terms=core.parse_terms(context or ""),
        diarize=bool(diarize), num_speakers=int(num_speakers or 0), speaker_names=speaker_names or "",
        output_dir=output_dir, formats=FORMATS + (("rttm",) if diarize else ()), skip_existing=False,
    )


def job_dir(src: str) -> str:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    stem = os.path.splitext(os.path.basename(src))[0] or "audio"
    d = os.path.join(OUT_ROOT, f"{stamp}_{stem}")
    os.makedirs(d, exist_ok=True)
    return d


def transcribe(sess: Session, base: Settings, src: str, *, model: str, language: str = "Japanese",
               context: str = "", diarize: bool = False, num_speakers: int = 0,
               speaker_names: str = "") -> Dict[str, Any]:
    """1ファイルを文字起こしして、画面に出すもの(本文・ファイル・まとめ)を返す。gradio なしでも呼べる"""
    if not src or not os.path.exists(src):
        raise ValueError("音声/動画ファイルをアップロードしてください")
    d = job_dir(src)
    # アップロードされた一時ファイルは消えることがあるので、出力先にコピーしてから処理
    local = os.path.join(d, os.path.basename(src))
    if os.path.abspath(src) != os.path.abspath(local):
        shutil.copy2(src, local)
    st = make_settings(base, model=model, language=language, context=context, diarize=diarize and diar_ready(),
                       num_speakers=num_speakers, speaker_names=speaker_names, output_dir=d)
    p = sess.preset_of(st)
    if p.engine != "vllm" and sess.h.vllm is not None:
        sess.h.stop_vllm()  # vLLM から別のモデルに切り替えたら VRAM を空ける
    o = sess.transcribe_file(local, st)
    paths = o.get("paths") or {}
    txt = ""
    for k in ("txt", "plain"):
        if paths.get(k) and os.path.exists(paths[k]):
            txt = open(paths[k], encoding="utf-8").read()
            break
    T = o.get("timing") or {}
    asr = (o.get("meta") or {}).get("asr") or {}
    diar = (o.get("meta") or {}).get("diar") or {}
    dur = float(o.get("duration") or 0)
    total = float(T.get("total") or 0)
    summary = [
        f"**{os.path.basename(src)}** — {core.fmt_dur(dur)} / {p.label}",
        f"合計 {core.fmt_dur(total)}（音声の x{dur / max(total, 1e-6):.0f} 倍速）",
        f"内訳: {timing_breakdown(T, asr)}",
        f"再推論 {o.get('retries', 0)} / 要確認 {o.get('flags', 0)}"
        + (f"・話者 {diar.get('speakers')} 人" if diar.get("speakers") else ""),
    ]
    if diarize and not diar_ready():
        summary.append("⚠️ 話者分離の環境(pyannote)が入っていないので、話者なしで出しました")
    if o.get("flags"):
        summary.append("⚠️ 要確認のクリップがあります。`_review.md` を見てください")
    files = [paths[k] for k in (*FORMATS, "rttm", "review") if paths.get(k) and os.path.exists(paths[k])]
    return {"text": txt, "files": files, "summary": "  \n".join(summary), "out": o, "dir": d}


def build(sess: Session, base: Optional[Settings] = None, title: str = "Qwen3-ASR 文字起こし v3"):
    """gradio の画面(Blocks)を作る"""
    import gradio as gr

    base = base or Settings()
    presets = available_presets()
    if not presets:
        raise RuntimeError("使えるモデルがありません。① のセットアップでモデルの環境を入れてください")
    labels = [p.label for p in presets]
    cur = find_preset(base.preset)
    default = cur.label if cur is not None and cur.label in labels else labels[0]
    lock = threading.Lock()  # Session は同時に1件だけ(ワーカーは1本ずつ)

    def run(file_path, mic_path, model, language, context, diarize, num_speakers, speaker_names):
        src = file_path or mic_path
        if not src:
            raise gr.Error("音声/動画ファイルをアップロードするか、マイクで録音してください")
        buf: List[str] = []
        blk = threading.Lock()
        res: Dict[str, Any] = {}

        def work() -> None:
            tee_out, tee_err = _Tee(sys.stdout, buf, blk), _Tee(sys.stderr, buf, blk)
            try:
                with lock, contextlib.redirect_stdout(tee_out), contextlib.redirect_stderr(tee_err):
                    res["r"] = transcribe(sess, base, src, model=model, language=language, context=context,
                                          diarize=diarize, num_speakers=num_speakers, speaker_names=speaker_names)
            except BaseException as e:  # 画面にエラーを出す
                res["e"] = e

        th = threading.Thread(target=work, daemon=True)
        th.start()
        while th.is_alive():
            yield "", None, "⏳ 処理中…（初回はモデルのダウンロードと読み込みに数分かかります）", _log_text(buf, blk)
            th.join(1.0)
        if "e" in res:
            e = res["e"]
            yield "", None, f"❌ {type(e).__name__}: {str(e)[:800]}", _log_text(buf, blk)
            return
        r = res["r"]
        yield r["text"], r["files"], r["summary"], _log_text(buf, blk)

    with gr.Blocks(title=title) as demo:
        gr.Markdown(f"## {title}\n音声や動画をアップロードして「文字起こし」を押してください。"
                    "結果の txt / srt / vtt / json / csv / md はいちばん下からダウンロードできます。")
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Tab("ファイル"):
                    f_in = gr.File(label="音声 / 動画", type="filepath",
                                   file_types=list(core.AUDIO_EXTS))
                with gr.Tab("マイク"):
                    mic = gr.Audio(label="録音", sources=["microphone"], type="filepath")
                model = gr.Dropdown(labels, value=default, label="モデル")
                language = gr.Dropdown(LANGS, value=base.language if base.language in LANGS else "Japanese",
                                       label="言語", allow_custom_value=True)
                context = gr.Textbox(label="固有名詞・専門用語（スペース / 読点区切り）",
                                     value=" ".join(base.context_terms), placeholder="田中 山田 Claude")
                with gr.Row():
                    diarize = gr.Checkbox(label="話者分離", value=bool(base.diarize) and diar_ready(),
                                          interactive=diar_ready(),
                                          info=None if diar_ready() else "① で pyannote を入れると使えます")
                    num_speakers = gr.Number(label="人数（0 = 自動）", value=int(base.num_speakers or 0), precision=0)
                speaker_names = gr.Textbox(label="話者名（例: 話者A=田中, 話者B=佐藤）", value=base.speaker_names)
                btn = gr.Button("文字起こし", variant="primary")
            with gr.Column(scale=2):
                summary = gr.Markdown()
                text = gr.Textbox(label="結果", lines=18, max_lines=40, buttons=["copy"]) if _has_buttons(gr) \
                    else gr.Textbox(label="結果", lines=18, max_lines=40, show_copy_button=True)
                files = gr.File(label="ダウンロード", file_count="multiple")
                with gr.Accordion("ログ", open=False):
                    logs = gr.Textbox(lines=12, max_lines=12, show_label=False)
        btn.click(run, [f_in, mic, model, language, context, diarize, num_speakers, speaker_names],
                  [text, files, summary, logs])
    return demo


def _has_buttons(gr) -> bool:
    """gradio 6 は Textbox(buttons=[...])、5 以前は show_copy_button"""
    import inspect

    try:
        return "buttons" in inspect.signature(gr.Textbox.__init__).parameters
    except Exception:
        return False


_DEMO: Dict[str, Any] = {}


def launch(sess: Session, base: Optional[Settings] = None, *, share: bool = False, port: int = 7860,
           height: int = 900, password: str = ""):
    """立ち上げる(もう動いていたら止めてから)。Colab ではセルの下に表示し、別タブで開くリンクも出す"""
    import gradio as gr  # noqa: F401

    stop()
    demo = build(sess, base)
    demo.queue(default_concurrency_limit=1)
    auth = ("asr", password) if password else None
    kw: Dict[str, Any] = dict(share=share, server_name="0.0.0.0" if not share else None, server_port=port,
                              prevent_thread_lock=True, quiet=True, auth=auth, allowed_paths=[OUT_ROOT])
    in_colab = runtime.in_colab() and "google.colab" in sys.modules and hasattr(sys.modules["google.colab"], "output")
    if in_colab and not share:
        kw["inline"] = False  # Colab のプロキシ経由で自分で表示する(下)
    demo.launch(**{k: v for k, v in kw.items() if v is not None})
    _DEMO["demo"] = demo
    if share and getattr(demo, "share_url", None):
        log(f"🌐 公開 URL（72時間・だれでも開けるので扱いに注意）: {demo.share_url}")
    elif in_colab:
        from google.colab import output  # type: ignore

        log("🌐 下の画面で使えます。別タブで開くときは次のリンクを押してください")
        output.serve_kernel_port_as_window(port, anchor_text="別タブで開く")
        output.serve_kernel_port_as_iframe(port, height=height)
    else:
        log(f"🌐 http://127.0.0.1:{port}/ で開けます")
    return demo


def stop() -> None:
    demo = _DEMO.pop("demo", None)
    if demo is not None:
        try:
            demo.close()
        except Exception:
            pass
