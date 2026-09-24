# -*- coding: utf-8 -*-
"""asrkit.pipeline — ノートブックから呼ぶ高レベルの処理

  sess = Session()
  sess.run(inputs, settings)            # 文字起こし(複数ファイル可)
  sess.compare(src, [プリセット...], settings, start=0, duration=300, reference="")  # モデル比較
"""
from __future__ import annotations

import dataclasses
import glob
import hashlib
import html
import json
import os
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import core, runtime
from .engines import ALIGNER_LANGS, lang_code, lang_name
from .presets import Preset, custom_preset, find_preset

VERSION = "3.0.0"
log = runtime.log


# =====================================================================
# 設定
# =====================================================================


@dataclass
class Settings:
    # 入出力
    output_dir: str = ""  # 空: 音源と同じフォルダ / フォルダ: その中 / ファイル名: そのパス(拡張子は無視)
    formats: Tuple[str, ...] = ("txt", "srt", "json")  # txt srt vtt json csv md plain rttm
    skip_existing: bool = False
    # モデル
    preset: str = "qwen3-1.7b"
    custom_engine: str = ""
    custom_model: str = ""
    language: str = "Japanese"  # "auto" で自動判定
    batch_size: int = 0  # 0 = GPU から自動
    max_new_tokens: int = 1024
    dtype: str = "auto"
    attn: str = "auto"
    # タイムスタンプ
    aligner: bool = True
    aligner_model: str = "Qwen/Qwen3-ForcedAligner-0.6B"
    prefer_aligner: bool = True  # モデル自身のタイムスタンプがあってもアライナーを優先
    # 音声
    channel: str = "mix"  # mix / left / right
    af: str = ""  # ffmpeg フィルタ
    # 区間検出
    vad: str = "fireredvad"  # fireredvad / silero / energy / none
    vad_threshold: float = 0.4
    vad_min_speech: float = 0.2
    vad_min_silence: float = 0.2
    vad_pad: float = 0.2
    energy_top_db: float = 45.0
    # クリップの区切り方。None = 自動(モデルのおすすめ。ふつうは 30 秒まで・6 秒以下の無音はつなぐ)
    max_clip: Optional[float] = None
    max_gap: Optional[float] = None
    overlap: float = 1.0
    # context
    context_label: str = "固有名詞・専門用語"
    context_terms: List[str] = field(default_factory=list)
    context_extra: str = ""
    # リトライ
    retry: bool = True
    # テキスト整形
    replacements: str = ""
    fillers: str = "off"  # off / safe / more
    halfwidth: bool = True
    # 字幕
    cue_max_chars: int = 30
    cue_max_dur: float = 6.0
    cue_gap: float = 0.8
    speaker_fmt: str = "{speaker}: {text}"
    # 話者分離
    diarize: bool = False
    diar_model: str = "pyannote/speaker-diarization-community-1"
    num_speakers: int = 0
    min_speakers: int = 0
    max_speakers: int = 0
    speaker_names: str = ""
    speaker_style: str = "話者A"
    smooth_speakers: bool = True
    # vLLM
    vllm_gpu_mem: float = 0.0  # 0 = 自動
    vllm_concurrency: int = 64
    # セカンドオピニオン: 最後まで怪しいクリップだけ別モデルでも読んで、良い方を採用(元の結果も JSON に残す)
    second_opinion: str = ""  # プリセットのキー or 表示名。空ならしない
    # その他
    cache: bool = True
    review: bool = True  # 要確認リスト(_review.md)を書き出し、ノートブックに音声つきで表示

    def replace(self, **kw: Any) -> "Settings":
        return dataclasses.replace(self, **kw)


DEFAULT_MAX_CLIP = 30.0
DEFAULT_MAX_GAP = 6.0


def auto_num(v: Any) -> Optional[float]:
    """フォームの「自動」/空欄 → None、それ以外は数値(「15秒」のような単位つきも可。0 は 0 のまま)"""
    s = str(v if v is not None else "").strip()
    if s.lower() in ("", "自動", "auto", "none"):
        return None
    return float(s.rstrip("秒sS").strip())


def clip_settings(st: "Settings", p: Preset) -> "Settings":
    """max_clip / max_gap が自動(None)なら、モデルのおすすめ値(無ければ 30 秒 / 6 秒)を入れた Settings を返す"""
    mc = st.max_clip if st.max_clip else (p.max_clip or DEFAULT_MAX_CLIP)
    mg = st.max_gap if st.max_gap is not None else (p.max_gap if p.max_gap is not None else DEFAULT_MAX_GAP)
    return st.replace(max_clip=float(mc), max_gap=float(mg))


def _load_sec(asr_info: Dict[str, Any]) -> float:
    """文字起こしの段階で、モデルの読み込み(vLLM はサーバーの起動)を待った秒数。使い回したときは 0。
    区間検出と並べて裏で読み込んだぶんは合計に効かないので、待った時間(load_wait)があればそちらを使う"""
    if asr_info.get("reused") or asr_info.get("cached"):
        return 0.0
    if asr_info.get("load_wait") is not None:
        return float(asr_info["load_wait"])
    return float(asr_info.get("load_sec") or 0.0)


def timing_breakdown(T: Dict[str, float], asr_info: Dict[str, Any]) -> str:
    """合計時間の内訳(合計と足し算が合うように)。「推論だけ」の速さと全体の速さの差がどこから来るかを見せる"""
    load = min(_load_sec(asr_info), float(T.get("asr") or 0.0))
    parts = [("音声の変換", T.get("audio")), ("区間検出", T.get("vad")), ("モデルの読み込み待ち", load),
             ("文字起こし", max(0.0, float(T.get("asr") or 0.0) - load)), ("セカンドオピニオン", T.get("second_opinion")),
             ("タイムスタンプ", T.get("align")), ("話者分離の待ち", T.get("diar_wait"))]
    known = sum(float(v or 0.0) for _, v in parts)
    if T.get("total"):
        parts.append(("書き出しなど", max(0.0, float(T["total"]) - known)))
    return "・".join(f"{k} {core.fmt_dur(float(v))}" for k, v in parts if v is not None and float(v) >= 0.05)


def _sha(obj: Any, n: int = 16) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()[:n]


# =====================================================================
# 入力ファイルの解決・出力先
# =====================================================================


def resolve_inputs(spec: str, recursive: bool = False) -> List[str]:
    """ファイル / フォルダ / ワイルドカード / 改行・カンマ区切りの複数指定 → 音声ファイルの一覧"""
    out: List[str] = []
    parts: List[str] = []
    for line in (spec or "").splitlines():
        line = line.strip().strip('"').strip("'")
        if not line:
            continue
        # カンマ区切りの複数指定にも対応(ただしカンマを含む実在パスはそのまま)
        if "," in line and not os.path.exists(line):
            parts += [x.strip().strip('"').strip("'") for x in line.split(",") if x.strip()]
        else:
            parts.append(line)
    for p in parts:
        if os.path.isdir(p):
            pat = "**/*" if recursive else "*"
            cands = sorted(glob.glob(os.path.join(glob.escape(p), pat), recursive=recursive))
            out += [c for c in cands if os.path.isfile(c) and c.lower().endswith(core.AUDIO_EXTS)]
        elif any(ch in p for ch in "*?["):
            out += sorted(c for c in glob.glob(p, recursive=True) if os.path.isfile(c))
        elif os.path.isfile(p):
            out.append(p)
        else:
            raise FileNotFoundError(f"音源が見つかりません: {p}")
    seen, uniq = set(), []
    for p in out:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def output_base(src: str, output_dir: str, many: bool = False) -> str:
    stem = os.path.splitext(os.path.basename(src))[0]
    od = (output_dir or "").strip()
    if not od:
        return os.path.splitext(src)[0]
    if many or os.path.isdir(od) or od.endswith(("/", os.sep)) or not os.path.splitext(od)[1]:
        return os.path.join(od, stem)
    return os.path.splitext(od)[0]


# =====================================================================
# VAD(カーネル側で動かす。どれも軽い)
# =====================================================================


def vad_fireredvad(w: core.Wav16, st: Settings) -> List[Tuple[float, float]]:
    """FireRedVAD。長い音声でもメモリを食わないよう 300 秒ごとに特徴量→確率を出してから後処理"""
    import torch
    from fireredvad import FireRedVad, FireRedVadConfig
    from huggingface_hub import snapshot_download

    repo = snapshot_download("FireRedTeam/FireRedVAD", allow_patterns=["VAD/*"])
    cfg = FireRedVadConfig(
        use_gpu=False, smooth_window_size=5, speech_threshold=float(st.vad_threshold),
        min_speech_frame=max(1, int(round(st.vad_min_speech * 100))),
        max_speech_frame=max(200, int((st.max_clip or DEFAULT_MAX_CLIP) * 100)),  # 長い発話は“間”の所で VAD 自身に割らせる
        min_silence_frame=max(1, int(round(st.vad_min_silence * 100))),
        merge_silence_frame=0, extend_speech_frame=0, chunk_max_frame=30000,
    )
    vad = FireRedVad.from_pretrained(os.path.join(repo, "VAD"), cfg)
    x = w.int16()
    L = 30000 * 160  # 300 秒ぶん(フレームの位置がずれないよう 160 の倍数)
    probs: List[Any] = []
    with torch.no_grad():
        for a in range(0, len(x), L):
            seg = np.asarray(x[a: a + L + 240])
            if seg.shape[0] < 400:
                break
            feats, _ = vad.audio_feat.extract((seg, core.SR))
            if feats.shape[0] == 0:
                continue
            if a + L < len(x):
                feats = feats[:30000]
            p, _ = vad.vad_model.forward(feats.unsqueeze(0))
            probs.append(p.detach().cpu().reshape(-1))
    if not probs:
        return []
    pr = torch.cat(probs).tolist()
    dec = vad.vad_postprocessor.process(pr)
    return [(float(s), float(e)) for s, e in vad.vad_postprocessor.decision_to_segment(dec, w.duration)]


def vad_silero(w: core.Wav16, st: Settings) -> List[Tuple[float, float]]:
    import torch
    from silero_vad import get_speech_timestamps, load_silero_vad

    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    model = load_silero_vad()
    x = torch.from_numpy(w.get(0, w.duration))
    ts = get_speech_timestamps(
        x, model, sampling_rate=core.SR, threshold=float(st.vad_threshold),
        min_speech_duration_ms=int(st.vad_min_speech * 1000), min_silence_duration_ms=int(st.vad_min_silence * 1000),
        speech_pad_ms=30, max_speech_duration_s=float(st.max_clip or DEFAULT_MAX_CLIP), return_seconds=True,
    )
    return [(float(t["start"]), float(t["end"])) for t in ts]


# =====================================================================
# セッション
# =====================================================================


class Session:
    """カーネルに1つだけ作って使い回す(モデルを読み込んだまま、設定だけ変えて何度でも回せる)"""

    def __init__(self) -> None:
        self.h = runtime.Handles()
        self.gpu = runtime.gpu_info()
        self.hf_token = runtime.get_secret("HF_TOKEN")
        self._vad_cache: Dict[str, List[Tuple[float, float]]] = {}
        self.last: Dict[str, Any] = {}
        # タイムスタンプ付けと話者分離のエンジン(テストではダミーに差し替える)
        self.aligner_env, self.aligner_kind = "qwen", "aligner"
        self.diar_env, self.diar_kind = "pyannote", "pyannote"

    # ---------------------------------------------------------- 小物
    def extra_env(self) -> Dict[str, str]:
        env = {}
        if self.hf_token:
            env["HF_TOKEN"] = self.hf_token
        return env

    def preset_of(self, st: Settings) -> Preset:
        if st.preset.startswith("カスタム") or st.preset == "custom":
            if not st.custom_model.strip():
                raise ValueError("カスタムを選んだときは custom_model に Hugging Face のモデル ID を入れてください")
            return custom_preset(st.custom_engine or "hf-pipeline", st.custom_model)
        p = find_preset(st.preset)
        if p is None:
            raise ValueError(f"プリセットが見つかりません: {st.preset}")
        return p

    def batch_size(self, st: Settings, p: Preset) -> int:
        if st.batch_size and st.batch_size > 0:
            return int(st.batch_size)
        model_gb = {"qwen": 5.0, "faster-whisper": 4.0, "nemo": 3.0, "hf-pipeline": 4.0, "hf-speechlm": 10.0,
                    "cohere": 5.0, "granite": 5.0, "vibevoice": 18.0}.get(p.engine, 5.0)
        per = 0.45 * max(1.0, (st.max_clip or DEFAULT_MAX_CLIP) / 30.0)
        bs = runtime.auto_batch_size(self.gpu, model_gb, per, cap=64 if p.engine == "qwen" else 32)
        if p.engine == "faster-whisper":
            bs = min(bs, 8)
        return bs

    def free(self, what: str = "all") -> None:
        if what in ("all", "vllm"):
            self.h.stop_vllm()
        if what == "all":
            self.h.close_all()
        log("🧹 VRAM を解放しました")

    # ---------------------------------------------------------- 前処理
    def prepare_audio(self, src: str, st: Settings, excerpt: Optional[Tuple[float, float]] = None) -> Tuple[str, str]:
        runtime._mkdirs()
        stt = os.stat(src)
        key = _sha([os.path.abspath(src), stt.st_size, int(stt.st_mtime), st.channel, st.af, excerpt])
        dst = os.path.join(runtime.WORK_DIR, f"{key}.wav")
        if not os.path.exists(dst):
            tmp = dst + ".tmp.wav"
            core.ffmpeg_to_wav16(src, tmp, channel=st.channel, af=st.af,
                                 start=excerpt[0] if excerpt else None, duration=excerpt[1] if excerpt else None)
            os.replace(tmp, dst)
            # 作業ファイルは新しい 6 個だけ残す(3時間で 350MB 程度あるため)
            olds = sorted(glob.glob(os.path.join(runtime.WORK_DIR, "*.wav")), key=os.path.getmtime)
            for p in olds[:-6]:
                if p != dst:
                    try:
                        os.remove(p)
                    except OSError:
                        pass
        return dst, key

    def vad(self, w: core.Wav16, wav_key: str, st: Settings, db: np.ndarray) -> List[Tuple[float, float]]:
        ck = _sha([wav_key, st.vad, st.vad_threshold, st.vad_min_speech, st.vad_min_silence, st.energy_top_db, st.max_clip])
        if ck in self._vad_cache:
            return self._vad_cache[ck]
        method = st.vad
        segs: List[Tuple[float, float]] = []
        try:
            if method == "fireredvad":
                segs = vad_fireredvad(w, st)
            elif method == "silero":
                segs = vad_silero(w, st)
            elif method == "energy":
                segs = core.energy_vad(db, top_db=st.energy_top_db, min_speech=st.vad_min_speech,
                                       min_silence=max(0.3, st.vad_min_silence))
            else:
                segs = [(0.0, w.duration)]
        except Exception as e:
            log(f"⚠️ VAD({method}) に失敗したので簡易VAD(energy)で続けます: {type(e).__name__}: {e}")
            segs = core.energy_vad(db, top_db=st.energy_top_db)
        speech = sum(e - s for s, e in segs)
        if w.duration > 5 and speech < 0.02 * w.duration:
            # 発話がほぼ見つからない(小さな声・遠いマイク・VAD の相性)→ 全体を静かな所で区切って全部読ませる
            log(f"⚠️ VAD({method}) で発話がほとんど見つかりませんでした({core.fmt_dur(speech)})。"
                "音声全体を静かな所で区切って文字起こしします(④の vad_threshold を下げるか、音声フィルタも試してください)")
            segs = [(0.0, w.duration)]
        self._vad_cache[ck] = segs
        return segs

    # ---------------------------------------------------------- 各ステージ
    def _asr(self, p: Preset, w: core.Wav16, wav_path: str, clips: List[core.Clip], st: Settings,
             lang: Optional[str], ctx: str, terms: List[str], db: np.ndarray, cache_path: Optional[str],
             bs: int, quiet: bool = False, preload: Optional[Future] = None) -> Tuple[List[core.ClipResult], Dict[str, Any]]:
        from tqdm.auto import tqdm

        policy = core.RetryPolicy(enabled=st.retry, punctuates=p.punctuates, lang=lang or "")
        if not p.context:
            ctx, terms = "", []
        done: Dict[int, core.ClipResult] = {}
        if cache_path and os.path.exists(cache_path):
            try:
                with open(cache_path, encoding="utf-8") as f:
                    for line in f:
                        d = json.loads(line)
                        if d.get("final"):
                            res = [core.ClipResult.from_dict(x) for x in d["results"]]
                            log(f"♻️ キャッシュから文字起こし結果を復元しました ({len(res)} クリップ)")
                            return res, {"cached": True}
                        r = core.ClipResult.from_dict(d)
                        done[r.clip.id] = r
                if done:
                    log(f"♻️ 途中から再開します(済み {len(done)}/{len(clips)} クリップ)")
            except Exception:
                done = {}
        cache_f = open(cache_path, "a", encoding="utf-8") if cache_path else None
        bar = tqdm(total=len(clips), desc="文字起こし", unit="clip", disable=quiet, dynamic_ncols=True)
        bar.update(len(done))
        stage_msgs = {"retry": "怪しいクリップを context なしで再推論", "split": "まだ怪しいクリップを分割して再推論"}

        def on_progress(stage: str, n: int, total: int, part: List[core.ClipResult]) -> None:
            if stage == "asr":
                if cache_f:
                    for r in part:
                        cache_f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
                    cache_f.flush()
                bar.n = n
                bar.refresh()
            elif not quiet:
                bar.set_postfix_str(f"{stage_msgs.get(stage, stage)} {n}/{total}")

        info: Dict[str, Any] = {}
        t0 = time.time()
        try:
            # モデルの読み込み(区間検出のあいだに裏で始めていれば、その残りを待つだけ)
            info = self._wait_or_load(preload, p, st, bs)
            info["load_wait"] = round(time.time() - t0, 2)
            t0 = time.time()  # 速度比較は読み込み(vLLM はサーバーの起動)を除いて測る
            if p.engine == "vllm":
                server = self.h.vllm
                if server is None or not server.alive():
                    raise runtime.WorkerError("vLLM サーバーが動いていません")
                fn = server.transcribe_fn(lang_code(lang), concurrency=st.vllm_concurrency)
                res = core.run_asr(clips, w.get, fn, context=ctx, ctx_terms=terms, ctx_label=st.context_label,
                                   policy=policy, batch_size=st.vllm_concurrency, db=db, on_progress=on_progress, done=done)
            else:
                wk = self.h.worker(p.env)
                r = wk.call(
                    "transcribe", key="asr", wav=wav_path, clips=[c.to_dict() for c in clips],
                    done=[d.to_dict() for d in done.values()], language=lang, context=ctx, ctx_terms=terms,
                    ctx_label=st.context_label, policy=asdict(policy), batch_size=bs,
                    on_event=lambda m: on_progress(m.get("stage", ""), int(m.get("done", 0)), int(m.get("total", 0)),
                                                   [core.ClipResult.from_dict(d) for d in m.get("results") or []]),
                )
                res = [core.ClipResult.from_dict(d) for d in r["results"]]
                info = dict(info, gpu_peak_gb=r.get("gpu_peak_gb"))
        finally:
            bar.close()
            if cache_f:
                cache_f.close()
        info["asr_sec"] = round(time.time() - t0, 2)
        if cache_path:
            with open(cache_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"final": True, "results": [r.to_dict() for r in res]}, ensure_ascii=False) + "\n")
        return res, info

    def _load_asr(self, p: Preset, st: Settings, bs: int) -> Dict[str, Any]:
        """ASR のモデルを読み込む(vLLM はサーバーを起動)。load_sec = 読み込みの秒数、reused = 読み込み済みを使い回した"""
        if p.engine == "vllm":
            g = self.gpu
            mem = st.vllm_gpu_mem or (min(0.6, max(0.3, (g.mem_gb - 14) / max(g.mem_gb, 1))) if g.ok else 0.5)
            t = time.time()
            server = self.h.vllm_server("vllm", p.model, gpu_mem=mem, extra_args=p.vllm_args, extra_env=self.extra_env())
            waited = time.time() - t
            return {"load_sec": server.start_sec if server.start_sec is not None else round(waited, 1),
                    "reused": waited < 1.0}
        opts: Dict[str, Any] = dict(p.options)
        opts.update(model=p.model, batch_size=bs)
        if p.engine in ("qwen", "hf-pipeline", "hf-speechlm", "cohere", "granite", "vibevoice"):
            opts.update(dtype=st.dtype)
        if p.engine == "qwen":
            opts.update(attn=st.attn, max_new_tokens=st.max_new_tokens)
        return dict(self.h.ensure_loaded(p.env, "asr", p.engine, opts, self.extra_env(), exclusive_group="asr"))

    def _wait_or_load(self, preload: Optional[Future], p: Preset, st: Settings, bs: int) -> Dict[str, Any]:
        if preload is not None:
            try:
                return dict(preload.result())  # 裏で読み込み中なら、ここで終わるのを待つ
            except Exception as e:  # もう一度ふつうに読み込む(だめならそこでエラーを出す)
                log(f"(裏での読み込みに失敗したので、もう一度読み込みます: {type(e).__name__}: {str(e)[:200]})")
        return self._load_asr(p, st, bs)

    def _aligner_opts(self, st: Settings, bs: int) -> Dict[str, Any]:
        return {"model": st.aligner_model, "dtype": st.dtype, "attn": st.attn, "batch_size": min(bs, 16)}

    @staticmethod
    def _background(fn: Callable[..., Any], *args: Any) -> Future:
        ex = ThreadPoolExecutor(max_workers=1)
        fut = ex.submit(fn, *args)
        ex.shutdown(wait=False)
        return fut

    def _align(self, p: Preset, results: List[core.ClipResult], wav_path: str, st: Settings, lang: Optional[str],
               cache_path: Optional[str], bs: int, quiet: bool = False) -> Dict[int, List[List[Any]]]:
        from tqdm.auto import tqdm

        if not st.aligner:
            return {}
        items = []
        for r in results:
            if not r.text or (r.words and not st.prefer_aligner):
                continue
            L = lang_name(r.language) or (lang or "")
            if L not in ALIGNER_LANGS:
                continue
            if r.clip.dur > 179:
                continue
            items.append({"id": r.clip.id, "start": r.clip.start, "end": r.clip.end, "text": r.text, "language": L})
        if not items:
            return {}
        key = _sha([st.aligner_model, [(d["id"], round(d["start"], 3), round(d["end"], 3), d["text"], d["language"]) for d in items]])
        if cache_path:
            cache_path = f"{cache_path}.{key}.align.json"
            if os.path.exists(cache_path):
                try:
                    return {int(k): v for k, v in json.load(open(cache_path, encoding="utf-8")).items()}
                except Exception:
                    pass
        if not os.path.exists(runtime.env_python(self.aligner_env)):
            log(f"⚠️ アライナー用の環境({self.aligner_env})がないので、タイムスタンプはモデル固有/概算になります")
            return {}
        self.h.ensure_loaded(self.aligner_env, "aligner", self.aligner_kind, self._aligner_opts(st, bs), self.extra_env())
        bar = tqdm(total=len(items), desc="タイムスタンプ", unit="clip", disable=quiet, dynamic_ncols=True)
        try:
            r = self.h.worker(self.aligner_env).call(
                "align", key="aligner", wav=wav_path, items=items, batch_size=min(bs, 16),
                on_event=lambda m: (setattr(bar, "n", int(m.get("done", 0))), bar.refresh()),
            )
        finally:
            bar.close()
        aligned = {int(k): v for k, v in r["aligned"].items()}
        if cache_path:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump({str(k): v for k, v in aligned.items()}, f, ensure_ascii=False)
        return aligned

    def _second_opinion(self, results: List[core.ClipResult], w: core.Wav16, wav_path: str, st: Settings,
                        lang: Optional[str], ctx: str, terms: List[str], db: np.ndarray, bs: int, quiet: bool,
                        segs: Sequence[Tuple[float, float]] = ()) -> int:
        """最後まで怪しいクリップだけ別のモデルで読み直し、怪しさが減るなら差し替える。元の結果は attempts に残す"""
        p2 = find_preset(st.second_opinion)
        main = self.preset_of(st)
        if p2 is None or p2.key == main.key:
            log(f"⚠️ セカンドオピニオンのモデルが見つからないか、本命と同じです: {st.second_opinion}")
            return 0
        if p2.engine == "vllm" and main.engine == "vllm":
            log("⚠️ vLLM どうしのセカンドオピニオンはできません(サーバーは1つだけ)")
            return 0
        if p2.engine != "vllm" and not os.path.exists(runtime.env_python(p2.env)):
            log(f"⚠️ セカンドオピニオン用の環境 {p2.env} が未インストールです(① で入れてください)")
            return 0
        bad = [r for r in results if r.flags]
        if not bad:
            return 0
        if not quiet:
            log(f"🩺 セカンドオピニオン: 怪しい {len(bad)} クリップを {p2.label} でも読みます")
        policy = core.RetryPolicy(enabled=False, punctuates=p2.punctuates, lang=lang or "")
        ctx_flags = {"context_label", "context_echo"}
        # context の復唱が疑われるクリップは context なしで、それ以外は context ありで読ませる
        groups = {
            (ctx if p2.context else ""): [r.clip for r in bad if not (set(r.flags) & ctx_flags)],
            "": [r.clip for r in bad if set(r.flags) & ctx_flags],
        } if ctx and p2.context else {"": [r.clip for r in bad]}
        res2: List[core.ClipResult] = []
        server = None
        if p2.engine == "vllm":
            server = self.h.vllm_server("vllm", p2.model, gpu_mem=st.vllm_gpu_mem or 0.4, extra_args=p2.vllm_args,
                                        extra_env=self.extra_env())
        else:
            opts: Dict[str, Any] = dict(p2.options)
            opts.update(model=p2.model, batch_size=bs)
            if p2.engine in ("qwen", "hf-pipeline", "hf-speechlm", "cohere", "granite", "vibevoice"):
                opts.update(dtype=st.dtype)
            self.h.ensure_loaded(p2.env, "asr2", p2.engine, opts, self.extra_env(), exclusive_group="asr2")
        # このモデルのおすすめより長いクリップは、発話の切れ目で区切り直して読ませ、あとで本文をつなぐ
        # (長いクリップで発話を読み飛ばすモデルだと、飛ばした結果のほうが「怪しくない」と判定されてしまうため)
        lim = float(p2.max_clip) if p2.max_clip else 0.0
        gap = float(p2.max_gap) if p2.max_gap is not None else DEFAULT_MAX_GAP
        pieces_of: Dict[int, List[core.Clip]] = {}
        next_id = max((r.clip.id for r in results), default=0) + 1
        for c in [r.clip for r in bad]:
            if lim and c.dur > lim + 0.5:
                inner = [(max(s, c.start), min(e, c.end)) for s, e in segs if e > c.start and s < c.end] or [(c.start, c.end)]
                cut = core.build_clips(inner, c.end, max_clip=lim, max_gap=gap, pad=min(st.vad_pad, 0.2), overlap=0.0, db=db)
                for k, x in enumerate(cut):
                    x.id, x.parent, x.start = next_id + k, c.id, max(x.start, c.start)
                next_id += len(cut)
                pieces_of[c.id] = cut or [c]
            else:
                pieces_of[c.id] = [c]
        for use_ctx, clips2 in groups.items():
            if not clips2:
                continue
            todo = [x for c in clips2 for x in pieces_of[c.id]]
            if server is not None:
                res2 += core.run_asr(todo, w.get, server.transcribe_fn(lang_code(lang), st.vllm_concurrency),
                                     context=use_ctx, policy=policy, batch_size=st.vllm_concurrency)
            else:
                r = self.h.worker(p2.env).call("transcribe", key="asr2", wav=wav_path, clips=[c.to_dict() for c in todo],
                                               language=lang, context=use_ctx, policy=asdict(policy), batch_size=bs)
                res2 += [core.ClipResult.from_dict(d) for d in r["results"]]
        by_id = {x.clip.id: x for x in res2}
        adopted = 0
        inflating = {"repetition", "too_dense", "context_echo", "context_label"}  # 元の本文が水増しされている疑い
        for r in bad:
            parts = [by_id.get(x.id) for x in pieces_of.get(r.clip.id, [])]
            if not parts or any(a is None for a in parts):
                continue
            if len(parts) == 1:
                text2, words2 = parts[0].text, parts[0].words
            else:  # 区切り直したもの: 本文をつなぐ(時刻はアライナーで付け直す)
                text2, words2 = core.join_texts([a.text for a in parts]), None
            f2 = core.quality_flags(text2, r.clip.dur, r.clip.speech, ctx_terms=terms, ctx_label=st.context_label,
                                    punctuates=p2.punctuates, lang=lang or "")
            att: Dict[str, Any] = {"model": p2.key, "text": text2, "flags": f2}
            if len(parts) > 1:
                att["pieces"] = len(parts)
            r.attempts.append(att)
            # 元より大きく短い結果は、元が水増しを疑われているとき以外は採らない(読み飛ばしの疑い)
            too_short = core.core_len(text2) < 0.6 * core.core_len(r.text) and not (set(r.flags) & inflating)
            if len(f2) < len(r.flags) and not too_short:
                r.attempts.append({"adopted": p2.key})
                r.text, r.flags, r.words = text2, f2, words2
                adopted += 1
            elif too_short and len(f2) < len(r.flags):
                att["rejected"] = "元より大きく短い(読み飛ばしの疑い)"
        if not quiet:
            log(f"      → {adopted} クリップを {p2.label} の結果に差し替えました(元の結果は JSON と要確認リストに残っています)")
        return adopted

    def _diar_start(self, wav_path: str, st: Settings, cache_path: Optional[str]) -> Optional[Future]:
        if not st.diarize:
            return None
        if not self.hf_token and self.diar_kind == "pyannote":
            log("⚠️ HF_TOKEN が見つからないので話者分離をスキップします(Colab のシークレットに登録してください)")
            return None
        key = _sha([st.diar_model, st.num_speakers, st.min_speakers, st.max_speakers])
        fut: Future = Future()
        if cache_path:
            cache_path = f"{cache_path}.{key}.diar.json"
            if os.path.exists(cache_path):
                try:
                    fut.set_result(json.load(open(cache_path, encoding="utf-8")))
                    return fut
                except Exception:
                    pass
        if not os.path.exists(runtime.env_python(self.diar_env)):
            log(f"⚠️ 話者分離の環境({self.diar_env})がありません。セットアップのセルで入れてください")
            return None
        env = self.extra_env()

        def job() -> Dict[str, Any]:
            # 読み込みから裏で(区間検出・文字起こしと並べて走る。失敗しても文字起こしは続ける)
            self.h.ensure_loaded(self.diar_env, "diar", self.diar_kind, {"model": st.diar_model}, env)
            r = self.h.worker(self.diar_env).call(
                "diarize", key="diar", wav=wav_path, num_speakers=st.num_speakers,
                min_speakers=st.min_speakers, max_speakers=st.max_speakers)
            d = {"turns": r["turns"], "exclusive": r.get("exclusive"), "seconds": r.get("seconds")}
            if cache_path:
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(d, f)
            return d

        ex = ThreadPoolExecutor(max_workers=1)
        fut2 = ex.submit(job)
        ex.shutdown(wait=False)
        return fut2

    # ---------------------------------------------------------- 1ファイル
    def transcribe_file(self, src: str, st: Settings, *, excerpt: Optional[Tuple[float, float]] = None,
                        out_base: Optional[str] = None, many: bool = False, quiet: bool = False) -> Dict[str, Any]:
        T: Dict[str, float] = {}
        t_all = time.time()
        p = self.preset_of(st)
        st = clip_settings(st, p)  # クリップ長・無音の「自動」をモデルのおすすめ値に
        lang = None if (st.language or "").lower() in ("auto", "", "自動") else st.language
        base = out_base or output_base(src, st.output_dir, many)
        os.makedirs(os.path.dirname(base) or ".", exist_ok=True)
        stem = os.path.basename(base)
        if st.skip_existing and all(os.path.exists(base + _ext(f)) for f in st.formats):
            log(f"⏭️ 出力が揃っているのでスキップ: {stem}")
            return {"skipped": True, "base": base}
        cache_dir = os.path.join(os.path.dirname(base) or ".", ".asr_v3_cache") if st.cache else None
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

        # 1) 音声
        t = time.time()
        wav_path, wav_key = self.prepare_audio(src, st, excerpt)
        w = core.Wav16(wav_path)
        db = core.frame_db(w)
        T["audio"] = time.time() - t
        if not quiet:
            log(f"[1/6] 音声 {core.fmt_dur(w.duration)} を読み込み ({core.fmt_dur(T['audio'])})")

        # 2) 話者分離(裏で並行して走らせる。音声さえあれば始められるので、区間検出より前に)
        diar_key = _sha([wav_key])
        diar_cache = os.path.join(cache_dir, f"{stem}.{diar_key}") if cache_dir else None
        diar_fut = self._diar_start(wav_path, st, diar_cache)

        # 区間検出のあいだに、裏で ASR のモデル(とアライナー)を読み込んでおく(最初の 1 件が速くなる)
        bs = self.batch_size(st, p)
        asr_pre = self._background(self._load_asr, p, st, bs)
        if st.aligner and self.aligner_env != p.env and os.path.exists(runtime.env_python(self.aligner_env)):
            self._background(self.h.ensure_loaded, self.aligner_env, "aligner", self.aligner_kind,
                             self._aligner_opts(st, bs), self.extra_env())

        # 3) 区間検出 → クリップ
        t = time.time()
        segs = self.vad(w, wav_key, st, db)
        max_clip = min(float(st.max_clip), 170.0) if st.aligner else float(st.max_clip)
        clips = core.build_clips(segs, w.duration, max_clip=max_clip, max_gap=st.max_gap, pad=st.vad_pad,
                                 overlap=st.overlap, db=db)
        T["vad"] = time.time() - t
        speech = sum(e - s for s, e in segs)
        if not quiet:
            n_hard = sum(1 for c in clips if c.cut)
            log(f"[2/6] 区間検出({st.vad}) 発話 {core.fmt_dur(speech)} → {len(clips)} クリップ"
                f"(上限 {max_clip:g}秒・無音 {st.max_gap:g}秒でつなぐ / 最長 {max((c.dur for c in clips), default=0):.1f}秒,"
                f" ハード切り {n_hard} か所)")

        # 4) 文字起こし
        terms = list(st.context_terms)
        ctx = core.build_context(st.context_label, terms, st.context_extra)
        asr_key = _sha([VERSION, wav_key, [c.to_dict() for c in clips], p.engine, p.model, p.options, lang, ctx,
                        st.retry, st.max_new_tokens])
        asr_cache = os.path.join(cache_dir, f"{stem}.{asr_key}.asr.jsonl") if cache_dir else None
        if not quiet:
            log(f"[3/6] 文字起こし: {p.label} / 言語={lang or '自動'} / バッチ={st.vllm_concurrency if p.engine == 'vllm' else bs}"
                + (f" / context={ctx[:60]}{'…' if len(ctx) > 60 else ''}" if ctx and p.context else ""))
        t = time.time()
        results, asr_info = self._asr(p, w, wav_path, clips, st, lang, ctx, terms, db, asr_cache, bs, quiet,
                                      preload=asr_pre)
        T["asr"] = time.time() - t
        n_retry = sum(1 for r in results if len(r.attempts) > 1)
        n_flag = sum(1 for r in results if r.flags)
        if not quiet:
            if asr_info.get("cached"):
                speed = "キャッシュから復元"
            else:  # 倍速はモデルの読み込み・vLLM の起動を除いた推論だけの時間で(⑥の表と同じ)
                infer = float(asr_info.get("asr_sec") or T["asr"])
                load = 0.0 if asr_info.get("reused") else float(asr_info.get("load_sec") or 0.0)
                speed = (f"推論だけで {core.fmt_dur(infer)} = x{w.duration / max(infer, 1e-6):.0f} 倍速"
                         + (f" / モデルの読み込み {core.fmt_dur(load)}" if load >= 0.5 else ""))
            log(f"      → {len(results)} クリップ / 再推論 {n_retry} / 要確認 {n_flag} ({speed})")

        # 4.5) セカンドオピニオン(怪しいクリップだけ別モデルで)
        if st.second_opinion and n_flag:
            t = time.time()
            n_adopt = self._second_opinion(results, w, wav_path, st, lang, ctx, terms, db, bs, quiet, segs)
            n_flag = sum(1 for r in results if r.flags)
            T["second_opinion"] = time.time() - t
            asr_info["second_opinion_adopted"] = n_adopt

        # 5) タイムスタンプ
        t = time.time()
        aligned = self._align(p, results, wav_path, st, lang, os.path.join(cache_dir, f"{stem}.{asr_key}") if cache_dir else None, bs, quiet)
        words, ts_stats = core.compose_words(results, aligned, segs)
        T["align"] = time.time() - t
        # 本文と字幕の文字数の照合(重なり部分の除去で少し減るのは正常。大きく減ったら警告)
        n_text = sum(core.core_len(r.text) for r in results)
        n_words = sum(core.core_len(w_.word) for w_ in words)
        coverage = n_words / n_text if n_text else 1.0
        ts_stats["coverage"] = round(coverage, 3)
        if n_text and coverage < 0.9:
            log(f"⚠️ 字幕に残った文字が本文の {coverage * 100:.0f}% です(タイムスタンプ付けで欠けた可能性。_review.md を確認してください)")
        n_flag = sum(1 for r in results if r.flags)  # タイムスタンプ付けで付いたフラグ(trimmed)も数える
        if not quiet:
            src_j = {"aligner": "アライナー", "engine": "モデル固有", "approx": "概算"}
            log(f"[4/6] タイムスタンプ {len(words)} 語 (" + ", ".join(f"{src_j.get(k, k)} {v}" for k, v in ts_stats.items()
                                                              if k in src_j) + f", 本文との一致 {coverage * 100:.0f}%)")

        # 6) 話者
        turns: List[List[Any]] = []
        diar_info: Dict[str, Any] = {}
        if diar_fut is not None:
            t = time.time()
            try:
                d = diar_fut.result()
                turns = d.get("turns") or []
                use = d.get("exclusive") or turns
                core.assign_speakers(words, use)
                changed = core.smooth_speakers(words) if st.smooth_speakers else 0
                diar_info = {"speakers": len({k for _, _, k in turns}), "turns": len(turns), "smoothed_words": changed,
                             "seconds": d.get("seconds")}
                if not quiet:
                    log(f"[5/6] 話者分離 {diar_info['speakers']} 人 / {len(turns)} ターン (文単位の補正 {changed} 語)")
            except Exception as e:
                log(f"⚠️ 話者分離に失敗しました(文字起こしは続けます): {e}")
            T["diar_wait"] = time.time() - t
        names = core.speaker_names(words, st.speaker_style, st.speaker_names)

        # 7) 書き出し
        meta = {
            "version": f"asr-v3 {VERSION}",
            "source": os.path.abspath(src),
            "excerpt": list(excerpt) if excerpt else None,
            "duration": round(w.duration, 3),
            "language": lang or "auto",
            "preset": p.key,
            "engine": p.engine,
            "model": p.model,
            "aligner": st.aligner_model if aligned else None,
            "diarization": st.diar_model if turns else None,
            "context": ctx if p.context else "",
            "vad": st.vad,
            "clips": {"n": len(clips), "max_clip": max_clip, "max_gap": st.max_gap},
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "timing_sec": {k: round(v, 2) for k, v in T.items()},
            "asr": asr_info,
            "timestamps": ts_stats,
            "diar": diar_info,
            "gpu": self.gpu.name,
        }
        if turns:
            meta["overlaps"] = core.overlap_regions(turns)  # 同時にしゃべっている時間帯(重なりありの結果から)
        paths, texts = write_outputs(base, words, results, turns, names, st, meta)
        T["total"] = time.time() - t_all
        meta["timing_sec"]["total"] = round(T["total"], 2)
        meta["timing_sec"]["load"] = round(_load_sec(asr_info), 2)
        if not quiet:
            log(f"[6/6] 書き出し完了 (合計 {core.fmt_dur(T['total'])} = 音声の x{w.duration / max(T['total'], 1e-6):.0f} 倍速)")
            log(f"      内訳: {timing_breakdown(T, asr_info)}")
            for k, v in paths.items():
                print(f"   📄 {v}")
        out = {"base": base, "paths": paths, "text": texts.get("plain", ""), "meta": meta, "words": len(words),
               "results": results, "duration": w.duration, "timing": T, "flags": n_flag, "retries": n_retry,
               "wav": wav_path, "names": names}
        self.last = out
        return out

    # ---------------------------------------------------------- 複数ファイル
    def run(self, inputs: Sequence[str], st: Settings, preview_lines: int = 8) -> List[Dict[str, Any]]:
        outs = []
        many = len(inputs) > 1
        for i, src in enumerate(inputs, 1):
            if many:
                print(f"\n========== ({i}/{len(inputs)}) {os.path.basename(src)} ==========")
            else:
                print(f"🎧 {src}")
            try:
                o = self.transcribe_file(src, st, many=many)
            except KeyboardInterrupt:
                log("⏹️ 中断しました(途中までの結果はキャッシュに残っています。もう一度実行すると続きから)")
                raise
            except Exception as e:
                if not many:
                    raise
                log(f"❌ 失敗: {src}: {type(e).__name__}: {e}")
                outs.append({"error": str(e), "src": src})
                continue
            outs.append(o)
            if st.review and o.get("paths", {}).get("review"):
                try:
                    show_review(o, max_items=8)
                except Exception as e:  # 表示だけの失敗で止めない
                    log(f"(要確認の表示に失敗: {e})")
            if preview_lines and o.get("paths", {}).get("txt"):
                with open(o["paths"]["txt"], encoding="utf-8") as f:
                    lines = f.read().splitlines()
                print("\n--- プレビュー ---")
                for l in lines[:preview_lines]:
                    print(l[:200])
                if len(lines) > preview_lines:
                    print(f"… (全 {len(lines)} 行)")
        return outs

    # ---------------------------------------------------------- モデル比較
    def compare(self, src: str, preset_keys: Sequence[str], st: Settings, start: float = 0.0, duration: float = 300.0,
                reference: str = "", out_dir: str = "", keep_loaded: bool = False) -> List[Dict[str, Any]]:
        """同じ区間を複数モデルで文字起こしして、速度・VRAM・(正解があれば)CER を並べる"""
        from IPython.display import HTML, display

        ref = reference
        if ref and os.path.isfile(ref):
            ref = open(ref, encoding="utf-8").read()
        stem = os.path.splitext(os.path.basename(src))[0]
        od = out_dir or os.path.join(os.path.dirname(output_base(src, st.output_dir)) or ".", "compare")
        os.makedirs(od, exist_ok=True)
        rows = []
        for key in preset_keys:
            p = find_preset(key)
            if p is None:
                log(f"⚠️ プリセットが見つかりません: {key}")
                continue
            if not os.path.exists(runtime.env_python(p.env)) and not (p.engine == "vllm" and os.path.exists(runtime.env_dir("vllm"))):
                log(f"⚠️ {p.label}: 環境 {p.env} が未インストールなのでスキップ")
                continue
            print(f"\n===== {p.label} =====")
            st2 = st.replace(preset=p.key, diarize=False, formats=("txt", "srt", "json"))
            t0 = time.time()
            try:
                o = self.transcribe_file(src, st2, excerpt=(start, duration), out_base=os.path.join(od, f"{stem}.{p.key}"), quiet=False)
            except Exception as e:
                log(f"❌ {p.label}: {type(e).__name__}: {str(e)[:500]}")
                rows.append({"preset": p.key, "label": p.label, "error": str(e)[:300]})
                continue
            wall = time.time() - t0
            text = o["text"]
            row = {
                "preset": p.key, "label": p.label, "text": text, "chars": core.core_len(text),
                "asr_sec": o["meta"]["asr"].get("asr_sec") or o["timing"].get("asr"), "wall_sec": wall,
                "load_sec": (o["meta"]["asr"] or {}).get("load_sec"), "gpu_peak_gb": (o["meta"]["asr"] or {}).get("gpu_peak_gb"),
                "flags": o["flags"], "duration": o["duration"], "srt": o["paths"].get("srt"),
                "clips": o["meta"].get("clips"),
            }
            row["x_realtime"] = o["duration"] / max(row["asr_sec"] or wall, 1e-6)
            if ref:
                row["cer"] = core.cer(ref, text)
                row["terms"] = core.term_recall(ref, text, st.context_terms) if st.context_terms else None
                row["numbers"] = core.number_recall(ref, text)
                row["negations"] = core.negation_check(ref, text)
                row["drops"] = core.drop_runs(ref, text)
            elif st.context_terms:
                row["terms_found"] = sum(text.count(t) for t in st.context_terms)
            rows.append(row)
            main = self.preset_of(st) if st.preset else None
            if not keep_loaded and (main is None or p.key != main.key):  # 本番で使うモデルは残しておく
                if p.engine == "vllm":
                    self.h.stop_vllm()
                else:
                    self.h.unload(p.env, "asr")
                    # ほかに使わないワーカーはプロセスごと止めて CPU のメモリを返す(標準の Colab は RAM 12GB ほど)
                    w = self.h.workers.get(p.env)
                    if (w is not None and not w.loaded and p.env not in (self.aligner_env, self.diar_env)
                            and (main is None or p.env != main.env)):
                        self.h.close_worker(p.env)
        # 表
        ok = [r for r in rows if "error" not in r]
        if ok and not ref and len(ok) >= 2:
            base_txt = ok[0]["text"]
            for r in ok[1:]:
                r["diff_vs_first"] = core.cer(base_txt, r["text"])
        display(HTML(compare_table_html(rows, bool(ref))))
        if len(ok) >= 2:
            a, b = (ref, ok[0]) if ref else (ok[0]["text"], ok[1])
            title = f"正解 → {b['label']}" if ref else f"{ok[0]['label']} → {b['label']}"
            display(HTML(f"<details><summary><b>差分: {html.escape(title)}</b>(赤=消えた / 緑=増えた)</summary>"
                         f"<div style='white-space:pre-wrap;line-height:1.7;font-size:14px'>{core.diff_html(a, b['text'])}</div></details>"))
        with open(os.path.join(od, f"{stem}.compare.json"), "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=1, default=str)
        return rows


def _ratio(x: Optional[Tuple[int, int]]) -> str:
    if not x or not x[1]:
        return "-"
    return f"{x[0]}/{x[1]}"


def _neg(x: Optional[Sequence[int]]) -> str:
    if not x:
        return "-"
    hit, tot, extra = x
    return (f"{hit}/{tot}" if tot else "-") + (f" (+{extra})" if extra else "")


def _drops(x: Optional[Sequence[int]]) -> str:
    if not x:
        return "-"
    n, chars = x
    return "なし" if not n else f"<b>{n} か所</b>（{chars} 字）"


def compare_table_html(rows: List[Dict[str, Any]], has_ref: bool) -> str:
    th = ("<tr><th>モデル</th><th>推論</th><th>倍速</th><th>読み込み</th><th>VRAM峰</th>"
          + ("<th>CER</th><th>用語</th><th>数字</th><th>否定</th><th>抜け</th>" if has_ref
             else "<th>1行目との差</th><th>用語の出現</th>")
          + "<th>要確認</th><th>冒頭</th></tr>")
    trs = []
    for r in rows:
        if "error" in r:
            trs.append(f"<tr><td>{html.escape(r['label'])}</td><td colspan=11 style='color:#c00'>{html.escape(r['error'][:200])}</td></tr>")
            continue
        metric = r.get("cer") if has_ref else r.get("diff_vs_first")
        mtxt = "-" if metric is None else f"{metric * 100:.1f}%"
        extra = (f"<td>{_ratio(r.get('terms'))}</td><td>{_ratio(r.get('numbers'))}</td>"
                 f"<td>{_neg(r.get('negations'))}</td><td>{_drops(r.get('drops'))}</td>" if has_ref
                 else f"<td>{r.get('terms_found', '-')}</td>")
        cl = r.get("clips") or {}
        ctxt = (f"<br><small style='color:#666'>{cl.get('n')} クリップ（上限 {cl.get('max_clip'):g}秒・無音 {cl.get('max_gap'):g}秒でつなぐ）</small>"
                if cl.get("max_clip") else "")
        load = r.get("load_sec")
        trs.append(
            f"<tr><td>{html.escape(r['label'])}{ctxt}</td><td>{(r.get('asr_sec') or 0):.1f}秒</td><td>x{r['x_realtime']:.0f}</td>"
            f"<td>{'-' if load is None else f'{load:.0f}秒'}</td>"
            f"<td>{r.get('gpu_peak_gb') or '-'}</td><td><b>{mtxt}</b></td>{extra}<td>{r.get('flags', 0)}</td>"
            f"<td style='max-width:520px'>{html.escape(r['text'][:160])}…</td></tr>")
    return ("<table style='border-collapse:collapse;font-size:13px' border=1 cellpadding=4>" + th + "".join(trs) + "</table>"
            + "<div style='font-size:12px;color:#666'>推論・倍速はモデルの読み込み(vLLM はサーバーの起動)を除いた時間。"
              "読み込みはそのモデルを最後に読み込んだときの時間です。</div>"
            + ("<div style='font-size:12px;color:#666'>CER は句読点・空白を除き NFKC 正規化した文字誤り率(低いほど良い)。"
               "用語 = 正解に出てくる context の語を正しく書けた数、数字 = 正解の数字を同じ形で書けた数(金額・日付の取り違えの目安)、"
               "否定 = 正解の「ない・ません」などが同じ所に残った数(+ は正解に無い否定。意味の反転の目安)、"
               "抜け = 正解の 8 字以上がまとめて抜けた箇所(発話の読み飛ばしの目安)。</div>" if has_ref else
               "<div style='font-size:12px;color:#666'>正解テキストがないので、1行目のモデルとの文字の食い違い率を参考表示しています(どちらが正しいかは分かりません)。</div>"))


def _ext(fmt: str) -> str:
    return {"plain": "_plain.txt", "review": "_review.md"}.get(fmt, "." + fmt)


def write_outputs(base: str, words: List[core.Word], results: List[core.ClipResult], turns: List[List[Any]],
                  names: Dict[str, str], st: Settings, meta: Dict[str, Any]) -> Tuple[Dict[str, str], Dict[str, str]]:
    fill = {"off": None, "safe": core.make_filler_regex(core.FILLERS_SAFE),
            "more": core.make_filler_regex(core.FILLERS_SAFE + core.FILLERS_MORE)}.get(st.fillers)
    post = core.TextPost(core.parse_replacements(st.replacements), fill, True, st.halfwidth)
    sents = core.make_sentences(words, post)
    paras = core.make_paragraphs(sents)
    cues = core.make_cues(words, post, core.CueRules(max_chars=int(st.cue_max_chars), max_dur=float(st.cue_max_dur),
                                                    gap=float(st.cue_gap)))
    paths: Dict[str, str] = {}
    texts: Dict[str, str] = {"plain": core.to_txt(paras, names, timestamps=False)}

    def put(fmt: str, text: str, enc: str = "utf-8") -> None:
        pth = base + _ext(fmt)
        tmp = pth + ".tmp"
        with open(tmp, "w", encoding=enc, newline="") as f:
            f.write(text)
        os.replace(tmp, pth)
        paths[fmt] = pth

    stem = os.path.basename(base)
    fm = set(st.formats)
    if "txt" in fm:
        put("txt", core.to_txt(paras, names))
    if "srt" in fm:
        put("srt", core.to_srt(cues, names, st.speaker_fmt))
    if "vtt" in fm:
        put("vtt", core.to_vtt(cues, names))
    if "json" in fm:
        put("json", core.to_json(sents, names, results, meta))
    if "csv" in fm:
        put("csv", core.to_csv(sents, names), enc="utf-8-sig")
    if "md" in fm:
        info = {"音源": os.path.basename(meta.get("source", "")), "長さ": core.fmt_dur(meta.get("duration", 0)),
                "モデル": meta.get("model"), "作成": meta.get("created")}
        if names:
            info["話者"] = "、".join(dict.fromkeys(names.values()))
        put("md", core.to_markdown(paras, names, stem, info))
    if "plain" in fm:
        put("plain", texts["plain"])
    if "rttm" in fm and turns:
        put("rttm", core.to_rttm(turns, stem, names))
    rv = base + "_review.md"
    if st.review and core.review_items(results):
        put("review", core.to_review_md(results, stem))
    elif os.path.exists(rv):
        os.remove(rv)  # 前回の古い要確認リストが残らないように
    return paths, texts


# =====================================================================
# 要確認リストを音声つきで表示
# =====================================================================


def _clip_audio_html(wav_path: str, start: float, end: float) -> str:
    """区間を小さい Opus にして <audio> に埋め込む(ノートブックが重くならないよう 24kbps)"""
    import base64
    import subprocess

    cmd = ["ffmpeg", "-loglevel", "error", "-ss", f"{max(0.0, start):.2f}", "-t", f"{max(0.3, end - start):.2f}",
           "-i", wav_path, "-c:a", "libopus", "-b:a", "24k", "-f", "ogg", "pipe:1"]
    p = subprocess.run(cmd, capture_output=True)
    if p.returncode != 0 or not p.stdout:
        return ""
    b64 = base64.b64encode(p.stdout).decode()
    return f"<audio controls preload='none' src='data:audio/ogg;base64,{b64}'></audio>"


def show_review(out: Dict[str, Any], max_items: int = 8) -> None:
    from IPython.display import HTML, display

    items = core.review_items(out.get("results") or [])
    if not items:
        return
    groups: List[List[core.ClipResult]] = []  # 2分割したものは元のクリップごとにまとめる
    for r in items:
        if r.clip.parent is not None and groups and groups[-1][0].clip.parent == r.clip.parent:
            groups[-1].append(r)
        else:
            groups.append([r])
    rows = []
    for g in groups[:max_items]:
        a0, a1 = g[0].clip.start, g[-1].clip.end
        flags = [f for r in g for f in r.flags]
        why = "、".join(dict.fromkeys(core.FLAG_JA.get(f, f) for f in flags)) if flags else "自動で差し替え済み"
        adopted = "".join(r.text for r in g)
        first = next((a.get("text") for a in g[0].attempts if "text" in a), adopted) or ""
        body = (f"<div style='line-height:1.7'>{core.diff_html(first, adopted)}</div>"
                "<div style='color:#888;font-size:11px'>赤=最初の結果から消えた / 緑=差し替えで増えた</div>"
                if first != adopted else f"<div>{html.escape(adopted[:300] or '(空)')}</div>")
        rows.append(
            f"<tr><td style='white-space:nowrap'>{core.fmt_hms(a0)}〜{core.fmt_hms(a1)}</td>"
            f"<td>{_clip_audio_html(out['wav'], a0, a1)}</td><td><b>{html.escape(why)}</b>{body}</td></tr>")
    more = (f"<div>ほか {len(groups) - max_items} 件は {html.escape(out['paths'].get('review', ''))} を見てください</div>"
            if len(groups) > max_items else "")
    display(HTML(f"<details open><summary><b>要確認 {len(groups)} 件</b>(音声を聞いて確かめられます)</summary>"
                 f"<table border=1 cellpadding=4 style='border-collapse:collapse;font-size:13px'>{''.join(rows)}</table>{more}</details>"))


# =====================================================================
# 議事録づくり(プロンプトを作る / Claude API で作る)
# =====================================================================

MINUTES_SYSTEM = """あなたは、会議の文字起こしから正確で読みやすい日本語の議事録を作るアシスタントです。
- 文字起こしは音声認識の結果なので、同音異義語の誤変換・固有名詞の誤り・言い直しやフィラーが含まれます。文脈から明らかな誤りは正しく直してかまいませんが、発言にない内容を推測で足さないでください。
- 自信がない固有名詞・数字・日付は【要確認】と書き添えてください。
- 話者ラベル(「話者A」など)は、文字起こし内で名前が明らかな場合だけ名前に置き換えてください。
- 重要な内容には、根拠になった時刻を [hh:mm:ss] の形で添えてください。"""

MINUTES_STYLES = {
    "議事録（決定事項・TODO つき）": """<transcript> の内容から議事録を Markdown で作ってください。構成:
# 会議名(わからなければ「会議」)
- 日時・参加者(文字起こしから分かる範囲)
## 要約(3〜5行)
## 議題ごとの内容(議題ごとに ### 見出し。誰が何を言ったか、根拠の時刻)
## 決定事項
## TODO(| 担当 | 期限 | 内容 | の表。分からない欄は「未定」)
## 保留・次回への持ち越し
## 要確認(聞き取りが怪しい箇所・数字・固有名詞)""",
    "要約（3行＋詳細）": """<transcript> の内容を要約してください。最初に3行の要約、続けて論点ごとの詳細(箇条書き・根拠の時刻つき)、最後に要確認の点を書いてください。""",
    "発言者ごとの要点": """<transcript> について、発言者ごとに主張・提案・懸念・引き受けたことを箇条書きでまとめてください(根拠の時刻つき)。最後に、発言者どうしで意見が分かれた点を整理してください。""",
}


def _minutes_request(text: str, style: str, glossary: Sequence[str] = ()) -> Tuple[str, str]:
    instr = MINUTES_STYLES.get(style, style)
    if glossary:
        instr += "\n\n参考: この会議で出てくる固有名詞・専門用語の候補: " + "、".join(glossary)
    return instr, f"<transcript>\n{text}\n</transcript>"


def make_minutes(transcript: str, *, style: str = "議事録（決定事項・TODO つき）", mode: str = "プロンプトだけ作る",
                 model: str = "claude-opus-5", glossary: Sequence[str] = (), out_path: str = "",
                 gemini_model: str = "google/gemini-3.5-flash") -> Optional[str]:
    """文字起こし(ファイルパス or 本文)から議事録を作る。書き出したファイルのパスを返す

    mode: "プロンプトだけ作る" / "Gemini（Colab AI・無料）" / "Claude API"
    """
    src_path = transcript if os.path.isfile(transcript) else ""
    text = open(src_path, encoding="utf-8").read() if src_path else transcript
    if not text.strip():
        raise ValueError("文字起こしが空です")
    base = os.path.splitext(src_path)[0] if src_path else os.path.join(runtime.BASE_DIR, "minutes")
    instr, tr = _minutes_request(text, style, glossary)

    if mode.startswith("プロンプト"):
        pth = out_path or base + "_minutes_prompt.md"
        with open(pth, "w", encoding="utf-8") as f:
            f.write(MINUTES_SYSTEM + "\n\n" + instr + "\n\n" + tr + "\n")
        log(f"📝 プロンプトを書き出しました: {pth}\n   中身をまるごと Claude / ChatGPT に貼り付けてください(約 {len(text):,} 文字)")
        return pth

    if mode.startswith("Gemini"):
        # Colab に組み込みの google.colab.ai(API キー不要。2026年6月から全ユーザー無料)
        try:
            from google.colab import ai
        except Exception as e:
            raise RuntimeError(f"google.colab.ai が使えません(Colab の画面から実行してください): {e}")
        log(f"🤖 {gemini_model}(Colab AI)で作成中…")
        chunks: List[str] = []
        for piece in ai.generate_text(MINUTES_SYSTEM + "\n\n" + instr + "\n\n" + tr, model_name=gemini_model, stream=True):
            if piece:
                print(piece, end="", flush=True)
                chunks.append(piece)
        print()
        pth = out_path or base + "_minutes.md"
        with open(pth, "w", encoding="utf-8") as f:
            f.write("".join(chunks))
        log(f"✅ 議事録を書き出しました: {pth}")
        return pth

    # ---- Claude API ----
    try:
        import anthropic
    except ImportError:
        import subprocess
        import sys

        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "anthropic"], check=True)
        import anthropic
    key = runtime.get_secret("ANTHROPIC_API_KEY")
    if not key and not os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        raise RuntimeError("Colab のシークレットに ANTHROPIC_API_KEY を登録して、このノートブックのアクセスを ON にしてください")
    client = anthropic.Anthropic(api_key=key) if key else anthropic.Anthropic()
    req: Dict[str, Any] = dict(
        model=model, max_tokens=64000, system=MINUTES_SYSTEM,
        messages=[{"role": "user", "content": [
            # 文字起こしを先頭に置いてキャッシュ(同じ文字起こしで別のスタイルを試すとき安くなる)
            {"type": "text", "text": tr, "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": instr},
        ]}],
    )
    # 安全フィルタで断られたとき、サーバー側で別モデルに引き継ぐ(Opus 5 系・Fable 系で有効)
    use_fallback = model.startswith(("claude-opus-5", "claude-fable-5"))
    log(f"🤖 {model} で作成中…(長い会議だと数分かかります)")
    try:
        if use_fallback:
            cm = client.beta.messages.stream(**req, betas=["server-side-fallback-2026-07-01"], fallbacks="default")
        else:
            cm = client.messages.stream(**req)
        with cm as stream:
            for chunk in stream.text_stream:
                print(chunk, end="", flush=True)
            msg = stream.get_final_message()
    except anthropic.AuthenticationError:
        raise RuntimeError("API キーが正しくないようです(ANTHROPIC_API_KEY を確認してください)")
    except anthropic.RateLimitError:
        raise RuntimeError("レート制限にかかりました。少し待ってからもう一度実行してください")
    except anthropic.APIStatusError as e:
        raise RuntimeError(f"Claude API のエラー ({e.status_code}): {e.message}")
    except anthropic.APIConnectionError:
        raise RuntimeError("Claude API に接続できませんでした(ネットワークを確認してください)")
    print()
    if msg.stop_reason == "refusal":
        log("⚠️ モデルが応答を断りました。内容を確認して、別のモデルかプロンプトだけ作るモードを試してください")
        return None
    out = "".join(b.text for b in msg.content if b.type == "text")
    if msg.stop_reason == "max_tokens":
        log("⚠️ 出力が上限で途中まで切れています")
    pth = out_path or base + "_minutes.md"
    with open(pth, "w", encoding="utf-8") as f:
        f.write(out)
    u = msg.usage
    log(f"✅ 議事録を書き出しました: {pth}\n   トークン: 入力 {u.input_tokens:,} / キャッシュ読 {getattr(u, 'cache_read_input_tokens', 0) or 0:,}"
        f" / キャッシュ書 {getattr(u, 'cache_creation_input_tokens', 0) or 0:,} / 出力 {u.output_tokens:,}")
    return pth
