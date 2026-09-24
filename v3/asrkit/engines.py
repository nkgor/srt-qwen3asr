# -*- coding: utf-8 -*-
"""asrkit.engines — ワーカー(各 venv)の中で動く、モデルごとのアダプタ

どのエンジンも transcribe(items, language) -> [{"text", "language", "words"?}] をそろえる。
items は [(float32 16kHz の波形, context 文字列), ...]。words はクリップ先頭基準の [(語, 開始, 終了)]。
重い import (torch など) はクラスの中でだけ行う。
"""
from __future__ import annotations

import gc
import glob
import os
import re
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from . import core

# Qwen3-ASR/アライナーの言語名 ⇔ ISO 639-1
LANG_CODES = {
    "Japanese": "ja", "English": "en", "Chinese": "zh", "Cantonese": "yue", "Korean": "ko", "French": "fr",
    "German": "de", "Spanish": "es", "Portuguese": "pt", "Italian": "it", "Russian": "ru", "Arabic": "ar",
    "Indonesian": "id", "Thai": "th", "Vietnamese": "vi", "Turkish": "tr", "Hindi": "hi", "Malay": "ms",
    "Dutch": "nl", "Swedish": "sv", "Danish": "da", "Finnish": "fi", "Polish": "pl", "Czech": "cs",
    "Filipino": "fil", "Persian": "fa", "Greek": "el", "Romanian": "ro", "Hungarian": "hu", "Macedonian": "mk",
}
CODE_LANGS = {v: k for k, v in LANG_CODES.items()}
ALIGNER_LANGS = {"Chinese", "English", "Cantonese", "French", "German", "Italian", "Japanese", "Korean",
                 "Portuguese", "Russian", "Spanish"}


def lang_code(language: Optional[str]) -> Optional[str]:
    if not language:
        return None
    return LANG_CODES.get(language, language if len(language) <= 3 else None)


def lang_name(code_or_name: Optional[str]) -> str:
    if not code_or_name:
        return ""
    s = str(code_or_name).split(",")[0].strip()
    if s in LANG_CODES:
        return s
    return CODE_LANGS.get(s.lower(), s[:1].upper() + s[1:].lower())


def _pad_min(a: np.ndarray, min_sec: float = 0.5) -> np.ndarray:
    n = int(min_sec * core.SR)
    a = np.asarray(a, np.float32)
    if a.shape[0] < n:
        a = np.pad(a, (0, n - a.shape[0]))
    return a


# ---------------------------------------------------------------- torch まわり


def torch_mod():
    import torch

    return torch


def pick_dtype(dtype: str = "auto"):
    torch = torch_mod()
    if dtype in (None, "", "auto"):
        if torch.cuda.is_available():
            return torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
        return torch.float32
    return {"bf16": torch.bfloat16, "bfloat16": torch.bfloat16, "fp16": torch.float16,
            "float16": torch.float16, "fp32": torch.float32, "float32": torch.float32}[dtype]


def pick_attn(attn: str = "auto") -> str:
    if attn and attn != "auto":
        return attn
    torch = torch_mod()
    if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8:
        try:
            import flash_attn  # noqa: F401

            return "flash_attention_2"
        except Exception:
            pass
    return "sdpa"


def device_str() -> str:
    torch = torch_mod()
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def free_cuda() -> None:
    gc.collect()
    try:
        torch = torch_mod()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def is_oom(e: BaseException) -> bool:
    msg = str(e).lower()
    return "out of memory" in msg or "cuda error: out of memory" in msg or type(e).__name__ == "OutOfMemoryError"


def reset_gpu_peak() -> None:
    """VRAM 峰の計測をリセット(同じワーカーでモデルを入れ替えても、前のモデルの峰を引きずらないように)

    torch をまだ読んでいないワーカーでは何もしない(faster-whisper などの CUDA ライブラリの読み込み順を変えないため)
    """
    torch = sys.modules.get("torch")
    if torch is None:
        return
    try:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def gpu_peak_gb() -> float:
    try:
        torch = torch_mod()
        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / 1e9, 2)
    except Exception:
        pass
    return 0.0


def with_oom_backoff(fn, items: List[Any], bs: int, set_bs=None, min_bs: int = 1, log=None) -> List[Any]:
    """CUDA OOM が出たらバッチを半分にしてやり直す(T4 などで便利)"""
    out: List[Any] = []
    i = 0
    while i < len(items):
        chunk = items[i: i + bs]
        try:
            if set_bs:
                set_bs(bs)
            out.extend(fn(chunk))
            i += len(chunk)
        except Exception as e:
            if not is_oom(e) or bs <= min_bs:
                raise
            free_cuda()
            bs = max(min_bs, bs // 2)
            if log:
                log(f"CUDA OOM → バッチを {bs} に下げて再試行")
    return out


def _log(msg: str) -> None:
    print(f"[engine] {msg}", file=sys.stderr, flush=True)


# =====================================================================
# エンジン本体
# =====================================================================


class Engine:
    kind = "base"
    punctuates = True
    native_timestamps = False
    uses_context = True

    def __init__(self, **opt: Any) -> None:
        self.opt = opt
        self.batch_size = int(opt.get("batch_size") or 8)

    def info(self) -> Dict[str, Any]:
        return {"summary": f"{self.kind}:{self.opt.get('model')}", "punctuates": self.punctuates,
                "native_timestamps": self.native_timestamps, "uses_context": self.uses_context,
                "gpu_peak_gb": gpu_peak_gb()}

    def transcribe(self, items: List[Tuple[np.ndarray, str]], language: Optional[str]) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def close(self) -> None:
        free_cuda()


class QwenASREngine(Engine):
    """Qwen3-ASR (qwen-asr パッケージ / transformers バックエンド)"""

    kind = "qwen"

    def __init__(self, model: str = "Qwen/Qwen3-ASR-1.7B", dtype: str = "auto", attn: str = "auto",
                 batch_size: int = 8, max_new_tokens: int = 1024, **opt: Any) -> None:
        super().__init__(model=model, batch_size=batch_size, **opt)
        from qwen_asr import Qwen3ASRModel

        self.attn = pick_attn(attn)
        self.dtype = pick_dtype(dtype)
        kw = dict(dtype=self.dtype, device_map=device_str(), max_inference_batch_size=self.batch_size,
                  max_new_tokens=int(max_new_tokens))
        if self.attn:
            kw["attn_implementation"] = self.attn
        try:
            self.m = Qwen3ASRModel.from_pretrained(model, **kw)
        except Exception as e:
            if self.attn == "flash_attention_2":
                _log(f"flash_attention_2 で失敗 → sdpa で再試行: {e}")
                kw["attn_implementation"] = self.attn = "sdpa"
                self.m = Qwen3ASRModel.from_pretrained(model, **kw)
            else:
                raise

    def info(self) -> Dict[str, Any]:
        d = super().info()
        d["summary"] = f"{self.opt.get('model')} attn={self.attn} dtype={str(self.dtype).replace('torch.', '')}"
        return d

    def _run(self, chunk: List[Tuple[np.ndarray, str]], language: Optional[str]) -> List[Dict[str, Any]]:
        auds = [(_pad_min(a), core.SR) for a, _ in chunk]
        ctxs = [c or "" for _, c in chunk]
        langs = [language] * len(chunk) if language else None
        res = self.m.transcribe(audio=auds, context=ctxs, language=langs, return_time_stamps=False)
        return [{"text": r.text or "", "language": r.language or (language or "")} for r in res]

    def transcribe(self, items, language):
        def set_bs(b):
            self.m.max_inference_batch_size = b

        return with_oom_backoff(lambda ch: self._run(ch, language), items, self.batch_size, set_bs, log=_log)

    def close(self) -> None:
        del self.m
        super().close()


class QwenAligner:
    """Qwen3-ForcedAligner-0.6B。どのASRの結果にも単語タイムスタンプを付けられる(11言語)"""

    kind = "aligner"

    def __init__(self, model: str = "Qwen/Qwen3-ForcedAligner-0.6B", dtype: str = "auto", attn: str = "auto",
                 batch_size: int = 8, **opt: Any) -> None:
        from qwen_asr import Qwen3ForcedAligner

        self.opt = dict(model=model, **opt)
        self.batch_size = int(batch_size or 8)
        self.attn = pick_attn(attn)
        kw = dict(dtype=pick_dtype(dtype), device_map=device_str())
        if self.attn:
            kw["attn_implementation"] = self.attn
        try:
            self.m = Qwen3ForcedAligner.from_pretrained(model, **kw)
        except Exception as e:
            if self.attn == "flash_attention_2":
                _log(f"aligner: flash_attention_2 で失敗 → sdpa: {e}")
                kw["attn_implementation"] = self.attn = "sdpa"
                self.m = Qwen3ForcedAligner.from_pretrained(model, **kw)
            else:
                raise

    def info(self) -> Dict[str, Any]:
        return {"summary": f"{self.opt.get('model')} attn={self.attn}", "gpu_peak_gb": gpu_peak_gb()}

    def align(self, items: List[Tuple[np.ndarray, str, str]]) -> List[List[Tuple[str, float, float]]]:
        """items: [(波形, テキスト, 言語名)] → [[(語, 開始, 終了)], ...](クリップ先頭基準)"""

        def run(chunk):
            res = self.m.align(audio=[(_pad_min(a), core.SR) for a, _, _ in chunk],
                               text=[t for _, t, _ in chunk], language=[l for _, _, l in chunk])
            return [[(it.text, float(it.start_time), float(it.end_time)) for it in r.items] for r in res]

        return with_oom_backoff(run, items, self.batch_size, log=_log)

    def close(self) -> None:
        del self.m
        free_cuda()


def _preload_nvidia_libs() -> None:
    """pip の nvidia-*-cu12 に入っている cuBLAS/cuDNN を先に読み込む(CTranslate2 用)"""
    import ctypes

    for sp in sys.path:
        for pat in ("nvidia/cublas/lib/libcublas*.so*", "nvidia/cudnn/lib/libcudnn*.so*", "nvidia/cuda_runtime/lib/libcudart*.so*"):
            for f in sorted(glob.glob(os.path.join(sp, pat))):
                try:
                    ctypes.CDLL(f, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass


class FasterWhisperEngine(Engine):
    """Whisper 系 (faster-whisper / CTranslate2)。kotoba-whisper の -faster 版もこれ"""

    kind = "faster-whisper"
    native_timestamps = True

    def __init__(self, model: str = "large-v3-turbo", compute_type: str = "auto", beam_size: int = 5,
                 batch_size: int = 8, word_timestamps: Any = "auto", chunk_length: Optional[int] = None,
                 **opt: Any) -> None:
        super().__init__(model=model, batch_size=batch_size, **opt)
        # kotoba-whisper は公式の使い方が chunk_length=15(15 秒の窓で読む)
        self.chunk_length = int(chunk_length) if chunk_length else None
        # 蒸留モデル(kotoba-whisper / distil-whisper。デコーダが 2 層)は、変換時に入った large-v3 用の
        # alignment_heads が存在しない層を指していて、単語タイムスタンプ(find_alignment)で segfault する。
        # その場合は単語時刻を出さず、Qwen3-ForcedAligner でタイムスタンプを付ける
        if word_timestamps == "auto":
            word_timestamps = not any(k in model.lower() for k in ("kotoba", "distil"))
        self.word_ts = bool(word_timestamps)
        self.native_timestamps = self.word_ts
        _preload_nvidia_libs()
        from faster_whisper import WhisperModel

        torch = None
        try:
            torch = torch_mod()
        except Exception:
            pass
        cuda = bool(torch and torch.cuda.is_available())
        if compute_type == "auto":
            compute_type = "float16" if cuda else "int8"
        self.beam = int(beam_size)
        self.m = WhisperModel(model, device="cuda" if cuda else "cpu", compute_type=compute_type)
        self.ct = compute_type

    def transcribe(self, items, language):
        out = []
        code = lang_code(language)
        for a, ctx in items:
            segs, info = self.m.transcribe(
                np.asarray(a, np.float32), language=code, beam_size=self.beam, initial_prompt=(ctx or None),
                word_timestamps=self.word_ts, vad_filter=False, condition_on_previous_text=False,
                temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0), compression_ratio_threshold=2.4,
                log_prob_threshold=-1.0, no_speech_threshold=0.6,
                **({"chunk_length": self.chunk_length} if self.chunk_length else {}),
            )
            segs = list(segs)
            text = "".join(s.text for s in segs).strip()
            words = [[w.word, float(w.start), float(w.end)] for s in segs for w in (s.words or [])]
            out.append({"text": text, "language": lang_name(getattr(info, "language", "") or code or ""), "words": words})
        return out

    def close(self) -> None:
        del self.m
        super().close()


class NemoEngine(Engine):
    """NVIDIA NeMo の ASR (例: nvidia/parakeet-tdt_ctc-0.6b-ja)。文字単位のタイムスタンプあり"""

    kind = "nemo"
    native_timestamps = True
    uses_context = False

    def __init__(self, model: str = "nvidia/parakeet-tdt_ctc-0.6b-ja", batch_size: int = 16, **opt: Any) -> None:
        super().__init__(model=model, batch_size=batch_size, **opt)
        import nemo.collections.asr as nemo_asr

        self.m = nemo_asr.models.ASRModel.from_pretrained(model_name=model)
        torch = torch_mod()
        if torch.cuda.is_available():
            self.m = self.m.cuda()
        self.m.eval()
        self.tmp = tempfile.mkdtemp(prefix="nemo_clips_")

    def _run(self, chunk):
        paths = []
        for i, (a, _) in enumerate(chunk):
            p = os.path.join(self.tmp, f"{i}.wav")
            core.write_wav16(p, _pad_min(a))
            paths.append(p)
        try:
            hyps = self.m.transcribe(paths, batch_size=len(paths), timestamps=True, verbose=False)
        except TypeError:
            hyps = self.m.transcribe(paths, batch_size=len(paths), return_hypotheses=True)
        if isinstance(hyps, tuple):  # 古い RNNT 系は (best, all)
            hyps = hyps[0]
        out = []
        for h in hyps:
            text = getattr(h, "text", h if isinstance(h, str) else "") or ""
            # 辞書に無い音は「⁇」(SentencePiece の unk)で出てくるので消す
            text = core.fix_ja_spaces(re.sub(r"\s*⁇\s*", " ", text)).strip() if "⁇" in text else text
            ts = getattr(h, "timestamp", None) or {}
            words = None
            unit = ts.get("word") if isinstance(ts, dict) else None
            if unit and " " in text.strip() and len(unit) > 1:  # 空白で区切る言語は単語単位
                words = [[u.get("word", ""), float(u.get("start", 0)), float(u.get("end", 0))] for u in unit]
            elif isinstance(ts, dict) and ts.get("char"):  # 日本語などは文字単位
                words = [[u.get("char", ""), float(u.get("start", 0)), float(u.get("end", 0))] for u in ts["char"]
                         if "⁇" not in str(u.get("char", ""))]
            out.append({"text": text, "language": "", "words": words})
        return out

    def transcribe(self, items, language):
        return with_oom_backoff(self._run, items, self.batch_size, log=_log)

    def close(self) -> None:
        del self.m
        super().close()


class HFPipelineEngine(Engine):
    """transformers の automatic-speech-recognition パイプライン(Whisper 系・kotoba-whisper など)"""

    kind = "hf-pipeline"

    def __init__(self, model: str, dtype: str = "auto", batch_size: int = 8, trust_remote_code: bool = False,
                 whisper: bool = True, **opt: Any) -> None:
        super().__init__(model=model, batch_size=batch_size, **opt)
        from transformers import pipeline

        torch = torch_mod()
        self.whisper = whisper
        kw: Dict[str, Any] = dict(model=model, device=device_str(), trust_remote_code=trust_remote_code)
        try:
            self.p = pipeline("automatic-speech-recognition", dtype=pick_dtype(dtype), **kw)
        except TypeError:
            self.p = pipeline("automatic-speech-recognition", torch_dtype=pick_dtype(dtype), **kw)
        self.torch = torch

    def _gen_kwargs(self, language, ctx):
        gk: Dict[str, Any] = {}
        if self.whisper:
            if language:
                gk["language"] = lang_code(language) or language
            gk["task"] = "transcribe"
            if ctx:
                try:
                    ids = self.p.tokenizer.get_prompt_ids(ctx, return_tensors="pt").to(self.p.device)
                    gk["prompt_ids"] = ids
                except Exception:
                    pass
        return gk

    def transcribe(self, items, language):
        # context ごとにまとめて流す(Whisper の prompt はバッチ内で共通)
        out: List[Optional[Dict[str, Any]]] = [None] * len(items)
        groups: Dict[str, List[int]] = {}
        for i, (_, c) in enumerate(items):
            groups.setdefault(c or "", []).append(i)
        for ctx, idxs in groups.items():
            auds = [{"raw": _pad_min(items[i][0]), "sampling_rate": core.SR} for i in idxs]

            def run(chunk):
                res = self.p(chunk, batch_size=len(chunk), generate_kwargs=self._gen_kwargs(language, ctx))
                return [r.get("text", "") if isinstance(r, dict) else str(r) for r in res]

            texts = with_oom_backoff(run, auds, self.batch_size, log=_log)
            for i, t in zip(idxs, texts):
                out[i] = {"text": (t or "").strip(), "language": language or ""}
        return [o or {"text": ""} for o in out]

    def close(self) -> None:
        del self.p
        super().close()


class CohereASREngine(Engine):
    """Cohere Transcribe (CohereLabs/cohere-transcribe-03-2026, 2B, 14言語・日本語あり)。
    transformers>=5.4 のネイティブ実装を使う(qwen-asr とは transformers のバージョンがぶつかるので hf 環境で)。
    タイムスタンプも context も無いので、時刻は Qwen3-ForcedAligner で付ける。"""

    kind = "cohere"
    uses_context = False

    def __init__(self, model: str = "CohereLabs/cohere-transcribe-03-2026", dtype: str = "auto", batch_size: int = 16,
                 max_new_tokens: int = 448, punctuation: bool = True, revision: str = "", **opt: Any) -> None:
        super().__init__(model=model, batch_size=batch_size, **opt)
        import transformers
        from transformers import AutoProcessor

        self.max_new_tokens = int(max_new_tokens)
        self.punctuation = bool(punctuation)
        dt = pick_dtype(dtype)
        cls = getattr(transformers, "CohereAsrForConditionalGeneration", None)
        last: Optional[Exception] = None
        self.m = None
        # 公開直後は HF 版の重みが PR ブランチ(refs/pr/6)にあったので、だめなら順に試す
        for rev in ([revision] if revision else []) + [None, "refs/pr/6"]:
            try:
                kw: Dict[str, Any] = {"revision": rev} if rev else {}
                self.proc = AutoProcessor.from_pretrained(model, **kw)
                if cls is not None:
                    self.m = cls.from_pretrained(model, dtype=dt, device_map=device_str(), **kw)
                else:
                    from transformers import AutoModelForSpeechSeq2Seq

                    self.m = AutoModelForSpeechSeq2Seq.from_pretrained(model, dtype=dt, device_map=device_str(),
                                                                      trust_remote_code=True, **kw)
                break
            except Exception as e:  # 次の候補へ
                last = e
                _log(f"cohere: revision={rev} で失敗: {type(e).__name__}: {str(e)[:300]}")
        if self.m is None:
            raise RuntimeError(f"Cohere Transcribe を読み込めませんでした(HF で規約に同意し HF_TOKEN を登録しましたか?): {last}")
        self.m.eval()

    def _run(self, chunk, code):
        torch = torch_mod()
        auds = [_pad_min(a) for a, _ in chunk]
        inputs = self.proc(auds, sampling_rate=core.SR, return_tensors="pt", language=code, punctuation=self.punctuation)
        idx = inputs.get("audio_chunk_index")
        inputs = inputs.to(self.m.device, dtype=self.m.dtype)
        with torch.inference_mode():
            out = self.m.generate(**inputs, max_new_tokens=self.max_new_tokens)
        try:
            texts = self.proc.decode(out, skip_special_tokens=True, audio_chunk_index=idx, language=code)
        except TypeError:
            texts = self.proc.batch_decode(out, skip_special_tokens=True)
        if isinstance(texts, str):
            texts = [texts]
        return [{"text": (t or "").strip(), "language": lang_name(code)} for t in texts]

    def transcribe(self, items, language):
        code = lang_code(language) or "ja"
        return with_oom_backoff(lambda ch: self._run(ch, code), items, self.batch_size, log=_log)

    def close(self) -> None:
        del self.m
        super().close()


class GraniteSpeechEngine(Engine):
    """IBM Granite Speech 4.1 2B(英・仏・独・西・葡・日)。キーワード(固有名詞)を渡せる。
    プロンプトは英語で書く決まり(モデルカードより)。transformers の hf 環境で動かす"""

    kind = "granite"

    def __init__(self, model: str = "ibm-granite/granite-speech-4.1-2b", dtype: str = "auto", batch_size: int = 8,
                 max_new_tokens: int = 448, **opt: Any) -> None:
        super().__init__(model=model, batch_size=batch_size, **opt)
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        self.proc = AutoProcessor.from_pretrained(model)
        self.tok = self.proc.tokenizer
        self.m = AutoModelForSpeechSeq2Seq.from_pretrained(model, device_map=device_str(), dtype=pick_dtype(dtype))
        self.m.eval()
        self.max_new_tokens = int(max_new_tokens)
        self.ctx_terms: List[str] = []

    def _prompt(self, use_kw: bool) -> str:
        if use_kw and self.ctx_terms:
            q = "transcribe the speech to text. Keywords: " + ", ".join(self.ctx_terms)
        else:
            q = "transcribe the speech with proper punctuation and capitalization."
        return self.tok.apply_chat_template([{"role": "user", "content": "<|audio|>" + q}], tokenize=False,
                                            add_generation_prompt=True)

    def _run(self, chunk: List[Tuple[np.ndarray, str]]) -> List[Dict[str, Any]]:
        torch = torch_mod()
        prompt = self._prompt(bool(chunk[0][1]))  # chunk の中はプロンプトが同じ
        wavs = [_pad_min(a) for a, _ in chunk]  # 長さが違っても processor が詰めてくれる(トークナイザは左詰め)
        inp = self.proc([prompt] * len(wavs), wavs, device=device_str(), return_tensors="pt").to(device_str())
        with torch.inference_mode():
            ids = self.m.generate(**inp, max_new_tokens=self.max_new_tokens, do_sample=False, num_beams=1)
        n = inp["input_ids"].shape[-1]
        texts = self.tok.batch_decode(ids[:, n:], add_special_tokens=False, skip_special_tokens=True)
        return [{"text": (t or "").strip()} for t in texts]

    def transcribe(self, items, language):
        # キーワードあり/なしでプロンプトが違うので、それぞれまとめてバッチ処理(以前は1件ずつで遅かった)
        out: List[Dict[str, Any]] = [{} for _ in items]
        for use_kw in (True, False):
            idx = [i for i, (_, c) in enumerate(items) if bool(c) == use_kw]
            if not idx:
                continue
            res = with_oom_backoff(self._run, [items[i] for i in idx], self.batch_size, log=_log)
            for i, r in zip(idx, res):
                out[i] = {"text": r["text"], "language": language or ""}
        return out

    def close(self) -> None:
        del self.m
        super().close()


class VibeVoiceEngine(Engine):
    """Microsoft VibeVoice-ASR(8B、50以上の言語、context を渡せる)。transformers ネイティブ版(-HF)を使う。
    本来は60分を一気に読んで話者も付けられるモデルだが、ここでは他のモデルと同じくクリップ単位で本文だけ使う"""

    kind = "vibevoice"

    def __init__(self, model: str = "microsoft/VibeVoice-ASR-HF", dtype: str = "auto", batch_size: int = 4,
                 max_new_tokens: int = 2048, **opt: Any) -> None:
        super().__init__(model=model, batch_size=batch_size, **opt)
        import transformers
        from transformers import AutoProcessor

        cls = getattr(transformers, "VibeVoiceAsrForConditionalGeneration")
        self.proc = AutoProcessor.from_pretrained(model)
        self.m = cls.from_pretrained(model, device_map=device_str(), dtype=pick_dtype(dtype))
        self.m.eval()
        self.max_new_tokens = int(max_new_tokens)
        self.tmp = tempfile.mkdtemp(prefix="vibevoice_")

    def _run(self, chunk):
        torch = torch_mod()
        paths = []
        for i, (a, _) in enumerate(chunk):
            p = os.path.join(self.tmp, f"{i}.wav")
            core.write_wav16(p, _pad_min(a))
            paths.append(p)
        prompts = [c or None for _, c in chunk]
        inputs = self.proc.apply_transcription_request(paths, prompt=prompts).to(self.m.device, self.m.dtype)
        with torch.inference_mode():
            out = self.m.generate(**inputs, max_new_tokens=self.max_new_tokens)
        gen = out[:, inputs["input_ids"].shape[1]:]
        texts = self.proc.decode(gen, return_format="transcription_only")
        if isinstance(texts, str):
            texts = [texts]
        return [{"text": (t or "").strip(), "language": ""} for t in texts]

    def transcribe(self, items, language):
        return with_oom_backoff(self._run, items, self.batch_size, log=_log)

    def close(self) -> None:
        del self.m
        super().close()


class PyannoteDiarizer:
    """pyannote.audio 4.x の話者分離(community-1 は exclusive 出力もあり)"""

    kind = "pyannote"

    def __init__(self, model: str = "pyannote/speaker-diarization-community-1", **opt: Any) -> None:
        from pyannote.audio import Pipeline

        torch = torch_mod()
        token = os.environ.get("HF_TOKEN") or None
        try:
            p = Pipeline.from_pretrained(model, token=token)
        except TypeError:
            p = Pipeline.from_pretrained(model, use_auth_token=token)
        if p is None:
            raise RuntimeError(
                f"{model} を読み込めませんでした。Hugging Face でモデルの利用規約に同意し、"
                "Colab のシークレットに HF_TOKEN を登録してください。")
        if torch.cuda.is_available():
            p.to(torch.device("cuda"))
        self.p = p
        self.opt = dict(model=model, **opt)

    def info(self) -> Dict[str, Any]:
        return {"summary": self.opt.get("model"), "gpu_peak_gb": gpu_peak_gb()}

    def diarize(self, wav_path: str, num_speakers: int = 0, min_speakers: int = 0, max_speakers: int = 0) -> Dict[str, Any]:
        torch = torch_mod()
        w = core.Wav16(wav_path)
        x = torch.from_numpy(w.get(0, w.duration)).unsqueeze(0)
        kw: Dict[str, Any] = {}
        if num_speakers and num_speakers > 0:
            kw["num_speakers"] = int(num_speakers)
        else:
            if min_speakers and min_speakers > 0:
                kw["min_speakers"] = int(min_speakers)
            if max_speakers and max_speakers > 0:
                kw["max_speakers"] = int(max_speakers)
        out = self.p({"waveform": x, "sample_rate": w.sr}, **kw)
        ann = getattr(out, "speaker_diarization", out)
        excl = getattr(out, "exclusive_speaker_diarization", None)

        def tracks(a):
            return [[round(float(t.start), 3), round(float(t.end), 3), str(k)] for t, _, k in a.itertracks(yield_label=True)]

        return {"turns": tracks(ann), "exclusive": tracks(excl) if excl is not None else None}

    def close(self) -> None:
        del self.p
        free_cuda()


class HFSpeechLMEngine(Engine):
    """transformers の「音声→テキスト」系 LLM (AutoProcessor + AutoModelForSpeechSeq2Seq/ImageTextToText 等)
    の汎用アダプタ。processor が apply_transcription_request を持つモデル(Voxtral など)と、
    chat template で音声を渡すモデルの両方をなるべく吸収する(実験的)。"""

    kind = "hf-speechlm"

    def __init__(self, model: str, dtype: str = "auto", batch_size: int = 4, max_new_tokens: int = 1024,
                 trust_remote_code: bool = True, prompt: str = "", **opt: Any) -> None:
        super().__init__(model=model, batch_size=batch_size, **opt)
        import transformers
        from transformers import AutoProcessor

        torch = torch_mod()
        self.torch = torch
        self.dtype = pick_dtype(dtype)
        self.max_new_tokens = int(max_new_tokens)
        self.prompt = prompt
        self.proc = AutoProcessor.from_pretrained(model, trust_remote_code=trust_remote_code)
        last: Optional[Exception] = None
        self.m = None
        for cls_name in ("AutoModelForSpeechSeq2Seq", "AutoModelForImageTextToText", "AutoModelForCausalLM", "AutoModel"):
            cls = getattr(transformers, cls_name, None)
            if cls is None:
                continue
            try:
                self.m = cls.from_pretrained(model, dtype=self.dtype, device_map=device_str(), trust_remote_code=trust_remote_code)
                break
            except Exception as e:  # 次の Auto クラスを試す
                last = e
        if self.m is None:
            raise RuntimeError(f"{model} を読み込めませんでした: {last}")
        self.m.eval()

    def _inputs(self, audio: np.ndarray, ctx: str, language: Optional[str]):
        p = self.proc
        code = lang_code(language)
        if hasattr(p, "apply_transcription_request"):
            kw: Dict[str, Any] = {"audio": audio, "model_id": self.opt.get("model")}
            if code:
                kw["language"] = code
            try:
                return p.apply_transcription_request(**kw, sampling_rate=core.SR, format=["wav"])
            except TypeError:
                return p.apply_transcription_request(**kw)
        text = self.prompt or ("以下の音声を日本語で正確に書き起こしてください。" if code == "ja" else "Transcribe the audio.")
        if ctx:
            text = f"{ctx}\n{text}"
        msgs = [{"role": "user", "content": [{"type": "audio", "audio": audio}, {"type": "text", "text": text}]}]
        try:
            return p.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                         return_tensors="pt", sampling_rate=core.SR)
        except Exception:
            return p(audio=audio, text=text, sampling_rate=core.SR, return_tensors="pt")

    def transcribe(self, items, language):
        torch = self.torch
        out = []
        for a, ctx in items:  # モデルごとに入力形式が違うので 1 件ずつ(確実さ優先)
            inp = self._inputs(_pad_min(a), ctx, language)
            inp = inp.to(self.m.device)
            for k, v in list(inp.items()):
                if hasattr(v, "is_floating_point") and v.is_floating_point():
                    inp[k] = v.to(self.dtype)
            with torch.inference_mode():
                ids = self.m.generate(**inp, max_new_tokens=self.max_new_tokens, do_sample=False)
            n_in = inp["input_ids"].shape[1] if "input_ids" in inp else 0
            gen = ids[:, n_in:] if n_in and ids.shape[1] > n_in else ids
            dec = getattr(self.proc, "batch_decode", None) or self.proc.tokenizer.batch_decode
            text = dec(gen, skip_special_tokens=True)[0]
            out.append({"text": (text or "").strip(), "language": language or ""})
        return out

    def close(self) -> None:
        del self.m
        super().close()


class DummyEngine(Engine):
    """テスト用(GPU なしで配管だけ確かめる)。長さに応じた決まった文章を返す"""

    kind = "dummy"

    def transcribe(self, items, language):
        out = []
        for a, ctx in items:
            sec = len(a) / core.SR
            if ctx and "ECHO" in ctx:
                out.append({"text": ctx, "language": language or "Japanese"})
            else:
                n = max(1, int(sec // 3))
                out.append({"text": "".join(f"これは{i + 1}番目の文です。" for i in range(n)), "language": language or "Japanese"})
        return out


class DummyAligner:
    kind = "dummy-aligner"

    def __init__(self, **opt: Any) -> None:
        self.opt = opt

    def info(self) -> Dict[str, Any]:
        return {"summary": "dummy aligner"}

    def align(self, items):
        out = []
        for a, text, _ in items:
            dur = len(a) / core.SR
            chars = [ch for ch in text if core.is_kept(ch)]
            n = max(1, len(chars))
            out.append([(ch, dur * i / n, dur * (i + 1) / n) for i, ch in enumerate(chars)])
        return out

    def close(self) -> None:
        pass


class DummyDiarizer:
    kind = "dummy-diar"

    def __init__(self, **opt: Any) -> None:
        self.opt = opt

    def info(self) -> Dict[str, Any]:
        return {"summary": "dummy diarizer"}

    def diarize(self, wav_path, num_speakers=0, min_speakers=0, max_speakers=0):
        w = core.Wav16(wav_path)
        turns, t, k = [], 0.0, 0
        while t < w.duration:
            turns.append([round(t, 3), round(min(w.duration, t + 10.0), 3), f"SPEAKER_{k % 2:02d}"])
            t += 10.0
            k += 1
        return {"turns": turns, "exclusive": turns}

    def close(self) -> None:
        pass


ENGINE_CLASSES = {
    "dummy": DummyEngine,
    "dummy-aligner": DummyAligner,
    "dummy-diar": DummyDiarizer,
    "qwen": QwenASREngine,
    "faster-whisper": FasterWhisperEngine,
    "nemo": NemoEngine,
    "hf-pipeline": HFPipelineEngine,
    "hf-speechlm": HFSpeechLMEngine,
    "cohere": CohereASREngine,
    "granite": GraniteSpeechEngine,
    "vibevoice": VibeVoiceEngine,
    "aligner": QwenAligner,
    "pyannote": PyannoteDiarizer,
}


def make_engine(kind: str, options: Dict[str, Any]):
    cls = ENGINE_CLASSES.get(kind)
    if cls is None:
        raise ValueError(f"未知のエンジン: {kind}")
    return cls(**options)
