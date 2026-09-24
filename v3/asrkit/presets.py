# -*- coding: utf-8 -*-
"""asrkit.presets — モデルのプリセットと、エンジンごとの venv の中身

新しいモデルを足したいときは PRESETS に1行足すだけ。フォームのドロップダウンにも自動で出る。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .runtime import EnvSpec

# ---------------------------------------------------------------------------------------------
# venv の中身
#   qwen      : Qwen3-ASR / Qwen3-ForcedAligner(タイムスタンプ付け)。システムの torch を使う
#   fw        : faster-whisper (Whisper large-v3 / turbo / kotoba-whisper-faster)
#   nemo      : NVIDIA NeMo (parakeet-tdt_ctc-0.6b-ja など)
#   hf        : 最新 transformers(新しめの音声LLM・kotoba-whisper の HF 版など)
#   pyannote  : 話者分離
#   vllm      : vLLM サーバー(A100/H100 で爆速。torch ごと別に入れるので分離環境)
# ---------------------------------------------------------------------------------------------

ENV_SPECS: Dict[str, EnvSpec] = {
    "qwen": EnvSpec(
        name="qwen",
        packages=["qwen-asr==0.0.6"],
        check="import qwen_asr, transformers, torch; print(f'qwen-asr OK / transformers {transformers.__version__} / torch {torch.__version__}')",
        note="Qwen3-ASR と タイムスタンプ用アライナー(必須)",
    ),
    "fw": EnvSpec(
        name="fw",
        packages=["faster-whisper>=1.2.1", "nvidia-cublas-cu12"],
        check="import faster_whisper, ctranslate2; print(f'faster-whisper {faster_whisper.__version__} / ctranslate2 {ctranslate2.__version__}')",
        note="Whisper 系(faster-whisper)",
    ),
    "nemo": EnvSpec(
        name="nemo",
        packages=["nemo_toolkit[asr]>=2.4"],
        check="import nemo, nemo.collections.asr; print(f'nemo {nemo.__version__}')",
        note="NVIDIA NeMo(parakeet 日本語モデル)。インストールに数分かかる",
    ),
    "hf": EnvSpec(
        name="hf",
        packages=["transformers>=5.5", "accelerate", "librosa", "soundfile", "sentencepiece"],
        check="import transformers, torch; print(f'transformers {transformers.__version__} / torch {torch.__version__}')",
        note="最新 transformers の音声モデル(Cohere / Granite / VibeVoice / カスタム)",
    ),
    "pyannote": EnvSpec(
        name="pyannote",
        packages=["pyannote.audio>=4.0.4,<5"],
        check="import pyannote.audio, torch; print(f'pyannote.audio {pyannote.audio.__version__} / torch {torch.__version__}')",
        note="話者分離(Hugging Face の HF_TOKEN と規約同意が必要)",
    ),
    "vllm": EnvSpec(
        name="vllm",
        packages=["vllm[audio]"],
        # uv がドライバの CUDA を見て合う torch を選ぶ(vLLM 公式の推奨)
        pip_args=["--torch-backend=auto"],
        isolated=True,
        check=("import vllm, torch; ok = torch.cuda.is_available(); "
               "print(f'vllm {vllm.__version__} / torch {torch.__version__} (cuda {torch.version.cuda}, GPU {ok})')"),
        note="vLLM(T4 でも動くが A100/H100/L4 で真価。インストール数分)",
    ),
}


@dataclass
class Preset:
    key: str
    label: str
    engine: str  # qwen / vllm / faster-whisper / nemo / hf-pipeline / hf-speechlm
    env: str
    model: str
    options: Dict[str, Any] = field(default_factory=dict)
    vllm_args: List[str] = field(default_factory=list)
    ja: str = "◎"  # 日本語の相性(目安)
    punctuates: bool = True
    context: bool = True  # context(固有名詞のヒント)を使えるか
    native_ts: bool = False
    # クリップの区切り方のおすすめ(None = ふつう: 30 秒まで・6 秒以下の無音はつなぐ)。
    # 長いクリップや途中の無音で発話を読み飛ばしやすいモデルは短めにする。④ で「自動」のときだけ使う
    max_clip: Optional[float] = None
    max_gap: Optional[float] = None
    gpu: str = "T4〜"  # 目安
    license: str = ""
    note: str = ""


PRESETS: List[Preset] = [
    Preset("qwen3-1.7b", "Qwen3-ASR 1.7B（標準・おすすめ）", "qwen", "qwen", "Qwen/Qwen3-ASR-1.7B",
           license="Apache-2.0", note="v1/v2 と同じ。context で固有名詞に強い"),
    Preset("qwen3-0.6b", "Qwen3-ASR 0.6B（軽量・速い）", "qwen", "qwen", "Qwen/Qwen3-ASR-0.6B",
           ja="○", license="Apache-2.0", note="1.7B より少し精度が落ちるぶん速い"),
    Preset("qwen3-1.7b-ja", "Qwen3-ASR 1.7B JA（neosophie・固有名詞に強い日本語調整版）", "qwen", "qwen",
           "neosophie/Qwen3-ASR-1.7B-JA", license="Apache-2.0",
           note="IT・ビジネス系の固有名詞 F1 が 0.59→0.65。全体の CER は 8.23%→8.92% と少し悪化の報告"),
    Preset("qwen3-1.7b-vllm", "Qwen3-ASR 1.7B × vLLM（A100/H100で爆速）", "vllm", "vllm", "Qwen/Qwen3-ASR-1.7B",
           gpu="T4〜(A100/H100 で真価)", license="Apache-2.0", note="中身は標準と同じモデル。大量バッチで数倍〜十数倍速い"),
    Preset("cohere-transcribe", "Cohere Transcribe 2B（別系統の有力候補）", "cohere", "hf",
           "CohereLabs/cohere-transcribe-03-2026", options={"batch_size": 16}, context=False, gpu="T4〜",
           license="Apache-2.0", note="日本語 CER: FLEURS 2.89% / CV 20.2%(Qwen 5.28% / 26.3%)。HF で規約同意が必要"),
    Preset("cohere-transcribe-vllm", "Cohere Transcribe 2B × vLLM", "vllm", "vllm", "CohereLabs/cohere-transcribe-03-2026",
           context=False, gpu="T4〜(A100/H100 で真価)", license="Apache-2.0", note="Cohere を vLLM で高速に"),
    Preset("whisper-large-v3-turbo", "Whisper large-v3-turbo（faster-whisper）", "faster-whisper", "fw", "large-v3-turbo",
           options={"beam_size": 5}, ja="○", native_ts=True, license="MIT", note="定番。速い"),
    Preset("whisper-large-v3", "Whisper large-v3（faster-whisper）", "faster-whisper", "fw", "large-v3",
           options={"beam_size": 5}, ja="○", native_ts=True, license="MIT", note="定番の最高精度版"),
    Preset("kotoba-whisper-v2", "kotoba-whisper v2.0（日本語特化 Whisper）", "faster-whisper", "fw",
           "kotoba-tech/kotoba-whisper-v2.0-faster", options={"beam_size": 5, "word_timestamps": False, "chunk_length": 15},
           max_clip=15.0, license="MIT",
           note="ReazonSpeech で学習した日本語特化の蒸留モデル。公式のおすすめどおり 15 秒ずつ読む"
                "(タイムスタンプはアライナーで付ける。faster-whisper の単語時刻はこのモデルだと segfault するため)"),
    Preset("parakeet-ja", "Parakeet TDT-CTC 0.6B ja（NVIDIA・日本語特化）", "nemo", "nemo", "nvidia/parakeet-tdt_ctc-0.6b-ja",
           options={"batch_size": 16}, context=False, native_ts=True, punctuates=False, license="CC-BY-4.0",
           note="とても速い日本語専用モデル(JSUT の CER 6.60% の報告)。句読点は少なめ"),
    Preset("granite-speech-4.1", "Granite Speech 4.1 2B（IBM・実験的）", "granite", "hf", "ibm-granite/granite-speech-4.1-2b",
           options={"batch_size": 8}, ja="○", punctuates=False, license="Apache-2.0",
           note="日本語対応の2B(2026年4月)。context の語をキーワードとして渡すが、関係ない所にも入れがちなので語は少なめに"),
    Preset("vibevoice-asr", "VibeVoice-ASR 8B（Microsoft・実験的）", "vibevoice", "hf", "microsoft/VibeVoice-ASR-HF",
           options={"batch_size": 4}, ja="○", gpu="A100/H100/L4(24GB〜)", license="MIT",
           note="50以上の言語・context 対応の8B。本来は60分一気読み＋話者付けのモデル"),
]


# flash-attn の配布済み whl(torch のバージョン, Python タグ) → URL。無ければ sdpa で動く
# (torch 2.11 用は公式の whl が無いので、CUDA 12.8 でビルドされたコミュニティ版。Colab のカーネルは 3.12 → 3.13 に変わった)
_FA_211 = "https://github.com/lesj0610/flash-attention/releases/download/v2.8.3-cu12-torch2.11/"
FLASH_WHEELS: Dict[tuple, str] = {
    ("2.11", tag): f"{_FA_211}flash_attn-2.8.3%2Bcu12torch2.11cxx11abiTRUE-{tag}-{tag}-linux_x86_64.whl"
    for tag in ("cp312", "cp313")
}


def flash_attn_wheel() -> Optional[str]:
    """いまのカーネルの torch / Python に合う flash-attn の whl の URL(無ければ None)"""
    import sys

    try:
        import torch

        tv = ".".join(torch.__version__.split("+")[0].split(".")[:2])
    except Exception:
        return None
    tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    return FLASH_WHEELS.get((tv, tag))


def preset_labels() -> List[str]:
    return [p.label for p in PRESETS]


def find_preset(key_or_label: str) -> Optional[Preset]:
    s = (key_or_label or "").strip()
    for p in PRESETS:
        if s in (p.key, p.label):
            return p
    return None


def custom_preset(engine: str, model: str, env: Optional[str] = None) -> Preset:
    """フォームの「カスタム」用: エンジン名と HF のモデル ID から作る"""
    engine = engine.strip()
    env = env or {"qwen": "qwen", "vllm": "vllm", "faster-whisper": "fw", "nemo": "nemo",
                  "hf-pipeline": "hf", "hf-speechlm": "hf", "cohere": "hf", "granite": "hf",
                  "vibevoice": "hf"}.get(engine, "hf")
    return Preset(f"custom:{engine}:{model}", f"カスタム {engine}: {model}", engine, env, model.strip(),
                  native_ts=engine == "nemo" or (engine == "faster-whisper" and not any(
                      k in model.lower() for k in ("kotoba", "distil"))), ja="?", note="カスタム")


def envs_for(preset: Preset, diarize: bool, aligner: bool) -> List[str]:
    need = [preset.env]
    if aligner and "qwen" not in need:
        need.append("qwen")
    if diarize:
        need.append("pyannote")
    return need
