# -*- coding: utf-8 -*-
"""v3/asrkit/*.py と下のセル定義から Qwen3-ASR_v3.ipynb を組み立てる

  python v3/build_notebook.py          # ../Qwen3-ASR_v3.ipynb を書き出す
  python v3/build_notebook.py --check  # 生成物が最新かだけ確認(CI 用)
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import textwrap

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "Qwen3-ASR_v3.ipynb")
LIB_FILES = ["__init__.py", "core.py", "runtime.py", "engines.py", "worker.py", "presets.py", "pipeline.py", "webui.py"]

sys.path.insert(0, HERE)
from asrkit import presets  # noqa: E402


def md(src: str, cid: str) -> dict:
    return {"cell_type": "markdown", "id": cid, "metadata": {"id": cid}, "source": _lines(src)}


def code(src: str, cid: str, form: bool = True, collapsed: bool = False) -> dict:
    meta = {"id": cid}
    if form:
        meta["cellView"] = "form"
    if collapsed:
        meta["collapsed"] = True
    return {"cell_type": "code", "execution_count": None, "id": cid, "metadata": meta, "outputs": [],
            "source": _lines(src)}


def _lines(src: str) -> list:
    src = textwrap.dedent(src).strip("\n") + "\n"
    lines = src.splitlines(keepends=True)
    lines[-1] = lines[-1].rstrip("\n")
    return lines


def _pylit(s: str) -> str:
    return json.dumps(s, ensure_ascii=False)


# =====================================================================
# セルの中身
# =====================================================================

LABELS = presets.preset_labels()
DEFAULT_LABEL = LABELS[0]

INTRO = f"""
# Qwen3-ASR 会議文字起こし v3

音声/動画 → **句読点つき SRT・読みやすいテキスト・話者つき議事録用テキスト** を作るノートブックです。

**使い方**: ⓪ → ① を実行 → ②〜④ のフォームを埋める → ⑤ 実行（メニューの「ランタイム → すべてのセルを実行」でもOK）

**画面でお手軽に**: ⓪ → ① → いちばん下の **⑨ Web UI** を実行すると、音声をアップロードするだけで文字起こしできる画面が出ます

| | v2 からの主な改善 |
|---|---|
| モデル | Qwen3-ASR 1.7B/0.6B・vLLM 版・Whisper large-v3/turbo・kotoba-whisper・Parakeet 日本語 などをプルダウンで切り替え。**⑥で並べて比較**も |
| 環境 | モデルごとに別の venv で動かすので **依存の衝突なし・再起動いらず** |
| 字幕 | アライナーの単語に **句読点・記号を付け直す**ので SRT も句読点つき。句読点の所で自然に改行 |
| 区切り | v2 で消えていた「長すぎる区間のハード切り」を復活（静かな所を探して切る＋重なり除去） |
| 失敗対策 | context 復唱・ループ・空振りを検出 → context なし → 2分割 の順で自動リトライ |
| 話者 | 文字起こしと**並行して**話者分離。文単位の多数決でチラつき補正・話者名の置換・RTTM |
| 出力 | txt / srt / vtt / json / csv(Excel) / md(議事録用) / プレーン |
| 長時間 | 結果を少しずつキャッシュ → 切断されても**続きから再開**。フォルダ一括処理・アップロード対応 |
| 速度 | GPU に合わせてバッチ自動調整・長さ順バッチ・OOM で自動縮小・A100/H100 は vLLM |

<!-- 詳細（この行と末尾のコメント記号を外すと表示される）

## しくみ
- 音声は ffmpeg で 16kHz モノラルに変換（動画もOK）→ VAD で発話区間 → 最大 30 秒のクリップ（モデルによっては短め）
- ASR は各モデル専用の venv の中の「ワーカー」プロセスで動く（カーネルは汚さない）
- タイムスタンプは Qwen3-ForcedAligner（11言語）で全モデル共通に付ける
- 話者分離は pyannote community-1 の exclusive 出力（重なりなし）を単語に割り当て
- 本体コードは GitHub の `v3/asrkit/` と同じもの（⓪のセルに埋め込み）

-->
"""

LIB_CELL_HEAD = """
#@title ⓪ ライブラリの展開（さわらなくてOK・そのまま実行）
#@markdown v3 の本体コード（asrkit）を書き出して読み込みます。コードは GitHub の `v3/asrkit/` と同じです。
import os, sys, importlib
ASR_HOME = os.environ.setdefault("ASR_V3_HOME", "/content/asr_v3")
_LIB = os.path.join(ASR_HOME, "lib")
_FILES = {}
"""

LIB_CELL_TAIL = """
for _p, _src in _FILES.items():
    _fp = os.path.join(_LIB, "asrkit", _p)
    os.makedirs(os.path.dirname(_fp), exist_ok=True)
    with open(_fp, "w", encoding="utf-8") as _f:
        _f.write(_src)
if _LIB not in sys.path:
    sys.path.insert(0, _LIB)
if "webui" in globals():  # 動いている Web UI(⑨)を止める
    try:
        webui.stop()
    except Exception:
        pass
if "SESS" in globals():  # 作り直す前に古いワーカーを止める
    try:
        SESS.free()
    except Exception:
        pass
    del SESS
import asrkit
from asrkit import core, runtime, engines, presets, pipeline, webui
for _m in (asrkit, core, runtime, engines, presets, pipeline, webui):
    importlib.reload(_m)
print(f"✅ asrkit {asrkit.__version__} を読み込みました ({_LIB})")
"""


def lib_cell() -> str:
    parts = [textwrap.dedent(LIB_CELL_HEAD).strip("\n")]
    for fn in LIB_FILES:
        src = open(os.path.join(HERE, "asrkit", fn), encoding="utf-8").read()
        if "'''" in src or "#@" in src or src.rstrip().endswith("\\"):
            raise SystemExit(f"{fn} に埋め込めない文字列があります")
        parts.append(f"_FILES[{fn!r}] = r'''{src}'''")
    parts.append(textwrap.dedent(LIB_CELL_TAIL).strip("\n"))
    return "\n".join(parts)


SETUP = """
#@title ① セットアップ（インストール）
#@markdown 使うものにチェック。入れたものは次から一瞬でスキップされます（ランタイムを削除するまで）。
#@markdown
#@markdown **Qwen3-ASR（必須）**: 標準モデル＋タイムスタンプ用のアライナー
qwen = True  #@param {type:"boolean"}
#@markdown **話者分離（pyannote）**: 「誰が話したか」を付ける。Hugging Face の `HF_TOKEN` が必要（下の説明）
pyannote = True  #@param {type:"boolean"}
#@markdown **vLLM**: Qwen3-ASR / Cohere を大幅に高速化（A100 / H100 で真価。インストール 3〜6 分）
vllm = False  #@param {type:"boolean"}
#@markdown **faster-whisper**: Whisper large-v3 / large-v3-turbo / kotoba-whisper を試す
faster_whisper = False  #@param {type:"boolean"}
#@markdown **NeMo**: NVIDIA Parakeet 日本語モデルを試す（インストール数分）
nemo = False  #@param {type:"boolean"}
#@markdown **最新 transformers**: Cohere Transcribe / Granite Speech / VibeVoice-ASR や「カスタム」を試す用
hf = False  #@param {type:"boolean"}
#@markdown ---
#@markdown **flash-attn**: A100/H100/L4 で Qwen を少し速く・省メモリに（配布済み whl があるときだけ。なければ sdpa で動きます）
flash_attn = True  #@param {type:"boolean"}
#@markdown whl が無いときにソースからビルドする（20〜30分）
build_flash_if_missing = False  #@param {type:"boolean"}
#@markdown **Google Drive をマウント**（音声が Drive にあるとき）
mount_drive = True  #@param {type:"boolean"}
#@markdown ---
#@markdown **HF_TOKEN の準備（話者分離を使うとき・初回だけ）**
#@markdown 1. https://huggingface.co/settings/tokens で Read トークンを作る
#@markdown 2. https://huggingface.co/pyannote/speaker-diarization-community-1 で規約に同意
#@markdown 3. Colab 左の🔑（シークレット）に `HF_TOKEN` という名前で登録し、このノートブックのアクセスを ON

import os, sys, subprocess, time
g = runtime.gpu_info()
if g.ok:
    print(f"GPU: {g.name} / {g.mem_gb:.0f}GB / compute {g.cc[0]}.{g.cc[1]} / driver {g.driver} (CUDA {g.cuda})")
else:
    print("⚠️ GPU が見つかりません。メニュー「ランタイム → ランタイムのタイプを変更」で GPU を選んでください")

# --- カーネル側に VAD だけ入れる(軽い) ---
_need = []
for _mod, _pkg in (("fireredvad", "fireredvad"), ("silero_vad", "silero-vad"), ("rapidfuzz", "rapidfuzz")):
    try:
        __import__(_mod)
    except Exception:
        _need.append(_pkg)
if _need:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *_need], check=False)

# --- flash-attn (qwen 環境にだけ入れる) ---
_specs = []
_names = [n for n, on in (("qwen", qwen), ("pyannote", pyannote), ("vllm", vllm), ("fw", faster_whisper),
                          ("nemo", nemo), ("hf", hf)) if on]
for _n in _names:
    _s = presets.ENV_SPECS[_n]
    if _n == "qwen" and flash_attn and g.ampere_plus:
        _whl = presets.flash_attn_wheel()
        if _whl:
            _s = runtime.EnvSpec(**{**_s.__dict__, "post": [[_whl]]})
        elif build_flash_if_missing:
            print("flash-attn の配布 whl が無いのでソースからビルドします(20〜30分)…")
            _s = runtime.EnvSpec(**{**_s.__dict__, "post": [["ninja"], ["flash-attn", "--no-build-isolation"]]})
        else:
            print("ℹ️ いまの torch に合う flash-attn の whl が無いので sdpa で動かします(速度差は小さめ)")
    _specs.append(_s)
if vllm and g.ok and not g.ampere_plus:
    print("ℹ️ T4 でも vLLM は動きますが、速さの差が大きいのは A100 / L4 / H100 です")

t0 = time.time()
_st = runtime.ensure_envs(_specs)
print(f"\\nセットアップ完了 ({core.fmt_dur(time.time() - t0)})" + ("" if all(s.ok for s in _st) else " ※失敗したものがあります(上のログ)"))

if pyannote and not runtime.get_secret("HF_TOKEN"):
    print("⚠️ HF_TOKEN が見つかりません。話者分離を使うなら上の手順でシークレットに登録してください")
if mount_drive and runtime.in_colab() and not os.path.ismount("/content/drive"):
    from google.colab import drive
    drive.mount("/content/drive")
if "SESS" not in globals():
    SESS = pipeline.Session()
"""


def settings_cell() -> str:
    labels = ", ".join(_pylit(l) for l in LABELS + ["カスタム"])
    return f"""
#@title ② 音声とモデル
#@markdown **音源**: ファイル / フォルダ（中の音声・動画をぜんぶ）/ ワイルドカード（例 `/content/drive/MyDrive/会議/*.m4a`）。改行やカンマで複数指定も可。**空欄なら実行時にアップロード画面**が出ます
src = ""  #@param {{type:"string"}}
#@markdown **出力先**: 空欄なら音源と同じフォルダ。フォルダを指定するとその中に音源名で作ります
output_dir = ""  #@param {{type:"string"}}
#@markdown ---
#@markdown **モデル**（⑥で比べられます。「カスタム」は ④ で指定）
model = {_pylit(DEFAULT_LABEL)}  #@param [{labels}]
#@markdown **言語**（auto で自動判定。Qwen3-ASR は 30 言語＋中国語方言）
language = "Japanese"  #@param ["Japanese", "auto", "English", "Chinese", "Korean", "Cantonese", "French", "German", "Spanish", "Portuguese", "Italian", "Russian", "Thai", "Vietnamese", "Indonesian"] {{allow-input: true}}
#@markdown ---
#@markdown **context_label**: 語のリストを包む見出し（[issue #321](https://github.com/TypeWhisper/typewhisper-mac/issues/321) 方式。見出しで包むと精度UP・漏れ激減）
context_label = "固有名詞・専門用語"  #@param ["固有名詞・専門用語", "Proper nouns", "Technical terms", "Vocabulary"] {{allow-input: true}}
#@markdown **context**: 固有名詞・専門用語（スペース/読点/カンマ区切り）
context = ""  #@param {{type:"string", placeholder:"Claude Codex Gemini 競争 脱落"}}
#@markdown **用語ファイル**（任意）: 1行1語などのテキストファイル。context に足されます
context_file = ""  #@param {{type:"string"}}
#@markdown *Tips*: 会議資料を Claude/ChatGPT に渡して「固有名詞・専門用語を読点区切りで列挙して」と頼むと楽。入れすぎると復唱を誘発しやすいけど、v3 は復唱を見つけたら context なしで自動でやり直します
#@markdown ---
#@markdown **出力形式**
txt = True  #@param {{type:"boolean"}}
srt = True  #@param {{type:"boolean"}}
json_ = True  #@param {{type:"boolean"}}
vtt = False  #@param {{type:"boolean"}}
csv = False  #@param {{type:"boolean"}}
#@markdown `csv` は Excel 用（BOM 付き）、`md` は議事録づくり用の Markdown、`plain` はタイムスタンプなしのテキスト
md = True  #@param {{type:"boolean"}}
plain = False  #@param {{type:"boolean"}}

_fmts = tuple(f for f, on in (("txt", txt), ("srt", srt), ("json", json_), ("vtt", vtt), ("csv", csv), ("md", md), ("plain", plain)) if on)
_terms = core.parse_terms(context)
if context_file.strip():
    _terms += [w for w in core.parse_terms(open(context_file.strip(), encoding="utf-8").read()) if w not in _terms]
print("モデル:", model, "/ 言語:", language)
print("context:", core.build_context(context_label, _terms) or "(なし)")
print("出力形式:", ", ".join(_fmts))
"""


DIAR = """
#@title ③ 話者分離
#@markdown **誰が話したか**を付けます（① で pyannote を入れて HF_TOKEN を登録しておく）。文字起こしと並行して走ります
diarize = True  #@param {type:"boolean"}
#@markdown **人数**: わかっていれば入れると精度が上がる（0 = 自動）。幅で指定したいときは min/max
num_speakers = 0  #@param {type:"integer"}
min_speakers = 0  #@param {type:"integer"}
max_speakers = 0  #@param {type:"integer"}
#@markdown **話者名**: `話者A=田中, 話者B=佐藤` のように置き換え（登場順に `田中, 佐藤` だけでもOK）。一度流して確認してから入れると楽
speaker_names = ""  #@param {type:"string"}
#@markdown **表記**
speaker_style = "話者A"  #@param ["話者A", "SPEAKER_00", "S1"]
#@markdown **文単位の補正**: 文の途中で話者がチラつくのを多数決で直す
smooth_speakers = True  #@param {type:"boolean"}
#@markdown **モデル**
diar_model = "pyannote/speaker-diarization-community-1"  #@param ["pyannote/speaker-diarization-community-1", "pyannote/speaker-diarization-3.1"] {allow-input: true}
#@markdown **RTTM も出力**（話者分離の評価ツール用）
rttm = False  #@param {type:"boolean"}
"""


ADV_TEMPLATE = """
#@title ④ 詳細設定（ふだんは触らなくてOK）
#@markdown ### 音声
#@markdown **チャンネル**: ステレオで左右に別の人のマイクが入っているときなど
channel = "mix"  #@param ["mix", "left", "right"]
#@markdown **音声フィルタ**: 小さい声が多い会議は dynaudnorm、ノイズが多いときは afftdn（ffmpeg のフィルタを直接書いてもOK）
audio_filter = "なし"  #@param ["なし", "音量をそろえる (loudnorm)", "小さい声を持ち上げる (dynaudnorm)", "ノイズを少し減らす (afftdn)", "低音ノイズ除去+音量 (highpass+loudnorm)"] {allow-input: true}
#@markdown ### 区間検出（VAD）とクリップ
vad = "fireredvad"  #@param ["fireredvad", "silero", "energy", "none"]
#@markdown **vad_threshold**: 発話と判定するしきい値（上げると厳しめ＝区間が減る）
vad_threshold = 0.4  #@param {type:"slider", min:0.05, max:0.95, step:0.05}
#@markdown **max_clip_sec**: 1回で ASR に渡す最大秒数。「自動」はモデルのおすすめ（ふつう 30 秒。kotoba-whisper などは短め）。長いほど文脈が効くが重い
max_clip_sec = "自動"  #@param ["自動", "10", "15", "20", "30", "45", "60"] {allow-input: true}
#@markdown **max_gap_sec**: これより長い無音をはさむ発話は別クリップにする。「自動」はモデルのおすすめ（ふつう 6 秒）
max_gap_sec = "自動"  #@param ["自動", "1", "2", "3", "6", "10"] {allow-input: true}
#@markdown **overlap_sec**: ハード切り（長い発話の強制分割）の前後の重なり。重複は自動で除去
overlap_sec = 1.0  #@param {type:"number"}
#@markdown ### 推論
#@markdown **batch_size**: 0 = GPU のメモリから自動（OOM したら自動で半分にして続行）
batch_size = 0  #@param {type:"integer"}
max_new_tokens = 1024  #@param {type:"integer"}
#@markdown **retry**: 怪しい結果（context の復唱・ループ・空振り）を自動でやり直す
retry = True  #@param {type:"boolean"}
#@markdown **aligner**: Qwen3-ForcedAligner で単語タイムスタンプを付ける（OFF だとモデル固有 or 概算）
aligner = True  #@param {type:"boolean"}
#@markdown **vllm_concurrency**: vLLM に同時に投げるクリップ数
vllm_concurrency = 64  #@param {type:"integer"}
#@markdown **セカンドオピニオン**: 最後まで怪しいクリップだけ別のモデルでも読んで、怪しさが減るなら差し替える（元の結果も残る。① でそのモデルの環境を入れておく）
second_opinion = "なし"  #@param [SECOND_OPINION_OPTIONS]
#@markdown **要確認リスト**: 怪しい区間・自動で直した区間を `_review.md` に書き出し、実行後に音声つきで表示
review = True  #@param {type:"boolean"}
#@markdown **カスタム**（② で「カスタム」を選んだとき）: エンジンと Hugging Face のモデル ID
custom_engine = "hf-pipeline"  #@param ["hf-pipeline", "hf-speechlm", "cohere", "granite", "vibevoice", "faster-whisper", "nemo", "qwen", "vllm"]
custom_model = ""  #@param {type:"string"}
#@markdown ### テキスト整形
#@markdown **fillers**: 「えー」「あのー」などを消す（safe = 伸ばし音のフィラーだけ / more = 「あの」「まあ」なども）
fillers = "off"  #@param ["off", "safe", "more"]
#@markdown **置換辞書**: `誤=>正` を `;` 区切りで（例 `クロード=>Claude; re:ジェミ[ニ二]=>Gemini`）。ファイル（1行1ルール）も可
replacements = ""  #@param {type:"string"}
replacements_file = ""  #@param {type:"string"}
#@markdown 全角英数字を半角に
halfwidth = True  #@param {type:"boolean"}
#@markdown ### 字幕（SRT/VTT）
cue_max_chars = 30  #@param {type:"integer"}
cue_max_sec = 6.0  #@param {type:"number"}
#@markdown これ以上の無音で字幕を切る（秒）
cue_gap_sec = 0.8  #@param {type:"number"}
#@markdown 話者つき字幕の書式
speaker_fmt = "{speaker}: {text}"  #@param ["{speaker}: {text}", "【{speaker}】{text}", "（{speaker}）{text}"] {allow-input: true}
#@markdown ### そのほか
#@markdown **キャッシュ**: 出力先の `.asr_v3_cache/` に途中結果を保存（切断されても続きから）
use_cache = True  #@param {type:"boolean"}
#@markdown 出力がそろっているファイルは飛ばす（フォルダ一括のとき便利）
skip_existing = False  #@param {type:"boolean"}
"""


def adv_cell() -> str:
    opts = ", ".join(_pylit(x) for x in ["なし"] + LABELS)
    return ADV_TEMPLATE.replace("[SECOND_OPINION_OPTIONS]", f"[{opts}]")


RUN = """
#@title ⑤ 実行
#@markdown 終わったら結果を zip でダウンロード（アップロードで使ったとき便利）
download_zip = False  #@param {type:"boolean"}
import os, glob
_AF = {"なし": "", "音量をそろえる (loudnorm)": "loudnorm=I=-20:TP=-2:LRA=11",
       "小さい声を持ち上げる (dynaudnorm)": "dynaudnorm=f=250:g=15",
       "ノイズを少し減らす (afftdn)": "afftdn=nf=-25",
       "低音ノイズ除去+音量 (highpass+loudnorm)": "highpass=f=80,loudnorm=I=-20:TP=-2:LRA=11"}
_rep = replacements.replace(";", "\\n")
if replacements_file.strip():
    _rep += "\\n" + open(replacements_file.strip(), encoding="utf-8").read()
ST = pipeline.Settings(
    output_dir=output_dir, formats=_fmts + (("rttm",) if rttm else ()), skip_existing=skip_existing,
    preset=("custom" if model == "カスタム" else model), custom_engine=custom_engine, custom_model=custom_model,
    language=language, batch_size=batch_size, max_new_tokens=max_new_tokens, aligner=aligner,
    channel=channel, af=_AF.get(audio_filter, audio_filter),
    vad=vad, vad_threshold=vad_threshold, max_clip=pipeline.auto_num(max_clip_sec), max_gap=pipeline.auto_num(max_gap_sec),
    overlap=overlap_sec,
    context_label=context_label, context_terms=_terms, retry=retry,
    replacements=_rep, fillers=fillers, halfwidth=halfwidth,
    cue_max_chars=cue_max_chars, cue_max_dur=cue_max_sec, cue_gap=cue_gap_sec, speaker_fmt=speaker_fmt,
    diarize=diarize, diar_model=diar_model, num_speakers=num_speakers, min_speakers=min_speakers,
    max_speakers=max_speakers, speaker_names=speaker_names, speaker_style=speaker_style, smooth_speakers=smooth_speakers,
    vllm_concurrency=vllm_concurrency, cache=use_cache,
    second_opinion=("" if second_opinion == "なし" else second_opinion), review=review,
)
if "SESS" not in globals():
    SESS = pipeline.Session()

# 音源: 空欄ならアップロード
_src = src.strip()
if not _src:
    from google.colab import files
    print("音声/動画ファイルを選んでください")
    _up = files.upload()
    _src = "\\n".join(os.path.abspath(k) for k in _up)
if "/content/drive" in _src + output_dir and not os.path.ismount("/content/drive"):
    from google.colab import drive
    drive.mount("/content/drive")
INPUTS = pipeline.resolve_inputs(_src)
if not INPUTS:
    raise FileNotFoundError(f"音声ファイルが見つかりません: {_src}")
print(f"{len(INPUTS)} ファイル: " + ", ".join(os.path.basename(p) for p in INPUTS[:5]) + (" …" if len(INPUTS) > 5 else ""))
OUTS = SESS.run(INPUTS, ST)

if download_zip:
    import zipfile
    from google.colab import files
    _zip = "/content/asr_v3_outputs.zip"
    with zipfile.ZipFile(_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for o in OUTS:
            for p in (o.get("paths") or {}).values():
                z.write(p, os.path.basename(p))
    files.download(_zip)
"""


def compare_cell() -> str:
    lines = [
        "#@title ⑥ モデル比較（おまけ）",
        "#@markdown 同じ区間を複数のモデルで文字起こしして、**速さ・VRAM・文字誤り率(CER)** を並べます。① で環境を入れたモデルだけ動きます",
        "#@markdown 正解テキストがあれば、用語・数字・否定（「ない」「ません」）を正しく書けたか、発話を**丸ごと読み飛ばした所**がないかも数えます",
        "#@markdown ④ のクリップ長が「自動」なら、モデルごとにおすすめの区切り方で読みます",
        "#@markdown 結果は出力先の `compare/` フォルダに（モデルごとの txt / srt / json）",
    ]
    names = []
    for i, p in enumerate(presets.PRESETS):
        var = "cmp_" + "".join(ch if ch.isalnum() else "_" for ch in p.key)
        names.append((var, p.key))
        default = "True" if p.key in ("qwen3-1.7b", "qwen3-1.7b-ja") else "False"
        lines.append(f"#@markdown {p.label}（{p.note}）" if p.note else f"#@markdown {p.label}")
        lines.append(f"{var} = {default}  #@param {{type:\"boolean\"}}")
    lines += [
        "#@markdown ---",
        "#@markdown **比べる区間**（秒）: 先頭から5分など。長いほど正確だけど時間がかかる",
        "start_sec = 0  #@param {type:\"number\"}",
        "duration_sec = 300  #@param {type:\"number\"}",
        "#@markdown **正解テキスト**（任意）: 同じ区間を人が書き起こしたテキストのファイルパス（または本文）。あると CER で比べられる",
        "reference = \"\"  #@param {type:\"string\"}",
        "#@markdown **音源**: 空欄なら ⑤ の1つ目のファイル",
        "compare_src = \"\"  #@param {type:\"string\"}",
        "",
        "_keys = [k for v, k in " + repr(names) + " if globals().get(v)]",
        "_csrc = compare_src.strip() or (INPUTS[0] if \"INPUTS\" in globals() and INPUTS else \"\")",
        "if not _csrc:",
        "    raise ValueError(\"比較する音源がありません(compare_src に入れるか、先に ⑤ を実行)\")",
        "if \"SESS\" not in globals():",
        "    SESS = pipeline.Session()",
        "_st = ST if \"ST\" in globals() else pipeline.Settings()",
        "CMP = SESS.compare(_csrc, _keys, _st, start=start_sec, duration=duration_sec, reference=reference)",
    ]
    return "\n".join(lines)


WEBUI = """
#@title ⑨ Web UI（Gradio・アップロードしてすぐ文字起こし）
#@markdown ブラウザの画面から、音声/動画のアップロード（またはマイク録音）→ モデルを選んで文字起こし → 結果をダウンロード、ができます
#@markdown - ① のあとならいつでも立ち上げられます（⑤ を実行済みなら、④ の細かい設定と ② の context を引き継ぎます）
#@markdown - ふだんは**このセルの下**に画面が出ます（あなたのブラウザからだけ見られます）
#@markdown - `share` を ON にすると、だれでも開ける公開 URL（gradio.live・72時間）ができます。会議の音声を扱うときは `password` も入れてください（ユーザー名は `asr`）
share = False  #@param {type:"boolean"}
password = ""  #@param {type:"string"}
port = 7860  #@param {type:"integer"}
import subprocess, sys
try:
    import gradio
except ImportError:
    print("gradio を入れています…")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gradio"], check=True)
if "SESS" not in globals():
    SESS = pipeline.Session()
DEMO = webui.launch(SESS, ST if "ST" in globals() else pipeline.Settings(), share=share, port=port, password=password)
"""


CLEAN = """
#@title ⑧ 後片付け（VRAM 解放）
#@markdown モデルを読み込んだままのワーカーと vLLM サーバーを止めて GPU メモリを空けます（次の実行では自動で読み込み直し）
if "webui" in globals():
    webui.stop()  # ⑨ の画面も止める
if "SESS" in globals():
    print("\\n".join(SESS.h.status()) or "(動いているものはありません)")
    SESS.free()
"""


def build() -> dict:
    cells = [
        md(INTRO, "v3intro"),
        code(lib_cell(), "v3lib", collapsed=True),
        code(SETUP, "v3setup"),
        code(settings_cell(), "v3main"),
        code(DIAR, "v3diar"),
        code(adv_cell(), "v3adv"),
        code(RUN, "v3run"),
        md("""
        ---
        ## おまけ
        - **⑥ モデル比較**: 同じ区間をいろいろなモデルで文字起こしして比べる
        - **⑦ 議事録づくり**: 文字起こし結果から議事録を作る（プロンプトを作るだけ / Colab の Gemini（無料）/ Claude API）
        - **⑨ Web UI**: ブラウザの画面でアップロード → 文字起こし（Gradio）
        """, "v3extra"),
        code(compare_cell(), "v3cmp"),
        code(minutes_cell(), "v3min"),
        code(CLEAN, "v3clean"),
        code(WEBUI, "v3webui"),
    ]
    nb = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "H100", "provenance": [], "toc_visible": True},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return nb


def minutes_cell() -> str:
    return MINUTES


MINUTES = """
#@title ⑦ 議事録づくり（おまけ）
#@markdown ⑤ の結果（`.md` か `.txt`）から議事録を作ります
#@markdown - **プロンプトだけ作る**: `_minutes_prompt.md` を書き出すので、Claude や ChatGPT に貼るだけ（外には何も送りません）
#@markdown - **Gemini（Colab AI・無料）**: Colab に組み込みの Gemini で、API キーなしでここで議事録まで作ります
#@markdown - **Claude API**: Colab のシークレットに `ANTHROPIC_API_KEY` を登録しておくと、ここで議事録まで作ります
#@markdown
#@markdown ※ Gemini / Claude を選ぶと、文字起こしの中身がそのサービスに送られます
mode = "プロンプトだけ作る"  #@param ["プロンプトだけ作る", "Gemini（Colab AI・無料）", "Claude API"]
style = "議事録（決定事項・TODO つき）"  #@param ["議事録（決定事項・TODO つき）", "要約（3行＋詳細）", "発言者ごとの要点"] {allow-input: true}
#@markdown 対象（空欄なら ⑤ の最後のファイル）
transcript = ""  #@param {type:"string"}
#@markdown Gemini のモデル（`from google.colab import ai; ai.list_models()` で一覧）
gemini_model = "google/gemini-3.5-flash"  #@param ["google/gemini-3.5-flash", "google/gemini-3.1-pro-preview"] {allow-input: true}
#@markdown Claude のモデル（既定は Claude Opus 5。安全フィルタで断られたときは自動で別モデルに引き継ぐ設定にしてあります）
claude_model = "claude-opus-5"  #@param ["claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5"] {allow-input: true}
import os
_t = transcript.strip()
if not _t and "OUTS" in globals() and OUTS:
    _paths = OUTS[-1].get("paths", {})
    _t = _paths.get("md") or _paths.get("txt") or ""
if not _t:
    raise ValueError("対象のテキストがありません(⑤ を実行するか transcript にパスを入れてください)")
MINUTES_OUT = pipeline.make_minutes(_t, style=style, mode=mode, model=claude_model, gemini_model=gemini_model,
                                    glossary=_terms if "_terms" in globals() else ())
"""


def main() -> None:
    nb = build()
    text = json.dumps(nb, ensure_ascii=False, indent=1) + "\n"
    if "--check" in sys.argv:
        cur = open(OUT, encoding="utf-8").read() if os.path.exists(OUT) else ""
        if cur != text:
            raise SystemExit("Qwen3-ASR_v3.ipynb が古いです。python v3/build_notebook.py を実行してください")
        print("up to date")
        return
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"wrote {OUT} ({len(text) // 1024} KB, {len(nb['cells'])} cells, sha1 {hashlib.sha1(text.encode()).hexdigest()[:10]})")


if __name__ == "__main__":
    main()
