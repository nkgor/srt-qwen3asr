# -*- coding: utf-8 -*-
"""asrkit.core — 音声の読み込み・区間分割・テキスト処理・出力の「純粋ロジック」

ノートブック本体(カーネル)とエンジン用ワーカー(別venv)の両方から import される。
import 時に必要なのは標準ライブラリと numpy だけ(torch などは使わない)。
"""
from __future__ import annotations

import bisect
import csv
import io
import json
import math
import os
import re
import struct
import subprocess
import unicodedata
import wave
import zlib
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

SR = 16000

# =====================================================================
# 時刻の表記
# =====================================================================


def _split_ms(t: float) -> Tuple[int, int, int, int]:
    ms = int(round(max(0.0, float(t)) * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return h, m, s, ms


def fmt_hms(t: float) -> str:
    """[HH:MM:SS] 用。ミリ秒は切り捨て"""
    t = max(0, int(t))
    return f"{t // 3600:02d}:{t % 3600 // 60:02d}:{t % 60:02d}"


def fmt_srt_time(t: float) -> str:
    h, m, s, ms = _split_ms(t)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def fmt_vtt_time(t: float) -> str:
    h, m, s, ms = _split_ms(t)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def fmt_dur(sec: float) -> str:
    """ログ用のざっくり表記 (例: 1:02:03 / 4:05 / 12.3秒)"""
    sec = float(sec)
    if sec < 60:
        return f"{sec:.1f}秒"
    t = int(round(sec))
    h, r = divmod(t, 3600)
    m, s = divmod(r, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


# =====================================================================
# 音声 I/O (ffmpeg でデコード → 16kHz モノラル int16 WAV → memmap)
# =====================================================================

AUDIO_EXTS = (
    ".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".oga", ".opus", ".wma", ".aiff", ".aif",
    ".amr", ".3gp", ".webm", ".mp4", ".m4v", ".mov", ".mkv", ".avi", ".wmv", ".mts", ".ts",
)


def _ffmpeg_info(path: str) -> Dict[str, Any]:
    """ffprobe が無い環境用: `ffmpeg -i` の表示から読み取る"""
    try:
        p = subprocess.run(["ffmpeg", "-hide_banner", "-i", path], capture_output=True, text=True, timeout=120)
    except Exception:
        return {}
    err = p.stderr or ""
    out: Dict[str, Any] = {"channels": 0, "sample_rate": 0, "duration": None, "has_audio": False}
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", err)
    if m:
        out["duration"] = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    m = re.search(r"Stream #\S+.*?Audio:.*?(\d+) Hz,\s*([^,]+)", err)
    if m:
        out["has_audio"] = True
        out["sample_rate"] = int(m.group(1))
        lay = m.group(2).strip()
        mm = re.match(r"(\d+) channels", lay)
        out["channels"] = int(mm.group(1)) if mm else {"mono": 1, "stereo": 2}.get(lay, 2 if "." in lay else 1)
    return out


def ffprobe(path: str) -> Dict[str, Any]:
    """先頭の音声ストリームの情報(channels, sample_rate, duration)を返す。取れなければ空dict"""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "a:0",
        "-show_entries", "stream=channels,sample_rate,duration:format=duration",
        "-of", "json", path,
    ]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        d = json.loads(p.stdout or "{}")
    except FileNotFoundError:
        return _ffmpeg_info(path)
    except Exception:
        return {}
    st = (d.get("streams") or [{}])[0]
    fmt = d.get("format") or {}
    dur = st.get("duration") or fmt.get("duration")
    out = {
        "channels": int(st.get("channels") or 0),
        "sample_rate": int(st.get("sample_rate") or 0),
        "duration": float(dur) if dur not in (None, "N/A") else None,
        "has_audio": bool(st),
    }
    return out


def ffmpeg_to_wav16(
    src: str,
    dst: str,
    *,
    sr: int = SR,
    channel: str = "mix",
    af: str = "",
    start: Optional[float] = None,
    duration: Optional[float] = None,
) -> None:
    """どんな音声/動画でも 16kHz モノラル int16 の WAV に変換する。

    channel: "mix"(全チャンネル平均) / "left" / "right"
    af     : ffmpeg の追加フィルタ(例: "loudnorm=I=-20:TP=-2:LRA=11")
    """
    info = ffprobe(src)
    if info and not info.get("has_audio"):
        raise RuntimeError(f"音声ストリームが見つかりません: {src}")
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y"]
    if start:
        cmd += ["-ss", f"{float(start):.3f}"]
    if duration:
        cmd += ["-t", f"{float(duration):.3f}"]
    cmd += ["-i", src, "-map", "0:a:0", "-vn", "-sn", "-dn"]
    filters = []
    nch = info.get("channels") or 0
    if channel in ("left", "right") and nch >= 2:
        filters.append("pan=mono|c0=c0" if channel == "left" else "pan=mono|c0=c1")
    if af and af.strip():
        filters.append(af.strip())
    if filters:
        cmd += ["-af", ",".join(filters)]
    cmd += ["-ac", "1", "-ar", str(sr), "-c:a", "pcm_s16le", "-f", "wav", dst]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0 or not os.path.exists(dst):
        raise RuntimeError(f"ffmpeg での変換に失敗しました: {src}\n{p.stderr[-3000:]}")


def _parse_wav_header(path: str) -> Tuple[int, int, int, int, int]:
    """(data_offset, data_bytes, channels, sample_rate, bits) を返す。RIFF の LIST 等も読み飛ばす"""
    size_total = os.path.getsize(path)
    with open(path, "rb") as f:
        head = f.read(12)
        if len(head) < 12 or head[:4] != b"RIFF" or head[8:12] != b"WAVE":
            raise ValueError(f"WAV ではありません: {path}")
        ch = sr = bits = None
        while True:
            hdr = f.read(8)
            if len(hdr) < 8:
                raise ValueError(f"WAV の data チャンクが見つかりません: {path}")
            cid = hdr[:4]
            size = struct.unpack("<I", hdr[4:])[0]
            if cid == b"fmt ":
                fmt = f.read(size)
                _fmt_tag, ch, sr, _br, _ba, bits = struct.unpack("<HHIIHH", fmt[:16])
                if size % 2:
                    f.read(1)
            elif cid == b"data":
                off = f.tell()
                if size in (0, 0xFFFFFFFF) or off + size > size_total:
                    size = size_total - off
                if ch is None:
                    raise ValueError("fmt チャンクがありません")
                return off, size, ch, sr, bits
            else:
                f.seek(size + (size % 2), 1)


class Wav16:
    """16bit モノラル WAV を memmap で持つ。get(start, end) で float32 の切り出しを返す"""

    def __init__(self, path: str):
        off, nbytes, ch, sr, bits = _parse_wav_header(path)
        if ch != 1 or bits != 16:
            raise ValueError(f"16bit モノラルのみ対応: ch={ch} bits={bits}")
        self.path = path
        self.sr = int(sr)
        n = nbytes // 2
        self.data = np.memmap(path, dtype="<i2", mode="r", offset=off, shape=(n,)) if n else np.zeros(0, "<i2")

    def __len__(self) -> int:
        return int(self.data.shape[0])

    @property
    def duration(self) -> float:
        return len(self) / float(self.sr)

    def get(self, start: float, end: float) -> np.ndarray:
        a = max(0, int(round(start * self.sr)))
        b = min(len(self), int(round(end * self.sr)))
        if b <= a:
            return np.zeros(0, np.float32)
        return np.asarray(self.data[a:b], dtype=np.float32) / 32768.0

    def int16(self) -> np.ndarray:
        return np.asarray(self.data)


def wav_bytes(audio: np.ndarray, sr: int = SR) -> bytes:
    """float32 [-1,1] → WAV(int16) のバイト列 (HTTP 送信用)"""
    x = np.clip(np.asarray(audio, np.float32), -1.0, 1.0)
    pcm = (x * 32767.0).astype("<i2").tobytes()
    bio = io.BytesIO()
    with wave.open(bio, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm)
    return bio.getvalue()


def write_wav16(path: str, audio: np.ndarray, sr: int = SR) -> None:
    with open(path, "wb") as f:
        f.write(wav_bytes(audio, sr))


# =====================================================================
# エネルギー(dB) と エネルギーVAD
# =====================================================================

HOP = 0.02  # 20ms


def frame_db(wav: Wav16, hop: float = HOP, chunk_frames: int = 200_000) -> np.ndarray:
    """20ms ごとの RMS(dB, int16 スケール)。3時間でも数秒で終わるようにチャンク処理"""
    hop_n = max(1, int(round(hop * wav.sr)))
    n_frames = len(wav) // hop_n
    out = np.empty(n_frames, np.float32)
    for i in range(0, n_frames, chunk_frames):
        j = min(n_frames, i + chunk_frames)
        x = np.asarray(wav.data[i * hop_n: j * hop_n], dtype=np.float32).reshape(j - i, hop_n)
        out[i:j] = 10.0 * np.log10(np.mean(x * x, axis=1) + 1e-3)
    return out


def _runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    """True が続く区間 [i0, i1) の一覧"""
    if mask.size == 0:
        return []
    m = np.concatenate([[False], mask.astype(bool), [False]])
    d = np.diff(m.astype(np.int8))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def energy_vad(
    db: np.ndarray,
    hop: float = HOP,
    top_db: float = 45.0,
    min_speech: float = 0.25,
    min_silence: float = 0.4,
) -> List[Tuple[float, float]]:
    """依存なしの簡易VAD(v1 の librosa.effects.split 相当)。top_db は最大音量からの相対しきい値"""
    if db.size == 0:
        return []
    ref = float(np.percentile(db, 99.5))
    speech = db > (ref - top_db)
    segs = _runs(speech)
    merged: List[List[int]] = []
    gap_n = int(round(min_silence / hop))
    for a, b in segs:
        if merged and a - merged[-1][1] < gap_n:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    min_n = int(round(min_speech / hop))
    return [(a * hop, b * hop) for a, b in merged if b - a >= min_n]


def normalize_segments(
    segs: Iterable[Sequence[float]], total: float, min_len: float = 0.05
) -> List[Tuple[float, float]]:
    """並べ替え・範囲外カット・重なり結合"""
    out: List[List[float]] = []
    for s, e in sorted((float(a), float(b)) for a, b in segs):
        s, e = max(0.0, s), min(total, e)
        if e - s < min_len:
            continue
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(a, b) for a, b in out]


# =====================================================================
# クリップ(ASRに渡す単位)の組み立て
# =====================================================================


@dataclass
class Clip:
    id: int
    start: float  # 音声の切り出し開始(秒, パディング/オーバーラップ込み)
    end: float
    own_start: float = -1e18  # 重複除去用の担当区間(ハード切りの境界だけ有限)
    own_end: float = 1e18
    speech: float = 0.0  # クリップ内のVAD発話秒数
    parent: Optional[int] = None  # リトライで分割された場合の元クリップ
    cut: bool = False  # 右の境界がハード切り(長い発話を強制分割)か

    @property
    def dur(self) -> float:
        return self.end - self.start

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["own_start"] = None if self.own_start <= -1e17 else round(self.own_start, 3)
        d["own_end"] = None if self.own_end >= 1e17 else round(self.own_end, 3)
        d["start"], d["end"], d["speech"] = round(self.start, 3), round(self.end, 3), round(self.speech, 3)
        return d

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Clip":
        return Clip(
            id=int(d["id"]),
            start=float(d["start"]),
            end=float(d["end"]),
            own_start=-1e18 if d.get("own_start") is None else float(d["own_start"]),
            own_end=1e18 if d.get("own_end") is None else float(d["own_end"]),
            speech=float(d.get("speech") or 0.0),
            parent=d.get("parent"),
            cut=bool(d.get("cut", False)),
        )


def find_cut(db: Optional[np.ndarray], lo: float, hi: float, hop: float = HOP) -> float:
    """[lo, hi] の中でいちばん静かな時刻を返す(エネルギーが無ければ中央)"""
    if hi <= lo:
        return lo
    if db is None or db.size == 0:
        return (lo + hi) / 2
    a = max(0, int(lo / hop))
    b = min(db.size, int(math.ceil(hi / hop)))
    if b - a < 3:
        return (lo + hi) / 2
    seg = db[a:b]
    # 5フレーム(100ms)平滑してから最小を取る(瞬間的な無音より“間”を優先)
    k = min(5, seg.size)
    sm = np.convolve(seg, np.ones(k, np.float32) / k, mode="same")
    i = int(np.argmin(sm))
    return (a + i + 0.5) * hop


def _speech_in(segs: Sequence[Tuple[float, float]], a: float, b: float) -> float:
    tot = 0.0
    for s, e in segs:
        if e <= a:
            continue
        if s >= b:
            break
        tot += min(b, e) - max(a, s)
    return tot


def build_clips(
    segs: Sequence[Tuple[float, float]],
    total: float,
    *,
    max_clip: float = 30.0,
    max_gap: float = 6.0,
    pad: float = 0.5,
    overlap: float = 1.0,
    db: Optional[np.ndarray] = None,
    hop: float = HOP,
) -> List[Clip]:
    """VAD 区間を「余白込みで max_clip 秒以下」のクリップにまとめる。

    - モデルには前後を少し余分に聞かせる(ふつうの境界は pad 秒・ハード切りは overlap 秒)
    - そのかわり単語は「担当区間」(own_start〜own_end)に中点があるものだけ採用する
      → 境界で語頭/語尾が欠けるのを防ぎつつ、重なった部分の二重転写は出ない
      担当区間の境目は、ふつうの境界なら無音の真ん中、ハード切りなら切った時刻
    - 隣り合う発話は「余白を除いた長さが上限以内」かつ「無音が max_gap 以下」なら 1 クリップに結合
    - 1つの発話が長すぎるときは、なるべく静かな所でハード切り(v1 にあって v2 で抜けていた処理)
    """
    segs = normalize_segments(segs, total)
    if not segs:
        return []
    max_clip = max(2.0, float(max_clip))
    pad = max(0.0, min(float(pad), max_clip * 0.1))
    overlap = max(0.0, min(float(overlap), max_clip * 0.15))
    body = max_clip - 2 * max(pad, overlap)  # 余白を足しても max_clip を超えない本体の長さ

    # 1) 近い発話を結合
    groups: List[List[float]] = []
    cur = [segs[0][0], segs[0][1]]
    for s, e in segs[1:]:
        if e - cur[0] <= body and s - cur[1] <= max_gap:
            cur[1] = e
        else:
            groups.append(cur)
            cur = [s, e]
    groups.append(cur)

    # 2) 長すぎるグループをハード切り → pieces: (start, end, hard_left, hard_right)
    pieces: List[Tuple[float, float, bool, bool]] = []
    for gs, ge in groups:
        if ge - gs <= body:
            pieces.append((gs, ge, False, False))
            continue
        start = gs
        hard_left = False
        while ge - start > body:
            remain = ge - start
            n = math.ceil(remain / body)
            ideal = remain / n
            lo = start + max(ideal * 0.6, min(5.0, body * 0.5))
            hi = start + min(body, ideal * 1.25)
            cut = find_cut(db, lo, hi, hop)
            cut = min(max(cut, start + 1.0), start + body)
            pieces.append((start, cut, hard_left, True))
            start, hard_left = cut, True
        pieces.append((start, ge, hard_left, False))

    # 3) 余白と担当区間
    clips: List[Clip] = []
    for i, (s, e, hl, hr) in enumerate(pieces):
        if i == 0:
            own_s = -1e18
        elif hl:
            own_s = s
        else:
            own_s = (pieces[i - 1][1] + s) / 2  # 無音の真ん中
        if i + 1 == len(pieces):
            own_e = 1e18
        elif hr:
            own_e = e
        else:
            own_e = (e + pieces[i + 1][0]) / 2
        a = max(0.0, s - (overlap if hl else pad))
        b = min(total, e + (overlap if hr else pad))
        clips.append(Clip(id=i, start=a, end=b, own_start=own_s, own_end=own_e, speech=_speech_in(segs, a, b), cut=hr))
    return clips


def split_clip(c: Clip, db: Optional[np.ndarray], next_id: int, overlap: float = 0.5, hop: float = HOP) -> List[Clip]:
    """リトライ用: クリップを静かな所で2つに割る(担当区間も割る)"""
    mid_lo = c.start + c.dur * 0.3
    mid_hi = c.start + c.dur * 0.7
    cut = find_cut(db, mid_lo, mid_hi, hop)
    own_cut_lo = max(cut, c.own_start)
    own_cut_hi = min(cut, c.own_end)
    left = Clip(next_id, c.start, min(c.end, cut + overlap), c.own_start, own_cut_hi, 0.0, parent=c.id, cut=True)
    right = Clip(next_id + 1, max(c.start, cut - overlap), c.end, own_cut_lo, c.own_end, 0.0, parent=c.id, cut=c.cut)
    left.speech = c.speech * (left.dur / max(c.dur, 1e-6))
    right.speech = c.speech * (right.dur / max(c.dur, 1e-6))
    return [left, right]


# =====================================================================
# テキスト処理(句読点の付け直し・品質チェック・辞書置換・フィラー・CER)
# =====================================================================

OPENERS = set("「『（(［[｛{〈《【〔“‘«")
_SENT_END_JA = re.compile(r"[。！？!?…‥]+[」』）)\]】〕”’\"']*\s*$")
_PERIOD_END = re.compile(r"[.．][」』）)\]】〕”’\"']*\s+$")
_COMMA_END = re.compile(r"[、，,;；:：][」』）)\]】〕”’\"']*\s*$")
_CJK = r"぀-ヿ㐀-䶿一-鿿豈-﫿　-〿！-｠"
_JA_SPACE = re.compile(rf"(?<=[{_CJK}])[ \t　]+(?=[{_CJK}])")


def is_kept(ch: str) -> bool:
    """アライナー(qwen-asr)と同じ基準: 文字(L*)・数字(N*)・アポストロフィだけ残す"""
    if ch == "'":
        return True
    return unicodedata.category(ch)[:1] in ("L", "N")


def core_len(text: str) -> int:
    return sum(1 for ch in text if is_kept(ch))


def _match_token(text: str, tok: str, pos: int, max_skip: int = 400) -> Optional[Tuple[int, int]]:
    """tok を text[pos:] の中で探す。tok からは記号が抜けていることがあるので
    「残す文字は一致・それ以外は読み飛ばし可」の部分列マッチで対応を取る"""
    tok = "".join(ch for ch in tok if not ch.isspace())
    if not tok:
        return None
    limit = min(len(text), pos + max_skip + len(tok) * 4)
    first = tok[0]
    start = pos
    while True:
        a = text.find(first, start, limit)
        if a < 0:
            return None
        i, j = a, 0
        while i < len(text) and j < len(tok):
            if text[i] == tok[j]:
                i += 1
                j += 1
            elif not is_kept(text[i]):
                i += 1
            else:
                break
        if j == len(tok):
            return a, i
        start = a + 1


def _split_gap(gap: str) -> Tuple[str, str]:
    """語と語のあいだの文字列を「前の語のうしろ(句読点・空白)」と「次の語の頭(開き括弧)」に分ける"""
    for i, ch in enumerate(gap):
        if ch in OPENERS:
            return gap[:i], gap[i:]
    return gap, ""


def restore_display(text: str, tokens: Sequence[Sequence[Any]]) -> List[Dict[str, Any]]:
    """アライナーの語(句読点なし)を元の文章に対応づけて、句読点・記号・空白を語に付け直す。

    tokens: [(語, start, end), ...]   返り値: [{"word", "core", "start", "end"}, ...]
    対応が取れた範囲では "".join(word) が元の文章と一致する(SRT に句読点が戻る)。
    """
    text = text or ""
    toks = [(str(t[0]), float(t[1]), float(t[2])) for t in tokens if str(t[0]).strip()]
    if not toks:
        return []
    spans: List[Optional[Tuple[int, int]]] = []
    pos = 0
    for tok, _s, _e in toks:
        sp = _match_token(text, tok, pos)
        if sp is not None:
            pos = sp[1]
        spans.append(sp)
    idx = [i for i, sp in enumerate(spans) if sp is not None]
    if not idx:
        return [{"word": t, "core": t, "start": s, "end": max(s, e)} for t, s, e in toks]

    out: List[Dict[str, Any]] = []
    lead = text[: spans[idx[0]][0]]
    for k, i in enumerate(idx):
        a, b = spans[i]
        nxt = spans[idx[k + 1]][0] if k + 1 < len(idx) else len(text)
        trail, next_lead = _split_gap(text[b:nxt])
        if k + 1 == len(idx):
            trail, next_lead = text[b:], ""
        tok, s, e = toks[i]
        out.append({"word": lead + text[a:b] + trail, "core": tok, "start": s, "end": max(s, e)})
        lead = next_lead
    return out


def approx_tokens(text: str, start: float, end: float, speech: Sequence[Tuple[float, float]] = ()) -> List[Tuple[str, float, float]]:
    """タイムスタンプが全く無いモデル用の概算。文字数に比例して発話区間に割り当てる"""
    parts = [p for p in re.findall(r"[^、。！？!?,.，．\s]+[、。！？!?,.，．]*\s*", text or "") if p.strip()]
    if not parts:
        return []
    spans = [(max(start, s), min(end, e)) for s, e in speech if min(end, e) > max(start, s)] or [(start, end)]
    total_sp = sum(e - s for s, e in spans)
    weights = [max(1, core_len(p)) for p in parts]
    tot_w = float(sum(weights))

    def at(x: float) -> float:  # 発話時間上の位置 x(0..total_sp) → 実時刻
        for s, e in spans:
            if x <= e - s:
                return s + x
            x -= e - s
        return spans[-1][1]

    out, acc = [], 0.0
    for p, w in zip(parts, weights):
        a = at(acc / tot_w * total_sp)
        acc += w
        b = at(acc / tot_w * total_sp)
        out.append((p, a, max(a, b)))
    return out


def fix_ja_spaces(s: str) -> str:
    """日本語の文字どうしの間の余計な空白を消す(英単語の間の空白は残す)"""
    return _JA_SPACE.sub("", s)


_FW_ALNUM = {c: c - 0xFEE0 for c in list(range(0xFF10, 0xFF1A)) + list(range(0xFF21, 0xFF3B)) + list(range(0xFF41, 0xFF5B))}


def halfwidth_alnum(s: str) -> str:
    """全角英数字だけ半角に(！？（）などの全角記号はそのまま)"""
    return s.translate(_FW_ALNUM)


def parse_terms(s: str) -> List[str]:
    """context 用の語リスト: スペース/読点/カンマ/中黒/スラッシュ/改行区切り(重複除去・順序維持)"""
    seen, out = set(), []
    for w in re.split(r"[\s、，,・･/／;；]+", (s or "").strip()):
        w = w.strip()
        if w and w not in seen:
            seen.add(w)
            out.append(w)
    return out


def build_context(label: str, terms: Sequence[str], extra: str = "") -> str:
    """issue #321 方式:「見出し: A、B、C。」のフレームで語を渡す(英語見出しは , と .)"""
    parts = []
    if terms:
        sep, end = (", ", ".") if (label or "").isascii() else ("、", "。")
        parts.append((f"{label}: " if label else "") + sep.join(terms) + end)
    if extra and extra.strip():
        parts.append(extra.strip())
    return "\n".join(parts)


def compression_ratio(text: str) -> float:
    b = (text or "").encode("utf-8")
    if not b:
        return 0.0
    return len(b) / max(1, len(zlib.compress(b)))


_LOOP = re.compile(r"(.{1,20}?)\1{5,}", re.S)  # 同じ塊が6回以上連続
# 日本語のはずなのに出てきたらおかしい文字(ハングル・キリル・タイ・アラビア・デーヴァナーガリー)
_FOREIGN = {
    "Japanese": re.compile(r"[\uac00-\ud7af\u1100-\u11ff\u3130-\u318f\u0400-\u04ff\u0e00-\u0e7f\u0600-\u06ff\u0900-\u097f]"),
    "English": re.compile(r"[\u3040-\u30ff\uac00-\ud7af\u4e00-\u9fff]"),
}
_PUNCT = re.compile(r"[、。，．,.!?！？]")


def quality_flags(
    text: str,
    dur: float,
    speech: float,
    *,
    ctx_terms: Sequence[str] = (),
    ctx_label: str = "",
    punctuates: bool = True,
    max_cps: float = 18.0,
    lang: str = "",
) -> List[str]:
    """幻聴・ループ・context の復唱などの“怪しさ”を判定してフラグ名のリストを返す"""
    flags: List[str] = []
    t = (text or "").strip()
    if not t:
        if speech >= 2.0:
            flags.append("empty")  # 2秒以上しゃべっているのに空
        return flags
    core = core_len(t)
    if ctx_label and len(ctx_label) >= 2 and ctx_label in t:
        flags.append("context_label")  # 見出しの復唱
    if ctx_terms:
        residual, hits = t, 0
        for w in sorted(ctx_terms, key=len, reverse=True):
            if w and w in residual:
                hits += residual.count(w)
                residual = residual.replace(w, "")
        rest = core_len(residual)
        if hits >= 2 and rest <= max(3, int(core * 0.1)):
            flags.append("context_echo")  # 語を除くと中身がない=復唱してるだけ
    if dur > 0 and core / max(dur, 1.0) > max_cps:
        flags.append("too_dense")  # しゃべれる速さを超えている
    if _LOOP.search(t) or (len(t) >= 80 and compression_ratio(t) > 3.2):
        flags.append("repetition")
    if punctuates and core > 150 and len(_PUNCT.findall(t)) < 3:
        flags.append("no_punct")  # 句読点なしの長文(v1 からの判定)
    rx = _FOREIGN.get(lang)
    if rx is not None and rx.search(t):
        flags.append("foreign_script")  # 日本語指定なのにハングル等が混ざった
    return flags


def collapse_repeats(text: str, keep: int = 2) -> str:
    """6回以上つづく繰り返しを keep 回に縮める"""
    return _LOOP.sub(lambda m: m.group(1) * keep, text or "")


def parse_replacements(spec: str) -> List[Tuple[Any, str]]:
    """置換辞書。1行1ルール: 「誤 => 正」「誤→正」「誤<TAB>正」。re: で始めると正規表現"""
    rules: List[Tuple[Any, str]] = []
    for line in (spec or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^(.*?)\s*(?:=>|→|⇒|\t)\s*(.*)$", line)
        if not m:
            continue
        src, dst = m.group(1), m.group(2)
        if not src:
            continue
        if src.startswith("re:"):
            try:
                rules.append((re.compile(src[3:]), dst))
            except re.error:
                continue
        else:
            rules.append((src, dst))
    return rules


def apply_replacements(s: str, rules: Sequence[Tuple[Any, str]]) -> str:
    for src, dst in rules:
        s = src.sub(dst, s) if hasattr(src, "sub") else s.replace(src, dst)
    return s


FILLERS_SAFE = [
    "えーっと", "えーと", "えっと", "えーー", "えー", "ええと", "あのー", "あのう", "あー", "うーん", "うーむ",
    "んー", "そのー", "まー", "えぇ",
]
FILLERS_MORE = ["あの", "その", "まあ", "なんか", "ええ", "うん", "はい"]


def make_filler_regex(words: Sequence[str]) -> Optional["re.Pattern[str]"]:
    words = sorted({w for w in words if w}, key=len, reverse=True)
    if not words:
        return None
    alt = "|".join(re.escape(w) + "ー*" for w in words)
    # 文頭/句読点/空白の直後にあって、うしろに読点/空白/句点/文末が続くときだけ消す
    return re.compile(rf"(?:(?<=^)|(?<=[、。，．,.!?！？\s「『]))(?:{alt})(?:[、，,]\s*|\s+|(?=[。．.!?！？」』]|$))")


def remove_fillers(s: str, rx: Optional["re.Pattern[str]"]) -> str:
    if rx is None:
        return s
    prev = None
    while prev != s:  # 「えー、あのー、」のような連続も消す
        prev = s
        s = rx.sub("", s)
    s = re.sub(r"^[、，,\s]+", "", s)
    s = re.sub(r"[、，,]\s*([。．.!?！？])", r"\1", s)
    return s


def normalize_for_cer(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "").lower()
    return "".join(ch for ch in s if is_kept(ch) and ch != "'")


def edit_distance(a: str, b: str) -> int:
    try:
        from rapidfuzz.distance import Levenshtein  # type: ignore

        return int(Levenshtein.distance(a, b))
    except Exception:
        pass
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1]


_NUM = re.compile(r"[0-9０-９]+(?:[.,．，][0-9０-９]+)?|[〇一二三四五六七八九十百千万億兆]+")


def term_recall(ref: str, hyp: str, terms: Sequence[str]) -> Tuple[int, int]:
    """正解文に出てくる用語のうち、認識結果にも出てきた数 (hit, total)"""
    r, h = unicodedata.normalize("NFKC", ref or ""), unicodedata.normalize("NFKC", hyp or "")
    tot = hit = 0
    for t in terms:
        t = unicodedata.normalize("NFKC", t)
        n = r.count(t)
        if n:
            tot += n
            hit += min(n, h.count(t))
    return hit, tot


def number_recall(ref: str, hyp: str) -> Tuple[int, int]:
    """正解文の数字(算用数字・漢数字)が認識結果にも同じ形で出てきた数 (hit, total)。金額や日付の取り違えの目安"""
    rn = Counter(_NUM.findall(unicodedata.normalize("NFKC", ref or "")))
    hn = Counter(_NUM.findall(unicodedata.normalize("NFKC", hyp or "")))
    tot = sum(rn.values())
    hit = sum(min(c, hn.get(k, 0)) for k, c in rn.items())
    return hit, tot


def cer(ref: str, hyp: str) -> float:
    r, h = normalize_for_cer(ref), normalize_for_cer(hyp)
    if not r:
        return float("nan") if h else 0.0
    return edit_distance(r, h) / len(r)


# =====================================================================
# ASR をクリップに流す(リトライつき)。エンジン非依存
# =====================================================================


@dataclass
class RetryPolicy:
    enabled: bool = True
    without_context: bool = True  # 1) context を外して再推論
    split: bool = True  # 2) それでもダメなら静かな所で2分割して再推論
    split_min_sec: float = 8.0
    punctuates: bool = True
    max_cps: float = 18.0
    lang: str = ""  # 指定言語(文字種チェック用)


@dataclass
class ClipResult:
    clip: Clip
    text: str = ""
    language: str = ""
    flags: List[str] = field(default_factory=list)
    attempts: List[Dict[str, Any]] = field(default_factory=list)
    words: Optional[List[List[Any]]] = None  # エンジン固有のタイムスタンプ [(語, s, e)] (クリップ先頭基準)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "clip": self.clip.to_dict(),
            "text": self.text,
            "language": self.language,
            "flags": list(self.flags),
            "attempts": self.attempts,
            "words": self.words,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ClipResult":
        return ClipResult(
            clip=Clip.from_dict(d["clip"]),
            text=d.get("text") or "",
            language=d.get("language") or "",
            flags=list(d.get("flags") or []),
            attempts=list(d.get("attempts") or []),
            words=d.get("words"),
        )


TranscribeFn = Callable[[List[Tuple[np.ndarray, str]]], List[Dict[str, Any]]]


def run_asr(
    clips: Sequence[Clip],
    get_audio: Callable[[float, float], np.ndarray],
    transcribe: TranscribeFn,
    *,
    context: str = "",
    ctx_terms: Sequence[str] = (),
    ctx_label: str = "",
    policy: Optional[RetryPolicy] = None,
    batch_size: int = 8,
    db: Optional[np.ndarray] = None,
    on_progress: Optional[Callable[[str, int, int, List[ClipResult]], None]] = None,
    done: Optional[Dict[int, ClipResult]] = None,
) -> List[ClipResult]:
    """クリップ群を ASR にかける。

    pass1: 長い順に batch_size ずつ(パディングの無駄が減る & OOM は最初に出る)
    pass2: 怪しいクリップを context なしで再推論
    pass3: まだ怪しいクリップを2分割して再推論
    最後まで怪しいものは一番マシな結果を採用して flags を残す(JSON で確認できる)
    done: 途中再開用。pass1 が済んでいるクリップの結果(clip.id → 結果)。pass1 を飛ばす
    """
    policy = policy or RetryPolicy()
    clips = list(clips)
    if not clips:
        return []
    done = dict(done or {})

    def flags_of(text: str, c: Clip) -> List[str]:
        return quality_flags(
            text, c.dur, c.speech, ctx_terms=ctx_terms, ctx_label=ctx_label,
            punctuates=policy.punctuates, max_cps=policy.max_cps, lang=policy.lang,
        )

    def run_batch(items: List[Tuple[Clip, str]]) -> List[Dict[str, Any]]:
        auds = [(get_audio(c.start, c.end), ctx) for c, ctx in items]
        outs = transcribe(auds)
        if len(outs) != len(items):
            raise RuntimeError(f"エンジンの返り値の数が合いません: {len(outs)} != {len(items)}")
        return outs

    results: Dict[int, ClipResult] = {}
    for c in clips:
        if c.id in done:
            r = done[c.id]
            r.clip = c
            r.flags = flags_of(r.text, c)  # 判定基準が変わっていても大丈夫なように付け直す
            results[c.id] = r
    order = sorted((c for c in clips if c.id not in results), key=lambda c: c.dur, reverse=True)
    bs = max(1, int(batch_size))
    n_done = len(results)
    for i in range(0, len(order), bs):
        batch = order[i: i + bs]
        outs = run_batch([(c, context) for c in batch])
        part = []
        for c, o in zip(batch, outs):
            t = (o.get("text") or "").strip()
            f = flags_of(t, c)
            r = ClipResult(c, t, o.get("language") or "", f, [{"ctx": bool(context), "text": t, "flags": f}], o.get("words"))
            results[c.id] = r
            part.append(r)
        n_done += len(batch)
        if on_progress:
            on_progress("asr", n_done, len(clips), part)

    if not policy.enabled:
        return sorted(results.values(), key=lambda r: r.clip.start)

    # pass2: context を外す
    bad = [r for r in results.values() if r.flags]
    if bad and context and policy.without_context:
        for i in range(0, len(bad), bs):
            chunk = bad[i: i + bs]
            outs = run_batch([(r.clip, "") for r in chunk])
            for r, o in zip(chunk, outs):
                t = (o.get("text") or "").strip()
                f = flags_of(t, r.clip)
                r.attempts.append({"ctx": False, "text": t, "flags": f})
                if len(f) < len(r.flags) or not f:
                    r.text, r.flags, r.words = t, f, o.get("words")
                    r.language = o.get("language") or r.language
            if on_progress:
                on_progress("retry", min(i + bs, len(bad)), len(bad), chunk)

    # pass3: 2分割
    bad = [r for r in results.values() if r.flags and policy.split and r.clip.dur >= policy.split_min_sec]
    if bad:
        next_id = max(c.id for c in clips) + 1
        subs: List[Tuple[ClipResult, List[Clip]]] = []
        for r in bad:
            sc = split_clip(r.clip, db, next_id)
            next_id += 2
            subs.append((r, sc))
        # 分割後は、より良かった方の context 設定を使う
        items: List[Tuple[Clip, str]] = []
        for r, sc in subs:
            use_ctx = context if (r.attempts and r.attempts[0]["ctx"] and not r.attempts[0]["flags"]) else ""
            for c in sc:
                items.append((c, use_ctx))
        outs_all: List[Dict[str, Any]] = []
        for i in range(0, len(items), bs):
            outs_all.extend(run_batch(items[i: i + bs]))
        k = 0
        for r, sc in subs:
            sub_results = []
            for c in sc:
                o = outs_all[k]
                k += 1
                t = (o.get("text") or "").strip()
                sub_results.append(ClipResult(c, t, o.get("language") or "", flags_of(t, c), [{"ctx": bool(items[k - 1][1]), "text": t, "flags": flags_of(t, c)}], o.get("words")))
            n_bad_sub = sum(len(s.flags) for s in sub_results)
            if n_bad_sub < len(r.flags):
                del results[r.clip.id]
                for s in sub_results:
                    s.attempts = r.attempts + [{"split_from": r.clip.id}] + s.attempts
                    results[s.clip.id] = s
            else:
                r.attempts.append({"split": [s.text for s in sub_results], "flags": [s.flags for s in sub_results]})
        if on_progress:
            on_progress("split", len(subs), len(subs), [])

    # 最後まで残ったループは縮める
    for r in results.values():
        if "repetition" in r.flags:
            fixed = collapse_repeats(r.text)
            if fixed != r.text:
                r.attempts.append({"collapsed": True})
                r.text = fixed
                r.words = None  # テキストが変わったのでエンジン側タイムスタンプは使わない
    return sorted(results.values(), key=lambda r: (r.clip.start, r.clip.id))


# =====================================================================
# 単語列の組み立て(句読点復元・重複除去)
# =====================================================================


@dataclass
class Word:
    word: str  # 表示用(句読点・空白込み)
    start: float
    end: float
    core: str = ""
    speaker: Optional[str] = None
    clip: int = -1

    def to_dict(self) -> Dict[str, Any]:
        d = {"word": self.word, "start": round(self.start, 3), "end": round(self.end, 3)}
        if self.speaker is not None:
            d["speaker"] = self.speaker
        return d


def compose_words(
    results: Sequence[ClipResult],
    aligned: Dict[int, List[List[Any]]],
    speech_segs: Sequence[Tuple[float, float]] = (),
) -> Tuple[List[Word], Dict[str, int]]:
    """クリップごとのテキスト + タイムスタンプ → 全体の単語列。

    タイムスタンプの優先順位: アライナー > エンジン固有 > 文字数からの概算
    担当区間(ハード切りのオーバーラップ)の外に中点がある単語は捨てる。
    """
    words: List[Word] = []
    stats: Counter = Counter()
    for r in results:
        c = r.clip
        if not r.text:
            continue
        toks = aligned.get(c.id)
        src = "aligner"
        if toks:
            abs_toks = [(t[0], c.start + float(t[1]), c.start + float(t[2])) for t in toks]
        elif r.words:
            abs_toks = [(t[0], c.start + float(t[1]), c.start + float(t[2])) for t in r.words]
            src = "engine"
        else:
            abs_toks = approx_tokens(r.text, c.start, c.end, speech_segs)
            src = "approx"
        stats[src] += 1
        disp = restore_display(r.text, abs_toks)
        if not disp:  # タイムスタンプが全く取れなかった → 本文を落とさないよう概算で入れる
            disp = restore_display(r.text, approx_tokens(r.text, c.start, c.end, speech_segs))
            stats["rescued"] += 1
        kept = 0
        for d in disp:
            s = min(max(d["start"], c.start), c.end)
            e = min(max(d["end"], s), c.end)
            if e - s > 4.0:
                stats["long_words"] += 1  # 不自然に長い単語(アライナーの失敗の目安)
            mid = (s + e) / 2
            if c.own_start <= mid < c.own_end:
                words.append(Word(d["word"], s, e, d["core"], None, c.id))
                kept += core_len(d["word"])
        total_chars = core_len(r.text)
        # 余白の重なりで削れるのは普通だが、半分以上消えたらアライメントの失敗を疑う
        if total_chars >= 10 and kept < total_chars * 0.5:
            stats["trimmed_much"] += 1
            if "trimmed" not in r.flags:
                r.flags.append("trimmed")
    words.sort(key=lambda w: (w.start, w.end))
    # 時刻の逆転をならす(アライナーの誤差対策)
    for i in range(1, len(words)):
        if words[i].start < words[i - 1].start:
            words[i].start = words[i - 1].start
        if words[i].end < words[i].start:
            words[i].end = words[i].start
    return words, dict(stats)


# =====================================================================
# 話者分離の結果を単語に割り当てる
# =====================================================================


def assign_speakers(words: Sequence[Word], turns: Sequence[Sequence[Any]], max_dist: float = 1.0) -> None:
    """各単語に、単語と重なっている時間が最も長い話者を付ける(重ならなければ max_dist 秒以内の最寄り)。

    話者区間どうしが重なっていても(A:0〜10秒 の中に B:1〜2秒 がある等)取りこぼさないよう、
    「いちばん長い区間の長さ」ぶん手前から候補を集める(v2 は開始時刻の近い2区間しか見ておらず誤判定があった)。
    """
    turns = sorted(((float(s), float(e), str(k)) for s, e, k in turns if float(e) > float(s)), key=lambda x: x[0])
    if not turns:
        return
    starts = [t[0] for t in turns]
    max_len = max(e - s for s, e, _ in turns)
    for w in words:
        lo = bisect.bisect_left(starts, w.start - max_len - max_dist)
        hi = bisect.bisect_right(starts, w.end + max_dist)
        best, best_ov = None, 0.0
        near, near_d = None, float("inf")
        mid = (w.start + w.end) / 2
        for j in range(lo, hi):
            s, e, k = turns[j]
            ov = min(e, w.end) - max(s, w.start)
            if ov > best_ov:
                best, best_ov = k, ov
            d = 0.0 if s <= mid <= e else min(abs(mid - s), abs(mid - e))
            if d < near_d:
                near, near_d = k, d
        if best is None and near is not None and near_d <= max_dist:
            best = near  # 長さ0の単語や無音部分 → 最寄りの話者
        w.speaker = best
    # 割り当てられなかった単語は前後の話者で埋める
    last = None
    for w in words:
        if w.speaker is None:
            w.speaker = last
        else:
            last = w.speaker
    nxt = None
    for w in reversed(words):
        if w.speaker is None:
            w.speaker = nxt
        else:
            nxt = w.speaker


def overlap_regions(turns: Sequence[Sequence[Any]], min_dur: float = 0.2) -> List[Tuple[float, float, List[str]]]:
    """(重なりありの)話者区間から、2人以上が同時に話している時間帯を出す"""
    ev: List[Tuple[float, int, str]] = []
    for s, e, k in turns:
        if float(e) > float(s):
            ev.append((float(s), 1, str(k)))
            ev.append((float(e), -1, str(k)))
    ev.sort(key=lambda x: (x[0], x[1]))
    active: Counter = Counter()
    out: List[Tuple[float, float, List[str]]] = []
    start: Optional[float] = None
    for t, d, k in ev:
        before = sum(1 for v in active.values() if v > 0)
        active[k] += d
        after = sum(1 for v in active.values() if v > 0)
        if before < 2 <= after:
            start = t
        elif before >= 2 > after and start is not None:
            if t - start >= min_dur:
                out.append((start, t, sorted(x for x, v in active.items() if v > 0) or []))
            start = None
    # 話者名は「重なっていた人たち」を入れ直す
    res = []
    for s, e, _ in out:
        who = sorted({str(k) for ts, te, k in turns if min(float(te), e) - max(float(ts), s) > 0})
        res.append((round(s, 3), round(e, 3), who))
    return res


def smooth_speakers(words: Sequence[Word], min_share: float = 0.6, max_pause: float = 1.0) -> int:
    """文単位の多数決で「文の途中で話者がチラつく」のを直す。直した単語数を返す"""
    changed = 0
    for sent in split_sentences(words, max_pause=max_pause, split_on_speaker=False):
        dur: Counter = Counter()
        for w in sent:
            dur[w.speaker] += max(0.02, w.end - w.start)
        if not dur:
            continue
        spk, d = dur.most_common(1)[0]
        if spk is not None and d / sum(dur.values()) >= min_share:
            for w in sent:
                if w.speaker != spk:
                    w.speaker = spk
                    changed += 1
    # 1〜2語だけの割り込み(前後が同じ話者)も吸収
    ws = list(words)
    i = 0
    while i < len(ws):
        j = i
        while j < len(ws) and ws[j].speaker == ws[i].speaker:
            j += 1
        if 0 < i and j < len(ws) and (j - i) <= 2 and ws[i - 1].speaker == ws[j].speaker != ws[i].speaker:
            if ws[j - 1].end - ws[i].start < 0.8:
                for k in range(i, j):
                    ws[k].speaker = ws[i - 1].speaker
                    changed += 1
        i = j
    return changed


def speaker_names(words: Sequence[Word], style: str = "話者A", mapping_spec: str = "") -> Dict[str, str]:
    """登場順に 話者A, 話者B… (style) の名前を付け、mapping_spec で上書きする。

    mapping_spec: 「SPEAKER_00=田中, SPEAKER_01=佐藤」または「話者A=田中」または「田中, 佐藤」(登場順)
    """
    order: List[str] = []
    for w in words:
        if w.speaker is not None and w.speaker not in order:
            order.append(w.speaker)
    names: Dict[str, str] = {}
    for i, spk in enumerate(order):
        if style == "SPEAKER_00":
            names[spk] = spk
        elif style == "S1":
            names[spk] = f"S{i + 1}"
        else:
            names[spk] = "話者" + (chr(ord("A") + i) if i < 26 else str(i + 1))
    spec = (mapping_spec or "").strip()
    if spec:
        items = [x.strip() for x in re.split(r"[,、，\n]+", spec) if x.strip()]
        if all(("=" in x or "＝" in x) for x in items):
            for x in items:
                k, v = re.split(r"[=＝]", x, maxsplit=1)
                k, v = k.strip(), v.strip()
                for spk, nm in list(names.items()):
                    if k in (spk, nm):
                        names[spk] = v
        else:
            for spk, v in zip(order, items):
                names[spk] = v
    return names


# =====================================================================
# 文・段落・字幕の組み立て
# =====================================================================


def _ends_sentence(w: Word, nxt: Optional[Word]) -> bool:
    t = w.word
    if _SENT_END_JA.search(t):
        return True
    if _PERIOD_END.search(t):  # 英語のピリオドは後ろに空白があるときだけ(3.5 などを割らない)
        return True
    if nxt is None and re.search(r"[.．][」』）)\]】〕”’\"']*$", t.rstrip()):
        return True
    return False


def split_sentences(
    words: Sequence[Word], max_pause: float = 1.5, max_chars: int = 150, split_on_speaker: bool = True
) -> List[List[Word]]:
    sents: List[List[Word]] = []
    cur: List[Word] = []
    n = len(words)
    for i, w in enumerate(words):
        if cur and ((w.start - cur[-1].end > max_pause) or (split_on_speaker and w.speaker != cur[-1].speaker)):
            sents.append(cur)
            cur = []
        cur.append(w)
        nxt = words[i + 1] if i + 1 < n else None
        if _ends_sentence(w, nxt) or sum(len(x.word) for x in cur) >= max_chars:
            sents.append(cur)
            cur = []
    if cur:
        sents.append(cur)
    return sents


@dataclass
class TextPost:
    """出力テキストに最後にかける整形"""

    replacements: List[Tuple[Any, str]] = field(default_factory=list)
    filler_rx: Optional[Any] = None
    fix_spaces: bool = True
    halfwidth: bool = True

    def __call__(self, s: str) -> str:
        if self.fix_spaces:
            s = fix_ja_spaces(s)
        if self.halfwidth:
            s = halfwidth_alnum(s)
        if self.replacements:
            s = apply_replacements(s, self.replacements)
        if self.filler_rx is not None:
            s = remove_fillers(s, self.filler_rx)
        return s.strip()


def join_words(ws: Sequence[Word]) -> str:
    return "".join(w.word for w in ws).strip()


@dataclass
class Segment:
    start: float
    end: float
    text: str
    speaker: Optional[str] = None
    words: List[Word] = field(default_factory=list)


def make_sentences(words: Sequence[Word], post: TextPost, max_pause: float = 1.5) -> List[Segment]:
    out = []
    for s in split_sentences(words, max_pause=max_pause):
        txt = post(join_words(s))
        if not txt:
            continue
        out.append(Segment(s[0].start, s[-1].end, txt, s[0].speaker, list(s)))
    return out


def make_paragraphs(sents: Sequence[Segment], para_gap: float = 3.0, max_chars: int = 400) -> List[Segment]:
    paras: List[Segment] = []
    for s in sents:
        p = paras[-1] if paras else None
        if p and p.speaker == s.speaker and s.start - p.end <= para_gap and len(p.text) + len(s.text) <= max_chars:
            p.text += s.text if (p.text[-1:] in "。！？!?」』" or not p.text[-1:].isascii()) else " " + s.text
            p.end = s.end
            p.words.extend(s.words)
        else:
            paras.append(Segment(s.start, s.end, s.text, s.speaker, list(s.words)))
    return paras


@dataclass
class CueRules:
    max_chars: int = 30
    max_dur: float = 6.0
    gap: float = 0.8
    min_dur: float = 0.6
    tail_pad: float = 0.2
    punct_split_min: int = 12  # 句点で区切るのはこの文字数以上たまってから


def make_cues(words: Sequence[Word], post: TextPost, rules: Optional[CueRules] = None) -> List[Segment]:
    """単語列 → 字幕(SRT/VTT)のキュー。話者交代・無音・長さ・文字数・句読点で区切る"""
    rules = rules or CueRules()
    cues: List[Segment] = []
    cur: List[Word] = []

    def length(ws: Sequence[Word]) -> int:
        return len(join_words(ws))

    def flush() -> None:
        nonlocal cur
        if cur:
            txt = post(join_words(cur))
            if txt:
                cues.append(Segment(cur[0].start, cur[-1].end, txt, cur[0].speaker, list(cur)))
        cur = []

    n = len(words)
    # 各単語から「その文の終わり」までの文字数(読点で割るかどうかの先読み用)
    rest_len = [0] * n
    acc = 0
    for i in range(n - 1, -1, -1):
        nxt = words[i + 1] if i + 1 < n else None
        if _ends_sentence(words[i], nxt) or nxt is None or nxt.start - words[i].end >= rules.gap:
            acc = 0
        rest_len[i] = acc
        acc += len(words[i].word)

    def flush_at_break() -> None:
        """長さ/時間の上限で切るときは、なるべく直前の句読点で切って残りを次へ回す(「す。」だけの字幕を防ぐ)"""
        nonlocal cur
        k = None
        for j in range(len(cur) - 1, max(0, len(cur) // 3) - 1, -1):
            nx = cur[j + 1] if j + 1 < len(cur) else None
            if _ends_sentence(cur[j], nx) or _COMMA_END.search(cur[j].word):
                k = j
                break
        if k is None or k == len(cur) - 1:
            flush()
            return
        rest = cur[k + 1:]
        cur = cur[: k + 1]
        flush()
        cur = rest

    for i, w in enumerate(words):
        if cur:
            if w.start - cur[-1].end >= rules.gap or w.speaker != cur[0].speaker:
                flush()
            elif w.end - cur[0].start > rules.max_dur or length(cur) + len(w.word.strip()) > rules.max_chars:
                flush_at_break()
                # 残りを足してもまだ上限を超えるなら、そこで切る
                if cur and (w.end - cur[0].start > rules.max_dur or length(cur) + len(w.word.strip()) > rules.max_chars):
                    flush()
        cur.append(w)
        nxt = words[i + 1] if i + 1 < n else None
        L = length(cur)
        if _ends_sentence(w, nxt) and L >= rules.punct_split_min:
            flush()
        elif _COMMA_END.search(w.word) and L + rest_len[i] > rules.max_chars and L >= max(4, int(rules.max_chars * 0.35)):
            flush()  # 文の残りが入りきらないときだけ読点で割る
    flush()

    # タイミングの後処理: 最短表示時間・少し余韻・次のキューと重ねない
    for i, c in enumerate(cues):
        nxt_start = cues[i + 1].start if i + 1 < len(cues) else float("inf")
        end = max(c.end + rules.tail_pad, c.start + rules.min_dur)
        end = min(end, nxt_start - 0.001) if nxt_start < float("inf") else end
        c.end = max(end, c.start + 0.05)
    return cues


# =====================================================================
# 書き出し
# =====================================================================


def _label(seg: Segment, names: Dict[str, str]) -> str:
    return names.get(seg.speaker, seg.speaker or "") if seg.speaker is not None else ""


def to_srt(cues: Sequence[Segment], names: Dict[str, str], speaker_fmt: str = "{speaker}: {text}") -> str:
    out = []
    for i, c in enumerate(cues, 1):
        spk = _label(c, names)
        body = speaker_fmt.format(speaker=spk, text=c.text) if spk else c.text
        out.append(f"{i}\n{fmt_srt_time(c.start)} --> {fmt_srt_time(c.end)}\n{body}\n")
    return "\n".join(out)


def to_vtt(cues: Sequence[Segment], names: Dict[str, str]) -> str:
    out = ["WEBVTT", ""]
    for c in cues:
        spk = _label(c, names)
        body = f"<v {spk}>{c.text}" if spk else c.text
        out.append(f"{fmt_vtt_time(c.start)} --> {fmt_vtt_time(c.end)}\n{body}\n")
    return "\n".join(out)


def to_txt(paras: Sequence[Segment], names: Dict[str, str], timestamps: bool = True) -> str:
    lines = []
    for p in paras:
        spk = _label(p, names)
        head = f"[{fmt_hms(p.start)}] " if timestamps else ""
        lines.append(f"{head}{spk + ': ' if spk else ''}{p.text}")
    return "\n".join(lines) + ("\n" if lines else "")


def to_markdown(paras: Sequence[Segment], names: Dict[str, str], title: str, meta: Dict[str, Any]) -> str:
    lines = [f"# {title}", ""]
    for k, v in meta.items():
        lines.append(f"- {k}: {v}")
    lines += ["", "## 文字起こし", ""]
    for p in paras:
        spk = _label(p, names)
        who = f" **{spk}**" if spk else ""
        lines.append(f"**[{fmt_hms(p.start)}]**{who}  ")
        lines.append(p.text)
        lines.append("")
    return "\n".join(lines)


def to_csv(sents: Sequence[Segment], names: Dict[str, str]) -> str:
    bio = io.StringIO()
    w = csv.writer(bio, lineterminator="\n")
    w.writerow(["start", "end", "start_hms", "speaker", "text"])
    for s in sents:
        w.writerow([f"{s.start:.3f}", f"{s.end:.3f}", fmt_hms(s.start), _label(s, names), s.text])
    return bio.getvalue()


def to_rttm(turns: Sequence[Sequence[Any]], file_id: str, names: Optional[Dict[str, str]] = None) -> str:
    fid = re.sub(r"\s+", "_", file_id) or "audio"
    lines = []
    for s, e, k in sorted(turns, key=lambda x: float(x[0])):
        spk = (names or {}).get(k, k)
        spk = re.sub(r"\s+", "_", str(spk))
        lines.append(f"SPEAKER {fid} 1 {float(s):.3f} {float(e) - float(s):.3f} <NA> <NA> {spk} <NA> <NA>")
    return "\n".join(lines) + ("\n" if lines else "")


def to_json(
    sents: Sequence[Segment],
    names: Dict[str, str],
    results: Sequence[ClipResult],
    meta: Dict[str, Any],
) -> str:
    segs = []
    for i, s in enumerate(sents):
        d: Dict[str, Any] = {"id": i, "start": round(s.start, 3), "end": round(s.end, 3), "text": s.text}
        if s.speaker is not None:
            d["speaker"] = _label(s, names)
            d["speaker_id"] = s.speaker
        d["words"] = [w.to_dict() for w in s.words]
        segs.append(d)
    clips = []
    for r in results:
        d = r.clip.to_dict()
        d.update({"text": r.text, "flags": r.flags})
        if len(r.attempts) > 1 or r.flags:
            d["attempts"] = r.attempts
        clips.append(d)
    doc = dict(meta)
    doc["speakers"] = names
    doc["segments"] = segs
    doc["clips"] = clips
    return json.dumps(doc, ensure_ascii=False, indent=1)


FLAG_JA = {
    "empty": "発話があるのに文字が出なかった",
    "context_label": "context の見出しをそのまま出力(復唱)",
    "context_echo": "用語リストをくり返しているだけに見える(復唱)",
    "too_dense": "しゃべれる速さを超える文字数(幻聴?)",
    "repetition": "同じ言葉のループ",
    "no_punct": "句読点のない長文",
    "trimmed": "タイムスタンプ付けで大きく欠けた(アライメント失敗?)",
    "foreign_script": "指定した言語にない文字(ハングル等)が混ざった",
}


def review_items(results: Sequence[ClipResult]) -> List[ClipResult]:
    """要確認リストに載せるクリップ: 最後まで怪しいもの + 自動で差し替えたもの"""
    out = []
    for r in results:
        changed = any(("split_from" in a) or ("adopted" in a) for a in r.attempts) or (
            len([a for a in r.attempts if "text" in a]) > 1 and r.attempts[0].get("text") != r.text)
        if r.flags or changed:
            out.append(r)
    return out


def _attempt_line(a: Dict[str, Any]) -> str:
    src = f"別モデル {a['model']}" if "model" in a else ("context あり" if a.get("ctx") else "context なし")
    fl = a.get("flags") or []
    mark = f"  ⚠ {'、'.join(FLAG_JA.get(f, f) for f in fl)}" if fl else ""
    return f"  - 候補({src}): {a['text'][:400] or '(空)'}{mark}"


def to_review_md(results: Sequence[ClipResult], title: str) -> str:
    items = review_items(results)
    lines = [f"# 要確認リスト: {title}", "",
             "自動チェックで怪しいと判定された区間と、自動で再推論して差し替えた区間です。"
             "元の結果も残してあるので、音声を聞いて確かめてください。", ""]
    # 2分割で置き換わったクリップは、元のクリップごとにまとめる
    groups: List[List[ClipResult]] = []
    for r in items:
        if r.clip.parent is not None and groups and groups[-1][0].clip.parent == r.clip.parent:
            groups[-1].append(r)
        else:
            groups.append([r])
    for g in groups:
        first, last = g[0], g[-1]
        flags = [f for r in g for f in r.flags]
        why = "、".join(dict.fromkeys(FLAG_JA.get(f, f) for f in flags)) if flags else "自動で差し替え済み"
        split = first.clip.parent is not None
        lines.append(f"## [{fmt_hms(first.clip.start)} 〜 {fmt_hms(last.clip.end)}] {why}"
                     + (f"(静かな所で {len(g)} つに分けて読み直し)" if split and len(g) > 1 else ""))
        lines.append(f"- **採用した結果**: {''.join(r.text for r in g) or '(空)'}")
        shown = set()
        for r in g:
            for a in r.attempts:
                if "text" not in a:
                    continue
                key = (a.get("model"), a.get("ctx"), a["text"])
                if key in shown:
                    continue
                shown.add(key)
                lines.append(_attempt_line(a))
        lines.append("")
    return "\n".join(lines)


def diff_html(a: str, b: str, max_len: int = 20000) -> str:
    """2つのテキストの差分を HTML で(削除=赤, 追加=緑)。モデル比較用"""
    import difflib
    import html

    a, b = a[:max_len], b[:max_len]
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    out = []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            out.append(html.escape(a[i1:i2]))
        if op in ("delete", "replace"):
            out.append(f"<del style='background:#fdd;text-decoration:line-through'>{html.escape(a[i1:i2])}</del>")
        if op in ("insert", "replace"):
            out.append(f"<ins style='background:#dfd;text-decoration:none'>{html.escape(b[j1:j2])}</ins>")
    return "".join(out)
