# -*- coding: utf-8 -*-
"""設定まわり(クリップ長の「自動」)・比較表・Granite のバッチ分けの確認(GPU・ffmpeg なしで動く)"""
import os
import sys

import numpy as np

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))

from asrkit import engines, pipeline, presets  # noqa: E402


def test_auto_num():
    assert pipeline.auto_num("自動") is None
    assert pipeline.auto_num("") is None
    assert pipeline.auto_num(None) is None
    assert pipeline.auto_num("auto") is None
    assert pipeline.auto_num("15") == 15.0
    assert pipeline.auto_num(" 1.5 ") == 1.5
    assert pipeline.auto_num(20) == 20.0
    assert pipeline.auto_num("15秒") == 15.0 and pipeline.auto_num("2s") == 2.0
    assert pipeline.auto_num("0") == 0.0  # 無音 0 秒(どんな短い無音でも区切る)は「自動」ではない


def test_clip_settings_uses_preset_recommendation():
    kotoba = presets.find_preset("kotoba-whisper-v2")
    qwen = presets.find_preset("qwen3-1.7b")
    st = pipeline.Settings()
    assert st.max_clip is None and st.max_gap is None  # 既定は「自動」
    a = pipeline.clip_settings(st, kotoba)
    assert a.max_clip == kotoba.max_clip and a.max_clip < pipeline.DEFAULT_MAX_CLIP
    assert a.max_gap == (kotoba.max_gap if kotoba.max_gap is not None else pipeline.DEFAULT_MAX_GAP)
    b = pipeline.clip_settings(st, qwen)
    assert (b.max_clip, b.max_gap) == (pipeline.DEFAULT_MAX_CLIP, pipeline.DEFAULT_MAX_GAP)
    # 手で決めた値はモデルに関係なくそのまま(無音 0 秒 = どんなに短い無音でも区切る、も有効)
    c = pipeline.clip_settings(st.replace(max_clip=45, max_gap=0), kotoba)
    assert (c.max_clip, c.max_gap) == (45.0, 0.0)
    # 元の Settings は書き換えない
    assert st.max_clip is None and st.max_gap is None


def test_presets_clip_values_are_sane():
    for p in presets.PRESETS:
        if p.max_clip is not None:
            assert 5 <= p.max_clip <= 170, p.key  # アライナーの上限(180 秒)より短く
        if p.max_gap is not None:
            assert 0 <= p.max_gap <= 30, p.key


def test_compare_table_shows_clip_and_load():
    rows = [
        {"preset": "a", "label": "A", "text": "こんにちは", "asr_sec": 2.0, "x_realtime": 150.0, "load_sec": 12.3,
         "gpu_peak_gb": 5.1, "flags": 0, "cer": 0.08, "numbers": (6, 6), "terms": None,
         "negations": (3, 4, 1), "drops": (2, 37), "clips": {"n": 11, "max_clip": 15.0, "max_gap": 6.0}},
        {"preset": "b", "label": "B", "text": "x", "asr_sec": 1.0, "x_realtime": 300.0, "flags": 1, "cer": 0.1},
        {"preset": "c", "label": "C", "error": "boom"},
    ]
    h = pipeline.compare_table_html(rows, True)
    assert "上限 15秒" in h and "無音 6秒" in h and "12秒" in h and "boom" in h
    assert "3/4 (+1)" in h and "2 か所" in h and "37 字" in h


def test_granite_batches_by_prompt_and_keeps_order():
    eng = object.__new__(engines.GraniteSpeechEngine)  # 重みは読まずに、振り分けだけ確かめる
    eng.batch_size = 2
    calls = []

    def fake_run(chunk):
        calls.append([c for _, c in chunk])
        return [{"text": f"t{int(a[0])}"} for a, _ in chunk]

    eng._run = fake_run
    items = [(np.full(10, i, np.float32), "ctx" if i % 2 else "") for i in range(5)]
    out = eng.transcribe(items, "Japanese")
    assert [o["text"] for o in out] == [f"t{i}" for i in range(5)]  # 順番は元どおり
    assert all(o["language"] == "Japanese" for o in out)
    # 1 回の呼び出しの中はプロンプト(キーワードあり/なし)がそろっていて、バッチは 2 件まで
    assert all(len({bool(c) for c in call}) == 1 and len(call) <= 2 for call in calls)
    assert sum(len(c) for c in calls) == 5


def test_timing_breakdown_adds_up():
    """合計の内訳: 読み込みは文字起こしから分けて出し、足すと合計になる(使い回したときは読み込み 0)"""
    import re

    T = {"audio": 2.8, "vad": 14.2, "asr": 37.1, "align": 10.1, "diar_wait": 0.0, "total": 69.0}
    s = pipeline.timing_breakdown(T, {"load_sec": 34.0, "asr_sec": 2.7})
    assert "モデルの読み込み待ち 34.0秒" in s and "文字起こし 3.1秒" in s and "話者分離の待ち" not in s
    assert abs(sum(float(x) for x in re.findall(r"([0-9.]+)秒", s)) - 69.0) < 0.1
    s2 = pipeline.timing_breakdown(dict(T, asr=3.0, total=34.9), {"load_sec": 34.0, "reused": True})
    assert "モデルの読み込み" not in s2 and "文字起こし 3.0秒" in s2
    # 区間検出のあいだに裏で読み込んだときは、待った時間(load_wait)だけが合計に効く
    s3 = pipeline.timing_breakdown(dict(T, asr=23.0, total=54.9), {"load_sec": 34.0, "load_wait": 20.0})
    assert "モデルの読み込み待ち 20.0秒" in s3 and "文字起こし 3.0秒" in s3
