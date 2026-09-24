# -*- coding: utf-8 -*-
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from asrkit import core  # noqa: E402
from asrkit.core import Clip, ClipResult, Word  # noqa: E402


# ------------------------------------------------------------------ 時刻
def test_time_formats_round_properly():
    assert core.fmt_srt_time(0) == "00:00:00,000"
    assert core.fmt_srt_time(59.9996) == "00:01:00,000"  # 1000ms にならない
    assert core.fmt_srt_time(3661.2345) == "01:01:01,234"
    assert core.fmt_vtt_time(1.5) == "00:00:01.500"
    assert core.fmt_hms(3599.9) == "00:59:59"


# ------------------------------------------------------------------ 句読点の付け直し
def test_restore_display_japanese_punct():
    text = "今日は、晴れです。明日も「たぶん」晴れ！"
    toks = [("今日", 0, 1), ("は", 1, 2), ("晴れ", 2, 3), ("です", 3, 4), ("明日", 5, 6), ("も", 6, 7),
            ("たぶん", 7, 8), ("晴れ", 8, 9)]
    out = core.restore_display(text, toks)
    assert "".join(d["word"] for d in out) == text
    assert out[1]["word"] == "は、"
    assert out[3]["word"] == "です。"
    assert out[5]["word"] == "も"  # 開き括弧は次の語へ
    assert out[6]["word"] == "「たぶん」"
    assert out[-1]["word"] == "晴れ！"


def test_restore_display_symbols_inside_tokens():
    text = "C++とAI-drivenで50%削減、OK?"
    toks = [("C", 0, 1), ("と", 1, 2), ("AIdriven", 2, 3), ("で", 3, 4), ("50", 4, 5), ("削減", 5, 6), ("OK", 6, 7)]
    out = core.restore_display(text, toks)
    assert "".join(d["word"] for d in out) == text
    assert out[0]["word"] == "C++"
    assert out[2]["word"] == "AI-driven"
    assert out[4]["word"] == "50%"


def test_restore_display_english_spaces():
    text = "Hello, world. It's fine."
    toks = [("Hello", 0, 1), ("world", 1, 2), ("It's", 2, 3), ("fine", 3, 4)]
    out = core.restore_display(text, toks)
    assert "".join(d["word"] for d in out) == text
    assert out[0]["word"] == "Hello, "


def test_restore_display_unmatched_token_keeps_text():
    text = "あいうえお"
    toks = [("あい", 0, 1), ("ZZZ", 1, 2), ("えお", 2, 3)]
    out = core.restore_display(text, toks)
    assert "".join(d["word"] for d in out) == text


# ------------------------------------------------------------------ クリップ
def test_build_clips_merge_and_limits():
    segs = [(0.5, 3.0), (3.5, 10.0), (11.0, 20.0), (40.0, 45.0)]
    clips = core.build_clips(segs, 60.0, max_clip=30, max_gap=6, pad=0.2)
    # 0.5〜20 は1つ(19.5秒), 40〜45 は無音20秒あくので別
    assert len(clips) == 2
    assert clips[0].start == pytest.approx(0.3)
    assert clips[0].end == pytest.approx(20.2)
    # 担当区間の境目は無音の真ん中(20〜40 の中点)
    assert clips[0].own_start < -1e17 and clips[0].own_end == pytest.approx(30.0)
    assert clips[1].own_start == pytest.approx(30.0) and clips[1].own_end > 1e17
    assert clips[1].start == pytest.approx(39.8)


def test_build_clips_margins_overlap_but_ownership_partitions():
    # 無音 0.3 秒しかない所でクリップが割れる → 余白 0.5 秒は隣に食い込むが担当区間で二重採用しない
    segs = [(0.0, 14.0), (14.3, 27.0), (27.3, 40.0)]
    clips = core.build_clips(segs, 40.0, max_clip=30, pad=0.5, overlap=1.0)
    assert len(clips) >= 2
    for a, b in zip(clips, clips[1:]):
        assert a.end > b.start  # 余白どうしは重なる
        assert a.own_end == pytest.approx(b.own_start)
    for c in clips:
        assert c.dur <= 30.0 + 1e-6  # 余白込みで上限以内


def test_assign_speakers_with_nested_turns():
    # GPT 6 Pro の指摘の例: A が 0〜10 秒話していて、B(1〜2秒)・C(3〜4秒)が相づち → 5秒の単語は A
    ws = [Word("はい", 4.9, 5.1)]
    core.assign_speakers(ws, [(0.0, 10.0, "A"), (1.0, 2.0, "B"), (3.0, 4.0, "C")])
    assert ws[0].speaker == "A"
    ws = [Word("うん", 1.2, 1.9)]
    core.assign_speakers(ws, [(0.0, 10.0, "A"), (1.0, 2.0, "B")])
    assert ws[0].speaker in ("A", "B")


def test_overlap_regions():
    reg = core.overlap_regions([(0.0, 10.0, "A"), (1.0, 2.0, "B"), (3.0, 3.1, "C"), (9.0, 12.0, "B")])
    assert reg == [(1.0, 2.0, ["A", "B"]), (9.0, 10.0, ["A", "B"])]


def test_build_clips_hard_split_with_overlap_and_ownership():
    total = 100.0
    db = np.full(int(total / core.HOP), 60.0, np.float32)
    # 47秒あたりに静かな所を作る
    db[int(47.0 / core.HOP): int(47.2 / core.HOP)] = 0.0
    clips = core.build_clips([(0.0, 100.0)], total, max_clip=60, overlap=1.0, db=db)
    assert len(clips) == 2
    cut = clips[0].own_end
    assert 46.9 <= cut <= 47.3
    assert clips[1].own_start == pytest.approx(cut)
    assert clips[0].end == pytest.approx(cut + 1.0)
    assert clips[1].start == pytest.approx(cut - 1.0)
    for c in clips:
        assert c.dur <= 60.0 + 1e-6


def test_build_clips_never_exceeds_max_clip():
    rng = np.random.default_rng(0)
    total = 3 * 3600.0
    db = rng.uniform(30, 70, int(total / core.HOP)).astype(np.float32)
    segs = [(0.0, 1200.0), (1205.0, 1210.0), (1210.5, 5000.0), (5001.0, total)]
    for max_clip in (15, 30, 60, 120):
        clips = core.build_clips(segs, total, max_clip=max_clip, overlap=1.0, db=db)
        assert max(c.dur for c in clips) <= max_clip + 1e-6
        # 担当区間は隙間なく並ぶ(ハード切り部分)
        for a, b in zip(clips, clips[1:]):
            if a.own_end < 1e17:
                assert b.own_start == pytest.approx(a.own_end)


def test_energy_vad_simple():
    db = np.full(500, 10.0, np.float32)  # 10秒
    db[50:150] = 70  # 1.0〜3.0
    db[160:170] = 70  # 3.2〜3.4 (0.2秒の無音はつなぐ)
    db[300:305] = 70  # 0.1秒だけ → 捨てる
    segs = core.energy_vad(db, top_db=30, min_speech=0.25, min_silence=0.4)
    assert segs == [(1.0, 3.4)]


# ------------------------------------------------------------------ 品質チェック & リトライ
def test_quality_flags():
    assert core.quality_flags("", 10, 8) == ["empty"]
    assert core.quality_flags("", 10, 1) == []
    loop = "ありがとうございました。" * 8
    assert "repetition" in core.quality_flags(loop, 30, 20)
    assert "context_label" in core.quality_flags("固有名詞・専門用語: Claude", 5, 3, ctx_label="固有名詞・専門用語")
    assert "context_echo" in core.quality_flags("Claude、Gemini、Codex。", 5, 3, ctx_terms=["Claude", "Gemini", "Codex"])
    assert core.quality_flags("今日はClaudeとGeminiを比べます。", 5, 3, ctx_terms=["Claude", "Gemini"]) == []
    assert "too_dense" in core.quality_flags("あ" * 200, 5, 5)
    normal = "今日は会議の議題について話します。まず最初に予算の件ですが、昨年度と比べて少し増えています。"
    assert core.quality_flags(normal, 10, 9) == []


def test_run_asr_retry_without_context_and_split():
    clips = [Clip(0, 0.0, 10.0, speech=9.0), Clip(1, 10.0, 30.0, speech=18.0), Clip(2, 30.0, 35.0, speech=4.5)]
    calls = []

    def transcribe(items):
        calls.append([(len(a), ctx) for a, ctx in items])
        out = []
        for a, ctx in items:
            n = len(a) / core.SR
            if ctx and n > 15:  # 長いクリップは context ありだと復唱する
                out.append({"text": "固有名詞: Claude、Gemini。"})
            elif not ctx and n > 15:  # context なしでもループ → 分割へ
                out.append({"text": "はい。" * 10})
            else:
                out.append({"text": "これは普通の発話です。"})
        return out

    get_audio = lambda s, e: np.zeros(int((e - s) * core.SR), np.float32)  # noqa: E731
    res = core.run_asr(
        clips, get_audio, transcribe, context="固有名詞: Claude、Gemini。", ctx_terms=["Claude", "Gemini"],
        ctx_label="固有名詞", batch_size=2,
    )
    # 最初のバッチは長い順
    assert calls[0][0][0] == 20 * core.SR
    texts = [r.text for r in res]
    assert all("Claude" not in t for t in texts)
    # clip1 は2分割されて置き換わる
    assert [r.clip.parent for r in res].count(1) == 2
    assert all(not r.flags for r in res)
    # 分割の担当区間はつながっている
    subs = [r.clip for r in res if r.clip.parent == 1]
    assert subs[0].own_end == pytest.approx(subs[1].own_start)


def test_run_asr_keeps_best_and_collapses_loops():
    clips = [Clip(0, 0.0, 6.0, speech=5.0)]

    def transcribe(items):
        return [{"text": "そうですね" * 9} for _ in items]

    res = core.run_asr(clips, lambda s, e: np.zeros(10, np.float32), transcribe, context="", batch_size=4)
    assert res[0].flags == ["repetition"]
    assert res[0].text == "そうですね" * 2


# ------------------------------------------------------------------ 単語・話者・出力
def _res(cid, start, end, text, own=(-1e18, 1e18)):
    return ClipResult(Clip(cid, start, end, own[0], own[1]), text)


def test_compose_words_dedup_overlap():
    r0 = _res(0, 0.0, 31.0, "前半です。境界", own=(-1e18, 30.0))
    r1 = _res(1, 29.0, 60.0, "境界の後半です。", own=(30.0, 1e18))
    aligned = {
        0: [["前半", 1.0, 2.0], ["です", 2.0, 3.0], ["境界", 29.6, 30.6]],  # 中点30.1 → clip0 は捨てる
        1: [["境界", 0.6, 1.6], ["の", 1.6, 1.8], ["後半", 2.0, 3.0], ["です", 3.0, 4.0]],  # 絶対 29.6-30.6
    }
    words, stats = core.compose_words([r0, r1], aligned)
    assert core.join_words(words) == "前半です。境界の後半です。"
    assert stats == {"aligner": 2}


def test_assign_and_smooth_speakers():
    ws = [Word(w, s, s + 0.4) for w, s in [("今日は", 0), ("いい", 0.5), ("天気", 1.0), ("ですね。", 1.5),
                                          ("そう", 3.0), ("ですね。", 3.5)]]
    turns = [(0.0, 0.9, "S0"), (0.9, 1.2, "S1"), (1.2, 2.0, "S0"), (2.9, 4.0, "S1")]
    core.assign_speakers(ws, turns)
    assert ws[2].speaker == "S1"  # 天気 は S1 と最大重なり(チラつき)
    core.smooth_speakers(ws)
    assert [w.speaker for w in ws] == ["S0"] * 4 + ["S1"] * 2
    names = core.speaker_names(ws)
    assert names == {"S0": "話者A", "S1": "話者B"}
    names = core.speaker_names(ws, mapping_spec="話者A=田中, S1=佐藤")
    assert names == {"S0": "田中", "S1": "佐藤"}
    names = core.speaker_names(ws, mapping_spec="山田、鈴木")
    assert names == {"S0": "山田", "S1": "鈴木"}


def test_cues_and_srt():
    post = core.TextPost()
    text = "今日は会議です。まず予算について、説明します。次に日程です。"
    toks = [("今日", 0.0, 0.3), ("は", 0.3, 0.4), ("会議", 0.4, 0.8), ("です", 0.8, 1.0),
            ("まず", 1.5, 1.8), ("予算", 1.8, 2.2), ("について", 2.2, 2.6), ("説明", 2.7, 3.0), ("し", 3.0, 3.1),
            ("ます", 3.1, 3.3), ("次に", 5.0, 5.3), ("日程", 5.3, 5.7), ("です", 5.7, 6.0)]
    words = [Word(d["word"], d["start"], d["end"], d["core"]) for d in core.restore_display(text, toks)]
    cues = core.make_cues(words, post, core.CueRules(max_chars=16, punct_split_min=5))
    assert [c.text for c in cues] == ["今日は会議です。", "まず予算について、説明します。", "次に日程です。"]
    for a, b in zip(cues, cues[1:]):
        assert a.end <= b.start
    srt = core.to_srt(cues, {})
    assert srt.startswith("1\n00:00:00,000 --> 00:00:01,200\n今日は会議です。\n")
    vtt = core.to_vtt(cues, {})
    assert vtt.startswith("WEBVTT")


def test_sentence_split_does_not_break_decimals():
    text = "成長率は3.5%でした。Next step. OK"
    toks = [("成長", 0, .2), ("率", .2, .3), ("は", .3, .4), ("3", .4, .5), ("5", .5, .6), ("でし", .6, .7), ("た", .7, .8),
            ("Next", 1, 1.2), ("step", 1.2, 1.4), ("OK", 2, 2.2)]
    words = [Word(d["word"], d["start"], d["end"]) for d in core.restore_display(text, toks)]
    sents = core.split_sentences(words)
    assert [core.join_words(s) for s in sents] == ["成長率は3.5%でした。", "Next step.", "OK"]


def test_textpost_replacements_fillers_spaces():
    rules = core.parse_replacements("クロード => Claude\nre:ジェミ[ニ二] → Gemini\n# comment\nＡＩ\tAI")
    rx = core.make_filler_regex(core.FILLERS_SAFE)
    post = core.TextPost(rules, rx)
    assert post("えー、今日は クロード と ジェミ二の話です。") == "今日はClaudeとGeminiの話です。"
    assert post("あのー、ＡＩについて、えーと、話します。") == "AIについて、話します。"
    assert post("Hello world です") == "Hello world です"
    assert post("えーっと。") == "。" or post("えーっと。") == ""


def test_cer():
    assert core.cer("今日は晴れ。", "今日は晴れ") == 0.0
    assert core.cer("ＡＢＣ", "abc") == 0.0
    assert core.cer("あいうえお", "あいうえか") == pytest.approx(0.2)


def test_build_context():
    assert core.build_context("固有名詞・専門用語", ["Claude", "Gemini"]) == "固有名詞・専門用語: Claude、Gemini。"
    assert core.build_context("Proper nouns", ["Claude", "Gemini"]) == "Proper nouns: Claude, Gemini."
    assert core.build_context("", []) == ""
    assert core.parse_terms("Claude Codex、Gemini,競争・脱落 Claude") == ["Claude", "Codex", "Gemini", "競争", "脱落"]


def test_json_and_rttm_roundtrip():
    words = [Word("こんにちは。", 0.0, 1.0, speaker="S0"), Word("はい。", 1.5, 2.0, speaker="S1")]
    sents = core.make_sentences(words, core.TextPost())
    names = core.speaker_names(words)
    res = [_res(0, 0.0, 2.5, "こんにちは。はい。")]
    doc = json.loads(core.to_json(sents, names, res, {"version": "test"}))
    assert doc["segments"][1]["speaker"] == "話者B"
    assert doc["clips"][0]["own_start"] is None
    rttm = core.to_rttm([(0.0, 1.2, "S0"), (1.4, 2.1, "S1")], "会議 1", names)
    assert rttm.splitlines()[0] == "SPEAKER 会議_1 1 0.000 1.200 <NA> <NA> 話者A <NA> <NA>"


def test_approx_tokens_cover_range():
    toks = core.approx_tokens("今日は晴れ。明日は雨です。", 10.0, 20.0, [(11.0, 13.0), (15.0, 18.0)])
    assert toks[0][1] == pytest.approx(11.0)
    assert toks[-1][2] == pytest.approx(18.0)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg がない")
def test_ffmpeg_decode_and_wav16(tmp_path):
    src = tmp_path / "in.m4a"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
                    "-ac", "2", "-ar", "44100", str(src)], check=True)
    info = core.ffprobe(str(src))
    assert info["channels"] == 2 and info["has_audio"]
    dst = tmp_path / "out.wav"
    core.ffmpeg_to_wav16(str(src), str(dst), channel="left", af="loudnorm=I=-20:TP=-2:LRA=11")
    w = core.Wav16(str(dst))
    assert w.sr == 16000
    assert abs(w.duration - 3.0) < 0.1
    x = w.get(1.0, 2.0)
    assert x.dtype == np.float32 and len(x) == 16000
    db = core.frame_db(w)
    assert len(db) == int(len(w) / 320)
    b = core.wav_bytes(x)
    assert b[:4] == b"RIFF"


def test_foreign_script_flag():
    assert "foreign_script" in core.quality_flags("当時、怪しいワルドに注중していた", 5, 4, lang="Japanese")
    assert "foreign_script" not in core.quality_flags("当時、怪しいワールドに常駐していた", 5, 4, lang="Japanese")
    assert "foreign_script" not in core.quality_flags("注중", 5, 4, lang="")  # 言語指定なしなら見ない


def test_review_items_and_markdown():
    ok = ClipResult(Clip(0, 0, 5), "普通です。", attempts=[{"ctx": True, "text": "普通です。", "flags": []}])
    fixed = ClipResult(Clip(1, 5, 10), "直った。", attempts=[{"ctx": True, "text": "固有名詞: A、B。", "flags": ["context_label"]},
                                                          {"ctx": False, "text": "直った。", "flags": []}])
    bad = ClipResult(Clip(2, 10, 20), "はいはいはい", flags=["repetition"],
                     attempts=[{"ctx": False, "text": "はいはいはい", "flags": ["repetition"]}])
    items = core.review_items([ok, fixed, bad])
    assert [r.clip.id for r in items] == [1, 2]
    md = core.to_review_md([ok, fixed, bad], "会議")
    assert "自動で差し替え済み" in md and "同じ言葉のループ" in md and "context なし" in md


def test_term_and_number_recall():
    ref = "山田さんは3月5日に1,200円払いました。田中さんは三回来ました。"
    hyp = "山田さんは3月6日に1,200円払いました。中田さんは三回来ました。"
    assert core.term_recall(ref, hyp, ["山田", "田中"]) == (1, 2)
    assert core.number_recall(ref, hyp) == (3, 4)  # 3, 1,200, 三 は合う / 5 が 6 に


def test_drop_runs_counts_skipped_utterances():
    ref = "今日は良い天気ですね。明日は雨が降るでしょう。午後から会議です。"
    assert core.drop_runs(ref, ref) == (0, 0)
    # 真ん中の文を丸ごと読み飛ばした
    n, chars = core.drop_runs(ref, "今日は良い天気ですね。午後から会議です。")
    assert n == 1 and chars == len(core.normalize_for_cer("明日は雨が降るでしょう"))
    # 数文字の言い間違いは「抜け」ではない
    assert core.drop_runs(ref, "今日はいい天気ですね。明日は雨がふるでしょう。午後から会議です。") == (0, 0)
    # 偶然そろった字(「は」など)をはさんでも、ひと続きの抜けとして数える
    ref2 = "わたしはきのう図書館で本を借りました。あなたは何をしていましたか。"
    n2, c2 = core.drop_runs(ref2, "わたしはきのう図書館で本を借りました。は。")
    assert n2 == 1 and c2 >= 12


def test_negation_check():
    ref = "今日は行かない。雨は降りません。"
    assert core.negation_check(ref, ref) == (2, 2, 0)
    assert core.negation_check(ref, "今日は行く。雨は降りません。") == (1, 2, 0)  # 「行かない」→「行く」(意味が反転)
    assert core.negation_check("明日は行く。", "明日は行かない。") == (0, 0, 1)  # 正解に無い否定が出た


def test_join_texts():
    assert core.join_texts(["今日は", " 晴れ。", "", "Claude", "Code を使う"]) == "今日は晴れ。Claude Code を使う"
    assert core.join_texts([]) == ""
