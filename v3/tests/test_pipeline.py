# -*- coding: utf-8 -*-
"""ダミーエンジンで「ワーカー起動 → 文字起こし → タイムスタンプ → 話者 → 書き出し → キャッシュ」を通しで確認する"""
import json
import os
import shutil
import subprocess
import sys

import pytest

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg がない")


@pytest.fixture(scope="module")
def home(tmp_path_factory):
    d = tmp_path_factory.mktemp("asr_home")
    os.environ["ASR_V3_HOME"] = str(d)
    import importlib

    from asrkit import runtime

    importlib.reload(runtime)
    from asrkit import pipeline

    importlib.reload(pipeline)
    spec = runtime.EnvSpec(name="fake", packages=["numpy"], check="import numpy; print('numpy', numpy.__version__)")
    st = runtime.ensure_env(spec)
    assert st.ok, st.info
    return d


@pytest.fixture(scope="module")
def audio(tmp_path_factory):
    d = tmp_path_factory.mktemp("audio")
    src = str(d / "会議 テスト.m4a")
    # 0-20秒 発話, 20-24 無音, 24-70 発話(長いのでハード切りされる), 70-72 無音, 72-80 発話
    filt = ("sine=frequency=300:duration=80,volume='if(between(t,20,24)+between(t,70,72),0,1)':eval=frame")
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", filt, "-ac", "2", "-ar", "44100", src],
                   check=True)
    return src


def _session():
    from asrkit import pipeline, presets

    presets.PRESETS.append(presets.Preset("dummy", "ダミー", "dummy", "fake", "dummy-model"))
    sess = pipeline.Session()
    sess.aligner_env, sess.aligner_kind = "fake", "dummy-aligner"
    sess.diar_env, sess.diar_kind = "fake", "dummy-diar"
    return sess


def test_end_to_end(home, audio, tmp_path):
    from asrkit import pipeline

    sess = _session()
    out_dir = tmp_path / "out"
    st = pipeline.Settings(
        output_dir=str(out_dir), preset="dummy", vad="energy", max_clip=20, diarize=True, batch_size=2,
        formats=("txt", "srt", "vtt", "json", "csv", "md", "plain", "rttm"), context_terms=["Claude"],
        speaker_names="話者A=田中",
    )
    try:
        outs = sess.run(pipeline.resolve_inputs(audio), st)
        o = outs[0]
        for fmt in st.formats:
            assert os.path.exists(o["paths"][fmt]), fmt
        doc = json.load(open(o["paths"]["json"], encoding="utf-8"))
        assert doc["duration"] == pytest.approx(80.0, abs=0.2)
        assert doc["speakers"]["SPEAKER_00"] == "田中"
        assert doc["segments"][0]["text"].startswith("これは1番目の文です。")
        # ハード切りが起きていて、重なり部分の二重転写が無い(“N番目”の並びが単調)
        assert any(c["cut"] for c in doc["clips"])
        srt = open(o["paths"]["srt"], encoding="utf-8").read()
        assert "田中: " in srt and "。" in srt
        txt = open(o["paths"]["txt"], encoding="utf-8").read()
        assert txt.startswith("[00:00:0")
        csv_txt = open(o["paths"]["csv"], encoding="utf-8-sig").read()
        assert csv_txt.splitlines()[0] == "start,end,start_hms,speaker,text"
        # 2回目はキャッシュから(ワーカーを使わずに同じ結果)
        o2 = sess.transcribe_file(audio, st)
        assert o2["meta"]["asr"].get("cached") is True
        assert open(o2["paths"]["txt"], encoding="utf-8").read() == txt
        # 設定を変えずにモデルは読み込み済みのまま
        assert sess.h.workers["fake"].alive()
    finally:
        sess.free()


def test_resume_from_partial_cache(home, audio, tmp_path):
    from asrkit import pipeline

    sess = _session()
    st = pipeline.Settings(output_dir=str(tmp_path), preset="dummy", vad="energy", max_clip=20, batch_size=1,
                           formats=("txt",))
    try:
        o = sess.transcribe_file(audio, st)
        cache_dir = tmp_path / ".asr_v3_cache"
        files = [f for f in os.listdir(cache_dir) if f.endswith(".asr.jsonl")]
        assert len(files) == 1
        p = cache_dir / files[0]
        lines = open(p, encoding="utf-8").read().splitlines()
        # final 行と後半を消して「途中で落ちた」状態にする
        partial = [l for l in lines if '"final"' not in l][:2]
        open(p, "w", encoding="utf-8").write("\n".join(partial) + "\n")
        o2 = sess.transcribe_file(audio, st)
        assert open(o2["paths"]["txt"], encoding="utf-8").read() == open(o["paths"]["txt"], encoding="utf-8").read()
    finally:
        sess.free()


def test_context_echo_retry_through_worker(home, audio, tmp_path):
    from asrkit import pipeline

    sess = _session()
    st = pipeline.Settings(output_dir=str(tmp_path), preset="dummy", vad="energy", max_clip=20, batch_size=4,
                           formats=("json",), context_label="ECHO", context_terms=["Claude", "Gemini"], cache=False)
    try:
        o = sess.transcribe_file(audio, st)
        doc = json.load(open(o["paths"]["json"], encoding="utf-8"))
        texts = " ".join(s["text"] for s in doc["segments"])
        assert "ECHO" not in texts and "Claude" not in texts
        assert all(len(c.get("attempts", [])) >= 2 for c in doc["clips"])
    finally:
        sess.free()


def test_worker_error_is_reported(home):
    from asrkit import runtime

    w = runtime.Worker("fake")
    try:
        with pytest.raises(runtime.WorkerError) as ei:
            w.call("load", key="x", kind="no-such-engine", options={})
        assert "未知のエンジン" in str(ei.value)
        assert w.call("ping")["ok"]
    finally:
        w.close()


def test_second_opinion_and_review(home, audio, tmp_path):
    from asrkit import pipeline, presets

    sess = _session()
    presets.PRESETS.append(presets.Preset("dummy2", "ダミー2", "dummy", "fake", "dummy-model-2"))
    # context に ECHO を入れると dummy は復唱する → retry を切っておけば怪しいまま残る → dummy2 で読み直して差し替え
    st = pipeline.Settings(output_dir=str(tmp_path), preset="dummy", vad="energy", max_clip=20, batch_size=4,
                           formats=("json",), context_label="ECHO", context_terms=["Claude", "Gemini"], cache=False,
                           retry=False, second_opinion="dummy2")
    try:
        o = sess.transcribe_file(audio, st)
        assert o["meta"]["asr"]["second_opinion_adopted"] >= 1
        assert os.path.exists(o["paths"]["review"])
        txt = open(o["paths"]["review"], encoding="utf-8").read()
        assert "別モデル dummy2" in txt
        doc = json.load(open(o["paths"]["json"], encoding="utf-8"))
        assert all("ECHO" not in s["text"] for s in doc["segments"])
    finally:
        sess.free()


def test_second_opinion_resplits_long_clips(home, audio, tmp_path):
    """セカンドオピニオンのモデルのおすすめ(max_clip)より長いクリップは、区切り直して読ませて本文をつなぐ"""
    from asrkit import pipeline, presets

    sess = _session()
    presets.PRESETS.append(presets.Preset("dummy-short", "ダミー短", "dummy", "fake", "dummy-model-3", max_clip=8.0,
                                          max_gap=1.0))
    st = pipeline.Settings(output_dir=str(tmp_path), preset="dummy", vad="energy", max_clip=20, batch_size=4,
                           formats=("json",), context_label="ECHO", context_terms=["Claude"], cache=False,
                           retry=False, second_opinion="dummy-short")
    try:
        o = sess.transcribe_file(audio, st)
        assert o["meta"]["asr"]["second_opinion_adopted"] >= 1
        doc = json.load(open(o["paths"]["json"], encoding="utf-8"))
        tries = [a for c in doc["clips"] for a in c.get("attempts", []) if a.get("model") == "dummy-short"]
        assert tries and max(a.get("pieces", 1) for a in tries) >= 2
        assert all("ECHO" not in s["text"] for s in doc["segments"])
    finally:
        sess.free()


def test_compare_closes_idle_workers(home, audio, tmp_path):
    """⑥: 比べ終わった別環境のワーカーはプロセスごと止め、本命の環境は残す。表の材料(区切り方・抜け)もそろう"""
    from asrkit import pipeline, presets, runtime

    env2 = runtime.ensure_env(runtime.EnvSpec(name="fake2", packages=["numpy"],
                                              check="import numpy; print('numpy', numpy.__version__)"))
    assert env2.ok, env2.info
    sess = _session()
    presets.PRESETS.append(presets.Preset("dummy-other", "ダミー別環境", "dummy", "fake2", "dummy-model-4"))
    st = pipeline.Settings(output_dir=str(tmp_path), preset="dummy", vad="energy", batch_size=2, cache=False)
    try:
        rows = sess.compare(audio, ["dummy", "dummy-other"], st, start=0, duration=30, reference="これは1番目の文です。")
        assert [r["preset"] for r in rows] == ["dummy", "dummy-other"] and all("error" not in r for r in rows)
        assert "fake2" not in sess.h.workers  # 使い終わった別環境のワーカーは止めた
        assert "fake" in sess.h.workers and sess.h.workers["fake"].alive()  # 本命(とアライナー)の環境は残す
        assert rows[0]["clips"]["max_clip"] == pipeline.DEFAULT_MAX_CLIP and rows[0]["drops"] is not None
    finally:
        sess.free()


def test_minutes_prompt_only(tmp_path):
    from asrkit import pipeline

    t = tmp_path / "会議.md"
    t.write_text("**[00:00:01]** **話者A**  \n予算は来月決めます。\n", encoding="utf-8")
    p = pipeline.make_minutes(str(t), mode="プロンプトだけ作る", glossary=["予算"])
    body = open(p, encoding="utf-8").read()
    assert p.endswith("_minutes_prompt.md")
    assert "<transcript>" in body and "予算は来月決めます" in body and "TODO" in body and "予算" in body


def test_preset_max_clip_limits_clips(home, audio, tmp_path):
    """長いクリップで発話を飛ばすモデル(kotoba / Cohere)用: プリセットの max_clip で短く区切る"""
    from asrkit import pipeline, presets

    sess = _session()
    presets.PRESETS.append(presets.Preset("dummy-clip8", "ダミー(8秒)", "dummy", "fake", "dummy-model", max_clip=8))
    # max_clip / max_gap が「自動」(None)のときはプリセットのおすすめを使う
    st = pipeline.Settings(output_dir=str(tmp_path), preset="dummy-clip8", vad="energy", batch_size=2,
                           formats=("json",), cache=False)
    try:
        o = sess.transcribe_file(audio, st)
        doc = json.load(open(o["paths"]["json"], encoding="utf-8"))
        assert max(c["end"] - c["start"] for c in doc["clips"]) <= 8 + st.overlap + 0.01
        assert presets.find_preset("kotoba-whisper-v2").max_clip
        assert presets.find_preset("cohere-transcribe").max_clip
        assert presets.find_preset("cohere-transcribe-vllm").max_clip
        assert presets.find_preset("kotoba-whisper-v2").max_gap == presets.find_preset("cohere-transcribe").max_gap == 1.0
    finally:
        sess.free()


def _webui(tmp_path):
    import importlib

    from asrkit import webui

    importlib.reload(webui)
    webui.OUT_ROOT = str(tmp_path / "webui")
    return webui


def test_webui_transcribe(home, audio, tmp_path):
    from asrkit import pipeline

    webui = _webui(tmp_path)
    sess = _session()
    try:
        assert "ダミー" in [p.label for p in webui.available_presets()]
        r = webui.transcribe(sess, pipeline.Settings(vad="energy", max_clip=20, batch_size=2), audio,
                             model="ダミー", context="Claude", diarize=True)
        assert r["text"].startswith("[00:00:0")
        exts = {os.path.basename(f).split(".", 1)[-1] for f in r["files"]}
        assert {"txt", "srt", "vtt", "json", "csv", "md"} <= exts
        assert r["dir"].startswith(webui.OUT_ROOT)
        assert "pyannote" in r["summary"]  # 話者分離の環境が無いときは断って話者なしで出す
    finally:
        sess.free()


def test_webui_gradio_http(home, audio, tmp_path):
    """gradio の画面を本当に立ち上げて、HTTP 経由(gradio_client)で文字起こしする"""
    pytest.importorskip("gradio")
    gradio_client = pytest.importorskip("gradio_client")
    import socket

    from asrkit import pipeline

    webui = _webui(tmp_path)
    sess = _session()
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    try:
        webui.launch(sess, pipeline.Settings(preset="dummy", vad="energy", max_clip=20, batch_size=2), port=port)
        c = gradio_client.Client(f"http://127.0.0.1:{port}/", verbose=False)
        text, files, summary, logs = c.predict(gradio_client.handle_file(audio), None, "ダミー", "Japanese", "", False,
                                               0, "", api_name="/run")
        assert text.startswith("[00:00:0"), summary
        assert len(files) >= 6
        assert "倍速" in summary
        assert "[6/6]" in logs
    finally:
        webui.stop()
        sess.free()
