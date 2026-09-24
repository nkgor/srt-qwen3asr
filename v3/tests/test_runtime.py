# -*- coding: utf-8 -*-
"""runtime の venv まわり(GPU なしで確かめられる部分)"""
import os
import subprocess
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))

from asrkit import runtime  # noqa: E402

# Colab の google_generativeai-*-nspkg.pth と同じ中身(起動時に google をシステム側のパスだけで作る)
NSPKG_PTH = ("import sys, types, os;p = os.path.join(sys._getframe(1).f_locals['sitedir'], *('google',));"
             "importlib = __import__('importlib.util');__import__('importlib.machinery');"
             "m = sys.modules.setdefault('google', importlib.util.module_from_spec("
             "importlib.machinery.PathFinder.find_spec('google', [os.path.dirname(p)])));"
             "m = m or sys.modules.setdefault('google', types.ModuleType('google'));"
             "mp = (m or []) and m.__dict__.setdefault('__path__',[]);(p not in mp) and mp.append(p)\n")


def _mk(path, text=""):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


def test_google_namespace_fix(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "ENV_ROOT", str(tmp_path / "envs"))
    venv_sp = tmp_path / "envs" / "x" / "lib" / "python3.13" / "site-packages"
    sys_sp = tmp_path / "system"
    # venv: 新しい protobuf(google/ は __init__.py なしの名前空間)
    _mk(str(venv_sp / "google" / "protobuf" / "__init__.py"), "V = 'venv'\n")
    # システム: 古い protobuf ＋ google/__init__.py ＋ nspkg.pth ＋ ほかの google.* パッケージ
    _mk(str(sys_sp / "google" / "__init__.py"))
    _mk(str(sys_sp / "google" / "protobuf" / "__init__.py"), "V = 'system'\n")
    _mk(str(sys_sp / "google" / "colab" / "__init__.py"), "V = 'colab'\n")
    _mk(str(sys_sp / "google_generativeai-0.8.6-py3.13-nspkg.pth"), NSPKG_PTH)

    code = ("import site, sys; site.addsitedir(sys.argv[1]); site.addsitedir(sys.argv[2]); "
            "import google.protobuf, google.colab; print(google.protobuf.V, google.colab.V)")
    run = lambda: subprocess.run([sys.executable, "-S", "-c", code, str(venv_sp), str(sys_sp)],
                                 capture_output=True, text=True, check=True).stdout.split()
    assert run() == ["system", "colab"]  # 直す前: venv の protobuf が隠れる
    runtime._fix_google_namespace("x")
    assert (venv_sp / "_asr_v3_google_ns.pth").exists()
    assert run() == ["venv", "colab"]  # 直した後: venv が先、システムの google.* も読める
