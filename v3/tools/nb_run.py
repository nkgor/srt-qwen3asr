# -*- coding: utf-8 -*-
"""ノートブックのフォームに値を入れて、コードセルを上から順に実行するテスト用ハーネス

  python v3/tools/nb_run.py Qwen3-ASR_v3.ipynb overrides.json

overrides.json の例(セル ID ごとに「変数名: 値」。"*" は全セル共通):
  {"v3setup": {"pyannote": false, "mount_drive": false},
   "v3main": {"src": "/content/test.wav", "model": "Qwen3-ASR 0.6B（軽量・速い）"},
   "skip": ["v3cmp"]}

Colab 以外では google.colab の代わりに最小限のダミーを入れる(userdata は環境変数を読む)。
Colab CLI(`colab exec -f`)からも同じように使える。
"""
import json
import os
import re
import sys
import time
import traceback
import types


def apply_overrides(src: str, ov: dict) -> str:
    out = []
    for line in src.split("\n"):
        m = re.match(r"^(\w+)\s*=\s*(.+?)\s*#@param", line)
        if m and m.group(1) in ov:
            line = f"{m.group(1)} = {ov[m.group(1)]!r}  # override"
        out.append(line)
    return "\n".join(out)


def install_fake_colab() -> None:
    try:
        import google.colab  # noqa: F401

        return  # 本物の Colab
    except Exception:
        pass
    google = sys.modules.get("google") or types.ModuleType("google")
    colab = types.ModuleType("google.colab")

    class _UserData:
        @staticmethod
        def get(key):
            v = os.environ.get(key)
            if not v:
                raise KeyError(key)
            return v

    def _upload():
        raise RuntimeError("このハーネスではアップロードできません(src を指定してください)")

    colab.userdata = _UserData
    colab.drive = types.SimpleNamespace(mount=lambda *a, **k: print("(fake) drive.mount", a))
    colab.files = types.SimpleNamespace(upload=_upload, download=lambda p: print("(fake) files.download", p))
    google.colab = colab
    sys.modules["google"] = google
    sys.modules["google.colab"] = colab


def main() -> int:
    ov = json.load(open(sys.argv[2], encoding="utf-8")) if len(sys.argv) > 2 else {}
    return run(sys.argv[1], ov)


def run(nb_path: str, ov: dict) -> int:
    """Colab CLI からは `import nb_run; nb_run.run("/content/x.ipynb", {...})` のように呼べる"""
    nb = json.load(open(nb_path, encoding="utf-8"))
    skip = set(ov.get("skip", []))
    install_fake_colab()
    ns = {"__name__": "__main__"}
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        cid = cell.get("metadata", {}).get("id") or cell.get("id")
        if cid in skip:
            print(f"\n########## skip {cid}", flush=True)
            continue
        src = apply_overrides("".join(cell["source"]), {**ov.get("*", {}), **ov.get(cid, {})})
        print(f"\n########## cell {cid} ##########", flush=True)
        t0 = time.time()
        try:
            exec(compile(src, f"<cell {cid}>", "exec"), ns)
        except BaseException:
            traceback.print_exc()
            print(f"########## FAILED {cid} ({time.time() - t0:.1f}s)", flush=True)
            return 1
        print(f"########## done {cid} ({time.time() - t0:.1f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
