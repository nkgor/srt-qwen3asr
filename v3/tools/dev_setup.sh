#!/usr/bin/env bash
# ローカル(Colab 以外)でテストを回すための準備: Python 3.12 の venv と ffmpeg
#   bash v3/tools/dev_setup.sh && source .venv-v3/bin/activate && python -m pytest v3/tests -q
set -euo pipefail
cd "$(dirname "$0")/../.."
command -v uv >/dev/null || pip install -q uv
[ -d .venv-v3 ] || uv venv -q -p 3.12 .venv-v3
uv pip install -q --python .venv-v3/bin/python numpy pytest tqdm ipython pyarrow nbformat imageio-ffmpeg gradio gradio_client
if ! command -v ffmpeg >/dev/null; then
  FF=$(.venv-v3/bin/python -c "import imageio_ffmpeg as i; print(i.get_ffmpeg_exe())")
  ln -sf "$FF" .venv-v3/bin/ffmpeg
  echo "ffmpeg: $FF (.venv-v3/bin/ffmpeg にリンク)"
fi
echo "OK: source .venv-v3/bin/activate"
