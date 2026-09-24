# Colab 実機テストの手順（Claude Code on the web / Colab CLI）

v3 ノートブックを、Colab の GPU ランタイム（H100 / A100 など）で**通しで**動かして確かめる手順です。
Claude Code のクラウドセッションから [Google Colab CLI](https://github.com/googlecolab/google-colab-cli) を使います。

## 事前準備（人がやること・一度だけ）

1. **ネットワーク**: クラウド環境の設定（セッション上部の環境メニュー → Edit）で Network access を「Full」
   （または `colab.research.google.com` `*.googleapis.com` `accounts.google.com` `huggingface.co` `*.hf.co` `github.com` `*.githubusercontent.com` を許可）
2. **Colab のログイン情報**: 手元の PC で
   ```
   gcloud auth application-default login --scopes=openid,https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/userinfo.email,https://www.googleapis.com/auth/colaboratory
   ```
   できた `application_default_credentials.json`（Windows は `%APPDATA%\gcloud\`）の中身を、環境変数 `COLAB_ADC_JSON` に登録
3. **Hugging Face**: `HF_TOKEN`（Read）を環境変数に登録。次のモデルで規約に同意しておく
   - 必須: https://huggingface.co/pyannote/speaker-diarization-community-1
   - Cohere を試すなら: https://huggingface.co/CohereLabs/cohere-transcribe-03-2026
4. 環境変数は**新しいセッションから**有効になるので、このブランチで新しいセッションを開く
5. テストが終わったら `gcloud auth application-default revoke` でログイン情報を無効化

## Claude がやること

```bash
cd /home/user/srt-qwen3asr   # リポジトリ
# 1) 認証情報(ADC)を置く
mkdir -p ~/.config/gcloud
printf '%s' "$COLAB_ADC_JSON" > ~/.config/gcloud/application_default_credentials.json
export GOOGLE_APPLICATION_CREDENTIALS=~/.config/gcloud/application_default_credentials.json

# 2) Colab CLI(Python 3.12 以上が必要。uv が面倒を見てくれる)
# google-colab-cli 0.7.2 は jupyter-kernel-client==0.8 固定だが、0.8 には JupyterSubprotocol が無く exec が落ちる → 0.9 に上書き
uv tool install google-colab-cli --with jupyter-kernel-client==0.9.0 --overrides <(echo "jupyter-kernel-client==0.9.0")
colab --auth adc new -s asr --gpu H100          # 空きが無いと Service Unavailable。そのときは A100
colab --auth adc status -s asr

# 3) ファイルを送る
colab --auth adc upload -s asr Qwen3-ASR_v3.ipynb /content/Qwen3-ASR_v3.ipynb
colab --auth adc upload -s asr v3/tools/nb_run.py /content/nb_run.py

# 4) HF_TOKEN を VM の環境変数へ(チャットには出さない)
printf 'import os\nos.environ["HF_TOKEN"] = %s\nprint("HF_TOKEN set")\n' "$(python3 -c 'import json,os;print(json.dumps(os.environ["HF_TOKEN"]))')" | colab --auth adc exec -s asr

# 5) 通しテスト(テスト音声づくり → ⓪〜⑧ をフォームに値を入れて実行)
#    E2E_ENGINES で入れる環境、E2E_COMPARE で⑥の比較対象、E2E_MINUTES で音声の長さ(分)を指定できる
printf 'import os\nos.environ.update(E2E_MINUTES="6", E2E_CMP_SEC="300")\n' | colab --auth adc exec -s asr
colab --auth adc exec -s asr --timeout 5400 -f v3/tools/colab_e2e.py   # --timeout の既定は 30 秒(切れても VM 側は動き続ける)

# 6) 結果を回収
colab --auth adc download -s asr /content/e2e_out.tgz ./e2e_out.tgz
mkdir -p /tmp/e2e && tar xzf e2e_out.tgz -C /tmp/e2e

# 7) 片付け(課金を止める)
colab --auth adc stop -s asr
```

うまくいかないときは `colab --auth adc log -s asr -o log.md` で実行履歴を取り出せます。

カーネルが通しテストで埋まっている間も、**SSH** なら並行してログや venv を調べられます（`ssh` / `ssh-keygen` が要る。無ければ `apt-get install openssh-client`）:

```bash
ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519 -q
cat >> ~/.ssh/config <<'CFG'
Host colab-asr
  User root
  ProxyCommand env GOOGLE_APPLICATION_CREDENTIALS=/root/.config/gcloud/application_default_credentials.json colab --auth adc ssh --proxy-mode -s asr
  StrictHostKeyChecking no
  UserKnownHostsFile /dev/null
CFG
ssh colab-asr 'tail -50 /content/asr_v3/logs/vllm_*.log'
```

同時に張れる SSH は 1 本だけ（2 本目は HTTP 429）。前の接続が切れるまで数秒待ってから次を。
`colab exec` はセルと同じくカーネルの中で動くので、途中の変数(`SESS`, `OUTS`, `CMP` など)も続けて確かめられます。

## 確かめること

- [ ] ① すべての環境（qwen / pyannote / vllm / fw / nemo / hf）が入る。いまの Colab の torch / CUDA / ドライバで動くか
- [ ] flash-attn の whl が合うか（合わなければ sdpa で動くこと）
- [ ] ⑤ Qwen3-ASR 1.7B ＋ 話者分離で、txt / srt / vtt / json / csv / md / plain / rttm / review が出る
- [ ] CER（`/content/e2e/meeting_cv8.ref.txt` と比べる）と速度（何倍速か）
- [ ] ⑥ の比較表: Qwen 1.7B / JA 版 / 0.6B / Cohere / Granite / VibeVoice / Whisper turbo / kotoba / Parakeet / vLLM（Qwen・Cohere）
- [ ] vLLM サーバーが起動して、Qwen3-ASR を高速に処理できるか（`vllm` のバージョンと CUDA の相性）
- [ ] ⑧ で VRAM がちゃんと解放されるか
