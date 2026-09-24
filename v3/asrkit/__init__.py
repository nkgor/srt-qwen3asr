# -*- coding: utf-8 -*-
"""asrkit — Qwen3-ASR v3 ノートブックの本体

core     : 音声・区間分割・テキスト処理・書き出し(numpy だけで動く純粋ロジック)
runtime  : venv づくり・ワーカープロセス・vLLM サーバー・GPU/Colab まわり
engines  : ワーカーの中で動く各モデルのアダプタ(Qwen3-ASR / アライナー / Whisper / NeMo / pyannote …)
worker   : ワーカーのメインループ(JSON を1行ずつやりとり)
presets  : モデルのプリセットと venv の中身
pipeline : ノートブックから呼ぶ高レベル処理(文字起こし・モデル比較)
"""
__version__ = "3.0.0"
