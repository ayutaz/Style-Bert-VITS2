# Style-Bert-VITS2

**利用の際は必ず[お願いとデフォルトモデルの利用規約](/docs/TERMS_OF_USE.md)をお読みください。**

Bert-VITS2 with more controllable voice styles.

https://github.com/litagin02/Style-Bert-VITS2/assets/139731664/e853f9a2-db4a-4202-a1dd-56ded3c562a0

You can install via `pip install style-bert-vits2` (inference only).

- [**よくある質問** (FAQ)](/docs/FAQ.md)
- [🤗 オンラインデモはこちらから](https://huggingface.co/spaces/litagin/Style-Bert-VITS2-Editor-Demo)
- [Zennの解説記事](https://zenn.dev/litagin/articles/034819a5256ff4)

- [**リリースページ**](https://github.com/litagin02/Style-Bert-VITS2/releases/)、[更新履歴](/docs/CHANGELOG.md)

This repository is based on [Bert-VITS2](https://github.com/fishaudio/Bert-VITS2) v2.1 and Japanese-Extra, so many thanks to the original author!

**概要**

- 入力されたテキストの内容をもとに感情豊かな音声を生成する[Bert-VITS2](https://github.com/fishaudio/Bert-VITS2)のv2.1とJapanese-Extraを元に、感情や発話スタイルを強弱込みで自由に制御できるようにしたものです。
- 音声合成のみに使う場合は、グラボがなくてもCPUで動作します。
- 音声合成のみに使う場合、Pythonライブラリとして`pip install style-bert-vits2`でインストールできます。
- 元々「楽しそうな文章は楽しそうに、悲しそうな文章は悲しそうに」読むのがBert-VITS2の強みですので、スタイル指定がデフォルトでも感情豊かな音声を生成することができます。

## 使い方

- [よくある質問](/docs/FAQ.md)も参照してください。

### 動作環境

Windows コマンドプロンプト・WSL2・Linux(Ubuntu Desktop)での動作を確認しています。NVidiaのGPUが無い場合でも音声合成は可能です。

### インストール

Pythonの仮想環境・パッケージ管理ツールである[uv](https://github.com/astral-sh/uv)を使ってインストールします。

```bash
# uvのインストール（未導入の場合）
# Windows: powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
# macOS/Linux: curl -LsSf https://astral.sh/uv/install.sh | sh

git clone https://github.com/litagin02/Style-Bert-VITS2.git
cd Style-Bert-VITS2
uv sync --group infer                # 推論環境
uv run python initialize.py          # 必要なモデルとデフォルトTTSモデルをダウンロード
```

### 音声合成

音声合成WebUIは`uv run python app.py`で起動します（`--device cpu`でCPUモードで起動）。
インストール時にデフォルトのモデルがダウンロードされているので、すぐに使うことができます。

WebUIには以下の機能があり、タブで切り替えて使用できます：
- **音声合成** (`/inference`) — テキストからスタイルを指定して音声を生成
- **モーフィングUI** (`/morphing`) — スタイル間の補間操作で中間的な発話スタイルを生成
- **ベクトル演算UI** (`/vector-arithmetic`) — スタイルベクトルの算術操作（差分転写・加重平均・スケーリング等）

音声合成に必要なモデルファイルたちの構造は以下の通りです（手動で配置する必要はありません）。
```
model_assets
├── your_model
│   ├── config.json
│   ├── your_model_file1.safetensors
│   ├── your_model_file2.safetensors
│   ├── ...
│   └── style_vectors.npy
└── another_model
    ├── ...
```
このように、推論には`config.json`と`*.safetensors`と`style_vectors.npy`が必要です。モデルを共有する場合は、この3つのファイルを共有してください。

## Bert-VITS2との関係

基本的にはBert-VITS2のモデル構造を少し改造しただけです。[旧ベースモデル](https://huggingface.co/litagin/Style-Bert-VITS2-1.0-base)も[JP-Extraベースモデル](https://huggingface.co/litagin/Style-Bert-VITS2-2.0-base-JP-Extra)も、実質Bert-VITS2 v2.1 or JP-Extraと同じものです（不要な重みを削ってsafetensorsに変換したもの）。

具体的には以下の点が異なります。

- 感情埋め込みのモデルを変更（256次元の[wespeaker-voxceleb-resnet34-LM](https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM)へ、感情埋め込みというよりは話者識別のための埋め込み）
- 感情埋め込みもベクトル量子化を取り払い、単なる全結合層に。
- スタイルベクトルファイル`style_vectors.npy`を作ることで、そのスタイルを使って効果の強さも連続的に指定しつつ音声を生成することができる。
- 各種WebUIを作成
- safetensors形式のサポート、デフォルトでsafetensorsを使用するように
- その他軽微なbugfixやリファクタリング

## References
I used the following repositories:
- [Bert-VITS2](https://github.com/fishaudio/Bert-VITS2)
- [EasyBertVits2](https://github.com/Zuntan03/EasyBertVits2)

[The base model](https://huggingface.co/litagin/Style-Bert-VITS2-1.0-base) and [JP-Extra version](https://huggingface.co/litagin/Style-Bert-VITS2-2.0-base-JP-Extra) are derived from [Bert-VITS2 v2.1](https://huggingface.co/Garydesu/bert-vits2_base_model-2.1) and [Bert-VITS2 JP-Extra](https://huggingface.co/Stardust-minus/Bert-VITS2-Japanese-Extra). All credits go to the original author ([Fish Audio](https://github.com/fishaudio)).

In addition, [style_bert_vits2/nlp/japanese/user_dict/](style_bert_vits2/nlp/japanese/user_dict) module is based on the following repositories:
- [voicevox_engine](https://github.com/VOICEVOX/voicevox_engine)
and the license of this module is LGPL v3.

## LICENSE

This repository is licensed under the GNU Affero General Public License v3.0, the same as the original Bert-VITS2 repository. For more details, see [LICENSE](LICENSE).

In addition, [style_bert_vits2/nlp/japanese/user_dict/](style_bert_vits2/nlp/japanese/user_dict) module is licensed under the GNU Lesser General Public License v3.0, inherited from the original VOICEVOX engine repository. For more details, see [LGPL_LICENSE](LGPL_LICENSE).
