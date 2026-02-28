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
インストール時にデフォルトのモデルがダウンロードされているので、学習していなくてもそれを使うことができます。

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

基本的にはBert-VITS2のモデル構造を少し改造しただけです。[旧事前学習モデル](https://huggingface.co/litagin/Style-Bert-VITS2-1.0-base)も[JP-Extraの事前学習モデル](https://huggingface.co/litagin/Style-Bert-VITS2-2.0-base-JP-Extra)も、実質Bert-VITS2 v2.1 or JP-Extraと同じものを使用しています（不要な重みを削ってsafetensorsに変換したもの）。

具体的には以下の点が異なります。

- 感情埋め込みのモデルを変更（256次元の[wespeaker-voxceleb-resnet34-LM](https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM)へ、感情埋め込みというよりは話者識別のための埋め込み）
- 感情埋め込みもベクトル量子化を取り払い、単なる全結合層に。
- スタイルベクトルファイル`style_vectors.npy`を作ることで、そのスタイルを使って効果の強さも連続的に指定しつつ音声を生成することができる。
- 各種WebUIを作成
- safetensors形式のサポート、デフォルトでsafetensorsを使用するように
- その他軽微なbugfixやリファクタリング


## References
In addition to the original reference (written below), I used the following repositories:
- [Bert-VITS2](https://github.com/fishaudio/Bert-VITS2)
- [EasyBertVits2](https://github.com/Zuntan03/EasyBertVits2)

[The pretrained model](https://huggingface.co/litagin/Style-Bert-VITS2-1.0-base) and [JP-Extra version](https://huggingface.co/litagin/Style-Bert-VITS2-2.0-base-JP-Extra) is essentially taken from [the original base model of Bert-VITS2 v2.1](https://huggingface.co/Garydesu/bert-vits2_base_model-2.1) and [JP-Extra pretrained model of Bert-VITS2](https://huggingface.co/Stardust-minus/Bert-VITS2-Japanese-Extra), so all the credits go to the original author ([Fish Audio](https://github.com/fishaudio)):


In addition, [style_bert_vits2/nlp/japanese/user_dict/](style_bert_vits2/nlp/japanese/user_dict) module is based on the following repositories:
- [voicevox_engine](https://github.com/VOICEVOX/voicevox_engine)
and the license of this module is LGPL v3.

## LICENSE

This repository is licensed under the GNU Affero General Public License v3.0, the same as the original Bert-VITS2 repository. For more details, see [LICENSE](LICENSE).

In addition, [style_bert_vits2/nlp/japanese/user_dict/](style_bert_vits2/nlp/japanese/user_dict) module is licensed under the GNU Lesser General Public License v3.0, inherited from the original VOICEVOX engine repository. For more details, see [LGPL_LICENSE](LGPL_LICENSE).



Below is the original README.md.
---

<div align="center">

<img alt="LOGO" src="https://cdn.jsdelivr.net/gh/fishaudio/fish-diffusion@main/images/logo_512x512.png" width="256" height="256" />

# Bert-VITS2

VITS2 Backbone with multilingual bert

For quick guide, please refer to `webui_preprocess.py`.

简易教程请参见 `webui_preprocess.py`。

## 请注意，本项目核心思路来源于[anyvoiceai/MassTTS](https://github.com/anyvoiceai/MassTTS) 一个非常好的tts项目
## MassTTS的演示demo为[ai版峰哥锐评峰哥本人,并找回了在金三角失落的腰子](https://www.bilibili.com/video/BV1w24y1c7z9)

[//]: # (## 本项目与[PlayVoice/vits_chinese]&#40;https://github.com/PlayVoice/vits_chinese&#41; 没有任何关系)

[//]: # ()
[//]: # (本仓库来源于之前朋友分享了ai峰哥的视频，本人被其中的效果惊艳，在自己尝试MassTTS以后发现fs在音质方面与vits有一定差距，并且training的pipeline比vits更复杂，因此按照其思路将bert)

## 成熟的旅行者/开拓者/舰长/博士/sensei/猎魔人/喵喵露/V应当参阅代码自己学习如何训练。

### 严禁将此项目用于一切违反《中华人民共和国宪法》，《中华人民共和国刑法》，《中华人民共和国治安管理处罚法》和《中华人民共和国民法典》之用途。
### 严禁用于任何政治相关用途。
#### Video:https://www.bilibili.com/video/BV1hp4y1K78E
#### Demo:https://www.bilibili.com/video/BV1TF411k78w
#### QQ Group：815818430
## References
+ [anyvoiceai/MassTTS](https://github.com/anyvoiceai/MassTTS)
+ [jaywalnut310/vits](https://github.com/jaywalnut310/vits)
+ [p0p4k/vits2_pytorch](https://github.com/p0p4k/vits2_pytorch)
+ [svc-develop-team/so-vits-svc](https://github.com/svc-develop-team/so-vits-svc)
+ [PaddlePaddle/PaddleSpeech](https://github.com/PaddlePaddle/PaddleSpeech)
+ [emotional-vits](https://github.com/innnky/emotional-vits)
+ [fish-speech](https://github.com/fishaudio/fish-speech)
+ [Bert-VITS2-UI](https://github.com/jiangyuxiaoxiao/Bert-VITS2-UI)
## 感谢所有贡献者作出的努力
<a href="https://github.com/fishaudio/Bert-VITS2/graphs/contributors" target="_blank">
  <img src="https://contrib.rocks/image?repo=fishaudio/Bert-VITS2"/>
</a>

[//]: # (# 本项目所有代码引用均已写明，bert部分代码思路来源于[AI峰哥]&#40;https://www.bilibili.com/video/BV1w24y1c7z9&#41;，与[vits_chinese]&#40;https://github.com/PlayVoice/vits_chinese&#41;无任何关系。欢迎各位查阅代码。同时，我们也对该开发者的[碰瓷，乃至开盒开发者的行为]&#40;https://www.bilibili.com/read/cv27101514/&#41;表示强烈谴责。)
