# よくある質問

## `ModuleNotFoundError: No module named '_socket'`と出る

フォルダ名をインストールした時から変えていませんか？フォルダ名を変えるとパスが変わってしまい、インストール時に指定したパスと異なるためにエラーが出ます。フォルダ名を元に戻してください。

## APIサーバーで長い文章が合成できない

デフォルトで`server_fastapi.py`の入力文字上限は100文字に設定されています。
`config.yml`の`server.limit`の100を好きな数字に変更してください。
上限をなくしたい方は`server.limit`を-1に設定してください。

## その他

ググったり調べたりChatGPTに聞くか、それでも分からない場合・または手順通りやってもエラーが出る等明らかに不具合やバグと思われる場合は、GitHubの[Issue](https://github.com/litagin02/Style-Bert-VITS2/issues)に投稿してください。
