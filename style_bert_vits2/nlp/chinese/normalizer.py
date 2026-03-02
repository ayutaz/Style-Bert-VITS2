import re

import cn2an

from style_bert_vits2.nlp.symbols import PUNCTUATIONS


__REPLACE_MAP = {
    "：": ",",
    "；": ",",
    "，": ",",
    "。": ".",
    "！": "!",
    "？": "?",
    "\n": ".",
    "·": ",",
    "、": ",",
    "...": "…",
    "$": ".",
    "“": "'",
    "”": "'",
    '"': "'",
    "‘": "'",
    "’": "'",
    "（": "'",
    "）": "'",
    "(": "'",
    ")": "'",
    "《": "'",
    "》": "'",
    "【": "'",
    "】": "'",
    "[": "'",
    "]": "'",
    "—": "-",
    "～": "-",
    "~": "-",
    "「": "'",
    "」": "'",
}


def normalize_text(text: str) -> str:
    numbers = re.findall(r"\d+(?:\.?\d+)?", text)
    for number in numbers:
        text = text.replace(number, cn2an.an2cn(number), 1)
    text = replace_punctuation(text)
    return text


__REPLACE_PUNCTUATION_PATTERN = re.compile(
    "|".join(re.escape(p) for p in __REPLACE_MAP)
)
__NON_CHINESE_PATTERN = re.compile(r"[^\u4e00-\u9fa5" + "".join(PUNCTUATIONS) + r"]+")


def replace_punctuation(text: str) -> str:

    text = text.replace("嗯", "恩").replace("呣", "母")

    replaced_text = __REPLACE_PUNCTUATION_PATTERN.sub(
        lambda x: __REPLACE_MAP[x.group()], text
    )

    replaced_text = __NON_CHINESE_PATTERN.sub("", replaced_text)

    return replaced_text
