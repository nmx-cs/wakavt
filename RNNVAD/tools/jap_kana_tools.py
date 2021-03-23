# -*- coding:utf-8 -*-

# Unicode 13.0, http://www.unicode.org/charts/


def is_hiragana(char):
    # Hiragana
    # Small Kana Extension
    code = ord(char)
    return 0x3041 <= code <= 0x3096 \
           or 0x3099 <= code <= 0x309f \
           or 0x1b150 <= code <= 0x1b152


def is_katakana(char):
    # Katakana
    # Small Kana Extension
    # Katakana Phonetic Extensions
    # Halfwidth Katakana
    code = ord(char)
    return 0x30a0 <= code <= 0x30ff \
           or 0x1b164 <= code <= 0x1b167 \
           or 0x31f0 <= code <= 0x31ff \
           or 0xff66 <= code <= 0xff9f


def is_kana(char):
    return is_hiragana(char) or is_katakana(char)


def count_moras(string):
    return sum(is_kana(c) and c not in "ゃゅょゎャュョヮ" for c in string)
