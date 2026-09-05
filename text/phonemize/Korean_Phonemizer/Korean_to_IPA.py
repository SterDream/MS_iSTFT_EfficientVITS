import re
from functools import lru_cache
from jamo import h2j, j2hcj
from g2pk2 import G2p

HANGUL = str.maketrans({
    # 子音
    "ㄱ":"g", "ㄲ":"kʼ", "ㅋ":"kʰ",
    "ㄷ":"d", "ㄸ":"tʼ", "ㅌ":"tʰ",
    "ㅂ":"b", "ㅃ":"pʼ", "ㅍ":"pʰ",
    "ㅅ":"sʰ", "ㅆ":"sʼ",
    "ㅈ":"d͡ʑ", "ㅊ":"t͡ɕʰ", "ㅉ":"t͡ɕʼ",
    "ㅁ":"m", "ㄴ":"n", "ㅇ":"ŋ",
    "ㄹ":"ɾ",
    "ㅎ":"h",
    # 二重子音
    "ㄳ":"ks",
    "ㅄ":"bs",
    "ㄵ":"nd͡ʑ", "ㄶ":"n",
    "ㄺ":"rk", "ㄻ":"rm", "ㄼ":"rb", "ㄽ":"rs", "ㄾ":"rt", "ㄿ":"rp", "ㅀ":"r",
    # 母音
    "ㅑ":"j͡a", "ㅕ":"j͡ʌ", "ㅠ":"j͡u", "ㅛ":"j͡o",
    "ㅐ":"ɛ", "ㅔ":"e", "ㅒ":"j͡ɛ̝", "ㅖ":"j͡e",
    "ㅏ":"a", "ㅗ":"o", "ㅓ":"ʌ",
    "ㅣ":"i",
    "ㅜ":"u", "ㅡ":"ɯ",
    # 二重母音
    "ㅘ":"w͡a", "ㅝ":"w͡ʌ",
    "ㅚ":"ø", "ㅞ":"w͡e", "ㅙ":"w͡ɛ",
    "ㅟ":"w͡i", "ㅢ":"ɰ͡i",
})
MOEUM = set([
    # 母音
    "ㅑ", "ㅕ", "ㅠ", "ㅛ",
    "ㅐ", "ㅔ", "ㅒ", "ㅖ",
    "ㅏ", "ㅗ", "ㅓ",
    "ㅣ",
    "ㅜ", "ㅡ",
    # 二重母音
    "ㅘ", "ㅝ",
    "ㅚ", "ㅞ", "ㅙ",
    "ㅟ", "ㅢ",
])
WORD_INITIAL_MAP = str.maketrans({"b":"b̥", "d":"d̥", "g":"k"})
# for moeum
MOEUM_STR = "".join(MOEUM)
RE_SILENT_IEUNG = re.compile(rf'ㅇ(?=[{MOEUM_STR}])')
# for ㄹ
RE_LIQUID = re.compile(r"ɾ(?=[^aeiouʌɯ])|ɾ$")
# 
BASE = 0xAC00
CHOSUNG = 588
JUNGSUNG = 28

_g2p = G2p()

@lru_cache(maxsize=4096)
def _convert(text: str, g2p) -> tuple:
        # convert
        text = g2p(text, descriptive=True)
        # 모음 자음을 분리
        text = j2hcj(h2j(text))
        # 받침이 아닌 ㅇ을 지움
        text = RE_SILENT_IEUNG.sub('', text)
        # 한글 to IPA
        text = text.translate(HANGUL)
        # 단어 머리 ㄱㅂㅈㄷ을 변환(g b d͡ʑ d -> k p t͡ɕ t)
        if text[:3] == "d͡ʑ":
            text = "t͡ɕ" + text[3:]
        elif text[:1] in "bdg":
            text = text[:1].translate(WORD_INITIAL_MAP) + text[1:]
        # 어중의 ㄹ 발음을 ɾ -> l 
        text = RE_LIQUID.sub("l", text)
        return [text]

class G2P_Korean_to_Phoneme():
    def __init__(self):
        self.g2p = _g2p
    
    def decompose_char(ch):
        code = ord(ch)
        if 0xAC00 <= code <= 0xD7A3:
            code -= BASE
            cho = code // CHOSUNG
            jung = (code % CHOSUNG) // JUNGSUNG
            jong = code % JUNGSUNG
            return cho, jung, jong
        return ch
    
    def __call__(self, text):
        return _convert(text, self.g2p)


if __name__ == "__main__":
    h = G2P_Korean_to_Phoneme()
    a = h('사랑해.')
    print(a)
    #　--> t͡ɕʰilɾipaupʰeipʰʌɾɯlpausʰegeesʰʌpaugad͡ʑaŋpaud͡ʑoahanɯnpaumind͡ʑogɯnpauhanguginidapau