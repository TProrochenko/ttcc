class Tokenizer:
    def __init__(self):
        self.UNK = 0
        self.BOS = 2
        self.EOS = 3

    def encode(self, s: str, bos: bool = True, eos: bool = True) -> list[int]:
        t = [ord(x) if ord(x) < 128 else self.UNK for x in s]
        if bos:
            t = [self.BOS] + t
        if eos:
            t = t + [self.EOS]
        return t

    @staticmethod
    def decode(t: list[int]) -> str:
        s = "".join(chr(x) for x in t)
        return s
