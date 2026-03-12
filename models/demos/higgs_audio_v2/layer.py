class Layer:
    def __init__(self, attention, dual_ffn):
        self.attention = attention
        self.dual_ffn = dual_ffn

    def __call__(self, x):
        x = self.attention(x)
        x = self.dual_ffn(x)
        return x