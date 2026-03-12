class DualFFN:
    def __init__(self, hidden_dim):
        self.linear1 = Linear(hidden_dim, hidden_dim * 4)
        self.linear2 = Linear(hidden_dim * 4, hidden_dim)

    def __call__(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        return x