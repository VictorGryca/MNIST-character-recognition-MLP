class SGD:
    def __init__(self, Lr):
        self.Lr = Lr

    def atualizar(self, W, b, dw, db):
        """otimizador de gradient descent padronizado"""
        W = W - self.Lr * dw
        b = b - self.Lr * db
        return W, b
    