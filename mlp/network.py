import numpy as np
from mlp.activations import Activations
from mlp.losses import Losses
from mlp.optimizers import SGD


class MLP:
    def __init__(self, lr=0.01):
        self.act = Activations()
        self.loss_fn = Losses()
        self.optimizer = SGD(lr)

        # pesos e biases inicializados pequenos e aleatórios, diferente do feito no notebook (que foi com range)
        self.W1 = np.random.randn(128, 784) * 0.01
        self.b1 = np.zeros(128)

        self.W2 = np.random.randn(128, 128) * 0.01
        self.b2 = np.zeros(128)

        self.W3 = np.random.randn(10, 128) * 0.01
        self.b3 = np.zeros(10)


    def forward(self, x):
        # camada 1
        self.z1 = self.W1 @ x + self.b1
        self.a1 = self.act.relu(self.z1)

        # camada 2
        self.z2 = self.W2 @ self.a1 + self.b2
        self.a2 = self.act.relu(self.z2)

        # camada de saída
        self.z3 = self.W3 @ self.a2 + self.b3
        self.a3 = self.act.softmax(self.z3)

        return self.a3
