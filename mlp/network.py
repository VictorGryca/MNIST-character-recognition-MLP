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
    

    def backprop(self, x, y_true):
        # delta da saída: softmax + cross-entropy simplifica para a3 - y
        delta3 = self.a3 - y_true

        # gradientes da camada de saida
        dW3 = np.outer(delta3, self.a2)
        db3 = delta3

        # delta da camada 2
        delta2 = (self.W3.T @ delta3) * self.act.relu_derivative(self.z2)

        # gradientes da camada 2
        dW2 = np.outer(delta2, self.a1)
        db2 = delta2

        # delta da camada 1
        delta1 = (self.W2.T @ delta2) * self.act.relu_derivative(self.z1)

        # gradientes da camada 1
        dW1 = np.outer(delta1, x)
        db1 = delta1

        # atualiza todos os pesos
        self.W3, self.b3 = self.optimizer.update(self.W3, self.b3, dW3, db3)
        self.W2, self.b2 = self.optimizer.update(self.W2, self.b2, dW2, db2)
        self.W1, self.b1 = self.optimizer.update(self.W1, self.b1, dW1, db1)
