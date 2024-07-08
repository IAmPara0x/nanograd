
from nanograd.autograd import Tensor

class SGD:
    def __init__(self, parameters: list[Tensor], lr=1e-3):
        self.parameters = parameters
        self.lr = lr

    def zero_grad(self):
        for param in self.parameters:
            param.grad[param != 0] = 0

    def step(self):
        for param in self.parameters:
            param.value -= self.lr * param.grad
