#!/usr/bin/python

import torch
from nanograd.autograd import Tensor
from nanograd.utils import softmax, crossentropy_loss, uniform

if __name__ == "__main__":

    HEIGHT=28
    WIDTH=28
    N_CLASSES = 10
    TEST = True
    BATCH_SIZE = 8

    N = HEIGHT * WIDTH


    # Here we do a forward pass on a random tensor of same shape as an mnist image.
    # And then we check whether the gradients calculated by nanograd and pytorch match.  

    N = HEIGHT * WIDTH
    scale = lambda f_in: (3 / f_in) ** 0.5

    w1 = Tensor(uniform(lower=-scale(N), upper=scale(N), shape=(N,N)), _test=TEST)
    b1 = Tensor(torch.zeros(N), _test=TEST)

    w2 = Tensor(uniform(lower=-scale(N), upper=scale(N), shape=(N,N)), _test=TEST)
    b2 = Tensor(torch.zeros(N), _test=TEST)

    w3 = Tensor(uniform(lower=-scale(N_CLASSES), upper=scale(N_CLASSES), shape=(N,N_CLASSES)), _test=TEST)
    b3 = Tensor(torch.zeros(N_CLASSES), _test=TEST)


    # Generate a random tensor of same shape as an mnist image.
    x = Tensor(torch.randn(BATCH_SIZE, N), _test=TEST, requires_grad=False)
    y = torch.randint(low=0,high=9,size=(BATCH_SIZE,)).tolist()


    # Forward pass
    h1 = (x.matmul(w1) + b1).relu()
    h2 = (h1.matmul(w2) + b2).relu()
    logits = h2.matmul(w3) + b3

    probs = softmax(logits, 1)
    loss = crossentropy_loss(y, probs, N_CLASSES)
    loss.check_with_pytorch()


    # We can further check if the grads are actually same
    for name,parameter in [("w1", w1),("b1", b1),("w2", w2),("b2", b2),("w3", w3),("b3", b3)]:

        if torch.allclose(parameter.grad, parameter.value.grad):
            print(f"Grads of {name} matches with pytorch :0")
        else:
            raise ValueError(f"Grads of {name} doesn't match with pytorch :( ")
