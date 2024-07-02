#!/usr/bin/python3

import torch
from nanograd.engine import Tensor
from nanograd.utils import softmax,cmp,crossentropy_loss

if __name__ == "__main__":

    DEVICE = "cuda:0"
    EPOCHS = 2
    BATCH_SIZE=4
    HEIGHT=28
    WIDTH=28
    N_CLASSES = 10

    scale_factor = ((HEIGHT * WIDTH) ** -0.5)

    xTr = Tensor(torch.randn(BATCH_SIZE, HEIGHT * WIDTH) * scale_factor, _test=True)

    w1 = Tensor(torch.randn(HEIGHT * WIDTH, HEIGHT * WIDTH) * scale_factor, _test=True)
    b1 = Tensor(torch.randn(HEIGHT * WIDTH) * scale_factor, _test=True)

    w2 = Tensor(torch.randn(HEIGHT * WIDTH, N_CLASSES) * scale_factor, _test=True)
    b2 = Tensor(torch.randn(N_CLASSES), _test=True)

    h = (xTr.matmul(w1) + b1).relu()
    logits = (h.matmul(w2) + b2)

    probs = softmax(logits, 1)
    loss = crossentropy_loss([0,1,2,1], probs, 10)
    loss.check_with_pytorch()

    print(f"{loss=}")

    for name,tensor in [("w1",w1),("b1",b1),("w2",w2),("b2",b2),("h",h),("logits", logits),("probs", probs)]:
        cmp(name,tensor)
