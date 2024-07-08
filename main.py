#!/usr/bin/python3

# import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score
from torchvision.datasets import MNIST
from tqdm import tqdm

from nanograd.autograd import Tensor
from nanograd.optim import SGD
from nanograd.utils import crossentropy_loss, softmax, uniform

def init_model():

    N = HEIGHT * WIDTH
    scale = lambda f_in: (3 / f_in) ** 0.5

    w1 = Tensor(uniform(lower=-scale(N), upper=scale(N), shape=(N,N)), _test=TEST)
    b1 = Tensor(torch.zeros(N), _test=TEST)

    w2 = Tensor(uniform(lower=-scale(N), upper=scale(N), shape=(N,N)), _test=TEST)
    b2 = Tensor(torch.zeros(N), _test=TEST)

    w3 = Tensor(uniform(lower=-scale(N_CLASSES), upper=scale(N_CLASSES), shape=(N,N_CLASSES)), _test=TEST)
    b3 = Tensor(torch.zeros(N_CLASSES), _test=TEST)

    def model(x,y):

        h1 = (x.matmul(w1) + b1).relu()
        h2 = (h1.matmul(w2) + b2).relu()
        logits = h2.matmul(w3) + b3
        probs = softmax(logits, 1)
        loss = crossentropy_loss(y, probs, N_CLASSES)
        return (loss, probs)

    return model, dict(w1=w1,b1=b1,w2=w2,b2=b2,w3=w3,b3=b3)

if __name__ == "__main__":

    DEVICE = "cpu"
    EPOCHS = 5
    BATCH_SIZE=64
    HEIGHT=28
    WIDTH=28
    N_CLASSES = 10
    TEST = False
    TEST_SPLIT = 0.2

    # Dataset
    dataset = MNIST(root="/home/paradox/Desktop/ai/nanograd",download=True)

    data = dataset.train_data.float().reshape(-1, HEIGHT * WIDTH).to(DEVICE) / (255 / 2)
    labels = dataset.train_labels.long().to(DEVICE)

    idx = int(data.shape[0] * (1 - TEST_SPLIT))

    train_data = data[:idx]
    train_labels = labels[:idx]
    
    test_data = data[idx:]
    test_labels = labels[idx:]

    # Initialize Params

    model,params = init_model()
    # NOTE: Why the reducing parameters doesn't reduce the accuracy??????
    optimizer = SGD([*params.values()], lr=2e-2)

    for epoch in range(EPOCHS):

        print("\n")
        
        training_loss = []
        validation_loss = []
        accuracy = []

        for batch_idx in (pbar := tqdm(range(0, len(train_data), BATCH_SIZE))):

            x = Tensor(train_data[batch_idx: batch_idx + BATCH_SIZE], requires_grad=False, _test=TEST)
            y = train_labels[batch_idx: batch_idx + BATCH_SIZE] 

            (loss, _) = model(x,y.tolist())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            training_loss.append(float(loss.value)) 

            pbar.set_description(f"epoch: {epoch}, average_training_loss: {np.mean(training_loss):.4f}, loss: {loss.value.item():.4f}")

        for batch_idx in (pbar := tqdm(range(0, len(test_data), BATCH_SIZE))):
            
            x = Tensor(test_data[batch_idx: batch_idx + BATCH_SIZE])
            y = test_labels[batch_idx: batch_idx + BATCH_SIZE] 
        
            (loss, probs) = model(x, y.tolist())
            validation_loss.append(float(loss.value)) 

            preds = probs.value.argmax(dim=-1).numpy()
            acc = accuracy_score(y.cpu().numpy(), preds)
            accuracy.append(acc)
                
            pbar.set_description(f"epoch: {epoch}, average_validation_loss: {np.mean(validation_loss):.4f}, accuracy: {np.mean(accuracy):.4f}")

        # plt.plot(np.arange(len(training_loss)), training_loss, color="lightgreen")
        # plt.show()
