#!/usr/bin/python3

from nanograd.optim import SGD
from nanograd.engine import Tensor
from nanograd.utils import softmax,crossentropy_loss, uniform
import numpy as np

from sklearn.metrics import accuracy_score

import torch
from torchvision.datasets import MNIST
from tqdm import tqdm

def init_model():
    N = HEIGHT * WIDTH

    w1 = uniform(lower=-(1 / (N + N)), upper=(1 / (N + N)), shape=(N,N)); w1._test = TEST
    b1 = Tensor(torch.zeros(N), _test=TEST)

    w2 = uniform(lower=-(1 / (N + N)), upper=(1 / (N + N)), shape=(N,N)); w2._test = TEST
    b2 = Tensor(torch.zeros(N), _test=TEST)

    w3 = uniform(lower=-(1 / N_CLASSES), upper=(1 / N_CLASSES), shape=(N,N_CLASSES)); w2._test = TEST
    b3 = Tensor(torch.zeros(N_CLASSES), _test=TEST)

    def model(x,y):

        h1 = (x.matmul(w1) + b1).relu()
        h2 = (h1.matmul(w2) + b2).relu()
        logits = h2.matmul(w3) + b3
        probs = softmax(logits, 1)
        loss = crossentropy_loss(y, probs, N_CLASSES)
        return (loss, probs)

    return model,[w1,b1,w2,b2]

if __name__ == "__main__":

    DEVICE = "cpu"
    EPOCHS = 5
    BATCH_SIZE=64
    HEIGHT=28
    WIDTH=28
    N_CLASSES = 10
    TEST = False

    # Dataset
    dataset = MNIST(root="/home/paradox/Desktop/ai/pygrad",download=True)

    train_data =  dataset.train_data.float().reshape(-1, HEIGHT * WIDTH).to(DEVICE)
    train_labels = dataset.train_labels.long().to(DEVICE)
    
    test_data = dataset.test_data.float().reshape(-1, HEIGHT * WIDTH).to(DEVICE)
    test_labels = dataset.test_labels.long().to(DEVICE)

    # Initialize Params

    model,params = init_model()
    optimizer = SGD(params, lr=1e-3)

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

