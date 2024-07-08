# Nanograd

Nanograd is inspired by micrograd but works with tensors! The entire autograd engine is implemented in under ~280 lines of code. Nanograd leverages PyTorch tensors for matrix operations like multiplication and addition, making it straightforward to verify the gradients calculated by Nanograd. Instead of calling loss.backward(), you can use loss.check_with_pytorch() to ensure all gradients in the compute graph match those calculated by PyTorch.

# Example

Nanograd has a Tensor class similar to PyTorch's tensor, with an almost identical API. However, Nanograd computes gradients on its own.

```python

import torch
from nanograd import Tensor  # Assuming nanograd is correctly imported

# Enable testing mode
TEST = True

# Create input tensor without gradient tracking
input = Tensor(torch.randn(4, 8), requires_grad=False, _test=TEST)

# Create weight and bias tensors for the first layer
w1 = Tensor(torch.randn(8, 16), _test=TEST)
b1 = Tensor(torch.zeros(16), _test=TEST)

# Perform forward pass
hidden1 = (input.matmul(w1) + b1).relu().sum()

# Use hidden1.backward() or hidden1.check_with_pytorch() to verify gradients
hidden1.check_with_pytorch()

```

Notice how you can use `hidden1.backward()` or `hidden1.check_with_pytorch(). The latter verifies all gradients calculated by Nanograd against those calculated by PyTorch. This is one of the advantages of wrapping around PyTorch tensors, as it allows for both matrix operations and simultaneous verification of gradient calculations.

A detailed example of training on the MNIST dataset can be found in [mnist.py](./examples/mnist.py). Using only stochastic gradient descent and a simple neural network with a hidden layer, you can achieve an accuracy of 0.9715!
