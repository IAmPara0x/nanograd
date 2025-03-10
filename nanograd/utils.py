from nanograd.autograd import Tensor
import torch
import torch.nn.functional as F

def softmax(logits: Tensor, dim: int):
    norm_logits = logits - logits.max(dim, keepdim=True)
    probs = norm_logits.exp() / norm_logits.exp().sum(dim, keepdim=True)
    return probs

def crossentropy_loss(labels: list[int], probs: Tensor, n_classes):

    ohe_labels = Tensor(F.one_hot(torch.tensor(labels), n_classes).float(), _test=probs._test)
    loss = (ohe_labels * probs).sum(1).log().sum() / Tensor(float(len(labels)))
    return -loss

def cmp(label: str, x: Tensor):
    if torch.all(x.grad == x.value.grad):
        print(f"grads of tensor {label} matches with pytorch!")
    else:
        raise ValueError(f"grads of tensor {label} matches with pytorch!")

def uniform(lower: float, upper: float, shape) -> torch.Tensor:
    return (lower - upper) * torch.rand(shape) + upper
