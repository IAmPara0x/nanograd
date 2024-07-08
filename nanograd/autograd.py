import torch
from dataclasses import dataclass

@dataclass
class BroadcastDims:
    expanded_dims: list[int]
    replicated_dims: list[int]

class Tensor(object):

    def __init__(self, value, _childrens=[], _leaf=True, requires_grad=True, _test=False):

        if isinstance(value, torch.Tensor):
            self.value = value
        else:
            self.value = torch.tensor(value)

        self.grad = torch.zeros(self.value.shape)
        self.grad_fn = lambda: None
        self.requires_grad = requires_grad

        # NOTE: this will get invalid when the self.value is modified
        self.shape = self.value.shape

        self._childrens = _childrens
        self._label: str | None = None
        self._leaf = _leaf
        self._test = _test

        if _leaf and _test and requires_grad:
            self.value.requires_grad = True
        
        if (not _leaf) and _test:
            self.value.retain_grad()

    def __repr__(self):
        if self._label is not None:
            return f"Tensor(label={self._label}, value={self.value}, grad={self.grad})"
        else:
            return f"Tensor(value={self.value}, grad={self.grad})"

    @staticmethod
    def get_broadcasted_dims(x: torch.Tensor, y: torch.Tensor) -> tuple[BroadcastDims, BroadcastDims]:

        x_dims = list(x.shape)
        y_dims = list(y.shape)

        if x_dims == y_dims:
            return (BroadcastDims([],[]) , BroadcastDims([],[]))

        expanded_dims_x = []
        expanded_dims_y = []

        replicated_dims_x = []
        replicated_dims_y = []

        if len(x_dims) > len(y_dims):
            y_dims = [None for _ in range(len(x_dims) - len(y_dims))] + y_dims
        else:
            x_dims = [None for _ in range(len(y_dims) - len(x_dims))] + x_dims

        for dim, (dim_size_x, dim_size_y) in enumerate(zip(x_dims, y_dims)):

            if dim_size_x != dim_size_y and dim_size_y == None:
                expanded_dims_y.append(dim)
            elif dim_size_x != dim_size_y and dim_size_x == None:
                expanded_dims_x.append(dim)
            elif dim_size_x == 1:
                replicated_dims_x.append(dim)
            elif dim_size_y == 1:
                replicated_dims_y.append(dim)
            elif dim_size_x == dim_size_y:
                continue
            else:
                raise ValueError(f"tensors x and y of shape {x.shape=} and {y.shape} doesn't satisy broadcasting condition")

        return ( BroadcastDims(sorted(expanded_dims_x), sorted(replicated_dims_x))
                , BroadcastDims(sorted(expanded_dims_y), sorted(replicated_dims_y))
                )

    @staticmethod
    def isscalar(x: torch.Tensor):
        return x.shape == torch.Size([])

    def __add__(self, other: "Tensor") -> "Tensor":
        t = Tensor(self.value + other.value, _childrens=[self, other], _leaf=False, _test=self._test)

        @torch.no_grad()
        def _grad_fn():

            broadcasted_dims_self, broadcasted_dims_other = self.get_broadcasted_dims(self.value, other.value)
            if self.requires_grad: self.grad += self.reduce_dims(t.grad, broadcasted_dims_self)
            if other.requires_grad: other.grad += self.reduce_dims(t.grad, broadcasted_dims_other)

        t.grad_fn = _grad_fn
        return t

    def __mul__(self, other: "Tensor") -> "Tensor":

        t = Tensor(self.value * other.value, _childrens=[self, other], _leaf=False, _test=self._test)

        @torch.no_grad()
        def _grad_fn():

            broadcasted_dims_self, broadcasted_dims_other = self.get_broadcasted_dims(self.value, other.value)
            if self.requires_grad: self.grad += self.reduce_dims(t.grad * other.value, broadcasted_dims_self)
            if other.requires_grad: other.grad += self.reduce_dims(t.grad * self.value, broadcasted_dims_other)

        t.grad_fn = _grad_fn
        return t

    def __neg__(self) -> "Tensor":
        return Tensor(-1.0, _test=self._test, _leaf=True) * self

    def __pow__(self, n: float) -> "Tensor":

        assert isinstance(n, (float, int))

        t = Tensor(self.value ** n, _childrens=[self], _leaf=False, _test=self._test)

        @torch.no_grad()
        def _grad_fn():
            if self.requires_grad: self.grad += n * (self.value ** (n - 1)) * t.grad

        t.grad_fn = _grad_fn
        return t

    def __sub__(self, other: "Tensor") -> "Tensor":

        t = Tensor(self.value - other.value, _childrens=[self, other], _leaf=False, _test=self._test)

        @torch.no_grad()
        def _grad_fn():

            broadcasted_dims_self, broadcasted_dims_other = self.get_broadcasted_dims(self.value, other.value)
            if self.requires_grad: self.grad += self.reduce_dims(t.grad, broadcasted_dims_self)
            if other.requires_grad: other.grad += -1 * self.reduce_dims(t.grad, broadcasted_dims_other)

        t.grad_fn = _grad_fn
        return t

    def __truediv__(self, other: "Tensor") -> "Tensor":
        return self * (other ** -1)

    def backward(self):
        assert (self.value.shape == () and "Grads can be created for scalar inputs only")

        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._childrens:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = torch.tensor(1.0)
        for v in reversed(topo):
            v.grad_fn()
            if v._test and v.requires_grad:
                if not torch.allclose(v.grad, v.value.grad):
                    raise ValueError(f"grad of tensor {v} doesn't match with pytorch")

        if self._test:
            print("All grads of the tensors are in match with pytorch!")

    def check_with_pytorch(self):
        self.value.backward()
        self.backward()

    def exp(self) -> "Tensor":
        t = Tensor(self.value.exp(), _childrens=[self], _leaf=False, _test=self._test)

        @torch.no_grad()
        def _grad_fn():
            if self.requires_grad: self.grad += t.grad * t.value

        t.grad_fn = _grad_fn
        return t

    def log(self) -> "Tensor":
        t = Tensor(self.value.log(), _childrens=[self], _leaf=False, _test=self._test)

        @torch.no_grad()
        def _grad_fn():
            if self.requires_grad: self.grad += t.grad * (self.value ** -1)

        t.grad_fn = _grad_fn

        return t

    def matmul(self, other: "Tensor") -> "Tensor":
        t = Tensor(self.value @ other.value, _childrens=[self,other], _leaf=False, _test=self._test)

        @torch.no_grad()
        def _grad_fn():
            if self.requires_grad: self.grad += (t.grad @ other.value.T)
            if other.requires_grad: other.grad += (self.value.T @ t.grad)

        t.grad_fn = _grad_fn
        return t

    def max(self, dim: int | None = None, keepdim=False) -> "Tensor":

        if dim == None:
            t = Tensor(self.value.max(), _childrens=[self], _test=self._test, _leaf=False)
        else:
            t = Tensor(self.value.max(dim, keepdim=keepdim).values, _childrens=[self], _test=self._test, _leaf=False)

        @torch.no_grad()
        def _grad_fn():

            if dim == None:
                if self.requires_grad: self.grad += t.grad
                return

            indices = []

            for d,d_size in enumerate(self.value.shape):
                if d != dim:
                    indices.append(torch.arange(d_size))
                else:
                    indices.append(self.value.max(dim).indices)

            if keepdim:
                self.grad[indices] += t.grad.squeeze(dim)
            else:
                self.grad[indices] += t.grad

        t.grad_fn = _grad_fn
        return t

    @staticmethod
    def reduce_dims(x: torch.Tensor, broadcasted_dims: BroadcastDims):
        for dims in [broadcasted_dims.expanded_dims, broadcasted_dims.replicated_dims]:
            for dim in dims:
                x = x.sum(dim, keepdim=True)

        for n,dim in enumerate(broadcasted_dims.expanded_dims):
            x = x.squeeze(dim - n)
        return x

    def relu(self) -> "Tensor":

        t = Tensor(self.value.relu(), _childrens=[self], _leaf=False, _test=self._test)

        @torch.no_grad()
        def _grad_fn():
            self.grad[0 <= self.value] += t.grad[0 <= self.value]

        t.grad_fn = _grad_fn
        return t

    def sum(self, dim: int | None = None, keepdim=False):

        if dim != None:
            t = Tensor(self.value.sum(dim, keepdim=keepdim), _childrens=set([self]), _test=self._test, _leaf=False)
        else:
            t = Tensor(self.value.sum(), _childrens=set([self]), _test=self._test, _leaf=False)

        @torch.no_grad()
        def backwards():

            if dim != None and keepdim == False:
                if self.requires_grad: self.grad += t.grad.unsqueeze(dim)
            else:
                if self.requires_grad: self.grad += t.grad
            
        t.grad_fn = backwards
        return t

    def tanh(self) -> "Tensor":
        t = Tensor(self.value.tanh(), _childrens=[self], _leaf=False, _test=self._test)

        @torch.no_grad()
        def _grad_fn():
            self.grad += (1 - t.value ** 2) * t.grad

        t.grad_fn = _grad_fn
        return t
