"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
import math


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(
            init.kaiming_uniform(in_features, out_features, nonlinearity="relu", device=device, dtype=dtype)
        )
        if bias:
            init_value = init.kaiming_uniform(out_features, 1, nonlinearity="relu", device=device, dtype=dtype)
            init_value = init_value.reshape((1, out_features))
            self.bias = Parameter(
                init_value
            )
        else:
            self.bias = None
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        product = ops.matmul(X, self.weight)
        if self.bias is not None:
            product += self.bias
        return product
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        return ops.reshape(X, (X.shape[0], -1))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for module in self.modules:
            x = module.forward(x)
        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        one_hot = init.one_hot(logits.shape[1], y)
        logsumexp = ops.logsumexp(logits, axes=1)
        loss =  logsumexp - ops.summation(ops.multiply(logits, one_hot), axes=1)
        return ops.summation(loss / logits.shape[0], axes=0)
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(
            init.ones(dim, device=device, dtype=dtype)
        )
        self.bias = Parameter(
            init.zeros(dim, device=device, dtype=dtype)
        )
        self.running_mean = init.zeros(dim, device=device, dtype=dtype)
        self.running_var = init.ones(dim, device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            sum_x = ops.summation(x, axes=0)
            mean = sum_x / x.shape[0]
            var = ops.summation(ops.power_scalar(x - ops.broadcast_to(mean, x.shape), 2), axes=0) / x.shape[0]
            self.running_mean = self.momentum * mean.data + (1 - self.momentum) * self.running_mean.data
            self.running_var = self.momentum * var.data + (1 - self.momentum) * self.running_var.data
            mean_broadcast = ops.broadcast_to(mean, x.shape)
            x_hat = (x - mean_broadcast) / ops.broadcast_to(ops.power_scalar(var + self.eps, 0.5), x.shape)
            return ops.broadcast_to(self.weight, x_hat.shape) * x_hat + ops.broadcast_to(self.bias, x_hat.shape)
        else:
            mean = self.running_mean
            var = self.running_var
            x_hat = (x - ops.broadcast_to(mean, x.shape)) / ops.power_scalar(ops.broadcast_to(var, x.shape) + self.eps, 0.5)
            return ops.broadcast_to(self.weight, x_hat.shape) * x_hat + ops.broadcast_to(self.bias, x_hat.shape)
        ### END YOUR SOLUTION


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(
            init.ones(dim, device=device, dtype=dtype)
        )
        self.bias = Parameter(
            init.zeros(dim, device=device, dtype=dtype)
        )
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        #print("XXXXXXXXXXXXXXX", x, "\n")
        sum_x = ops.summation(x, axes=1)
        mean = sum_x / x.shape[1]
        mean = ops.reshape(mean, (x.shape[0], 1))
        mean_broadcast = ops.broadcast_to(mean, x.shape)
        var = ops.summation(ops.power_scalar(x - mean_broadcast, 2), axes=1) / x.shape[1]
        var = ops.reshape(var, (x.shape[0], 1))
        var_broadcast = ops.broadcast_to(var, x.shape)
        #print("VVVVVVVVVBBBBBB", var_broadcast, "\n")
        x_hat = (x - mean_broadcast) / ops.power_scalar(var_broadcast + self.eps, 0.5)
        return self.weight * x_hat + self.bias
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            mask = init.randb(*x.shape, p=self.p, device=x.device, dtype="float32")
            mask = mask * (1 / (1 - self.p))
            return ops.multiply(mask, x)
        else:
            return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return x + self.fn.forward(x)
        ### END YOUR SOLUTION
