from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api

class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)

def expand_dim(axes, from_shape, to_shape):
    ex_shape = [1] * len(to_shape)
    axis = []
    if axes is not None:
        axis = [axes] if not isinstance(axes, tuple) else list(axes)
    n = 0
    for i, _ in enumerate(ex_shape):
        if axes is not None and (i not in axis):
            ex_shape[i] = from_shape[n]
            n += 1
    return ex_shape

class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        max = array_api.max(Z, axis=self.axes, keepdims=True)
        max_not_keep = array_api.max(Z, axis=self.axes)
        return array_api.log(array_api.sum(array_api.exp(Z - max), axis=self.axes)) + max_not_keep
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        Z = node.inputs[0]
        maxz = array_api.max(Z.realize_cached_data(), self.axes)
        exshape = expand_dim(self.axes, maxz.shape, Z.shape)
        x = Z - array_api.broadcast_to(array_api.reshape(maxz, exshape), Z.shape)
        return broadcast_to(reshape(out_grad / summation(exp(x), self.axes), exshape), Z.shape) * exp(x)
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

