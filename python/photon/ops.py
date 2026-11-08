from typing import Any

class Op:
    def __call__(self, *args):
        """
        Forward pass through a generic op. 

        Each op implements its own  compute and gradient method.
        """
        from .tensor import Tensor
        raw_inputs = [arg._data for arg in args]

        # Run computation directly on ndarrays.
        out_ndarray = self.compute(*raw_inputs)
        
        # req grad if any input needs grad
        req_grad = any(arg._requires_grad for arg in args)
        
        # Construct new node in graph
        return Tensor._from_ndarray(
            out_ndarray, 
            device=args[0]._device, 
            requires_grad=req_grad, 
            inputs=list(args),
            op=self 
            )
    
    def compute(self, *args):
        """Computes the forward pass"""
        raise NotImplementedError

    def gradient(self, out_grad, out_node):
        """Computes the backward pass"""
        raise NotImplementedError


# EWISE/SCALAR OPS


class EwiseAdd(Op):
    def compute(self, a_ndarr, b_ndarr):
        return a_ndarr + b_ndarr
    
    def gradient(self, out_grad, out_node):
        return out_grad, out_grad

class ScalarAdd(Op):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a_ndarr):
        return a_ndarr + self.scalar
    
    def gradient(self, out_grad, out_node):
        return (out_grad,)

class EwiseSub(Op):
    def compute(self, a_ndarr, b_ndarr):
        return a_ndarr - b_ndarr
    
    def gradient(self, out_grad, out_node):
        return out_grad, Negate()(out_grad)

class ScalarSub(Op):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a_ndarr):
        return a_ndarr - self.scalar
    
    def gradient(self, out_grad, out_node):
        return (out_grad,)
    
class EwiseMul(Op):
    def compute(self, a_ndarr, b_ndarr):
        return a_ndarr * b_ndarr
    
    def gradient(self, out_grad, out_node):
        return out_node._inputs[1] * out_grad, out_node._inputs[0] * out_grad
    
class ScalarMul(Op):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a_ndarr):
        return a_ndarr * self.scalar
    
    def gradient(self, out_grad, out_node):
        return (out_grad * self.scalar,)

class EwiseDiv(Op):
    def compute(self, a_ndarr, b_ndarr):
        return a_ndarr / b_ndarr
    
    def gradient(self, out_grad, out_node):
        b = out_node._inputs[1]
        grad_a = (1 / b) * out_grad
        # Reuse output for gradient of b
        grad_b = Negate()(out_node / b) * out_grad 
        
        return grad_a, grad_b
    
class ScalarDiv(Op):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a_ndarr):
        return a_ndarr / self.scalar
    
    def gradient(self, out_grad, out_node):
        return (out_grad / self.scalar,)
    

class EwisePow(Op):
    def compute(self, a_ndarr, b_ndarr):
        return a_ndarr ** b_ndarr
    
    def gradient(self, out_grad, out_node):
        #input tensors
        a = out_node._inputs[0]
        b = out_node._inputs[1]
        
        return b * a ** (b-1) * out_grad, a.log() * out_node * out_grad
    
class ScalarPow(Op):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a_ndarr):
        return a_ndarr ** self.scalar
    
    def gradient(self, out_grad, out_node):
        a = out_node._inputs[0]
        return (self.scalar * a ** (self.scalar - 1) * out_grad,)


# UNARY OPS


class Negate(Op):
    def compute(self, a_ndarr):
        return a_ndarr.neg()

    def gradient(self, out_grad, out_node):
        return (Negate()(out_grad),)

class Log(Op):
    def compute(self, a_ndarr):
        return a_ndarr.log()
    
    def gradient(self, out_grad, out_node):
        return (out_grad  / out_node._inputs[0],)


class Exp(Op):
    def compute(self, a_ndarr):
        return a_ndarr.exp()
    
    def gradient(self, out_grad, out_node):
        return (out_grad * out_node,)


class Tanh(Op):
    def compute(self, a_ndarr):
        return a_ndarr.tanh()
    
    def gradient(self, out_grad, out_node):
        return (out_grad - out_grad * out_node * out_node,)


# REDUCTION 

class Summation(Op):
    def __init__(self, axes = None, keepdims = False):
        self.axes = list(axes) if axes is not None else None
        self.keepdims = keepdims

    def compute(self, a_ndarr):
        rank = len(a_ndarr.shape)
        # reduce all by default 
        axes = self.axes if self.axes is not None else list(range(rank))

        return a_ndarr.sum(axes, self.keepdims)

    def gradient(self, out_grad, out_node):
        in_shape = out_node._inputs[0].shape
        axes = self.axes if self.axes is not None else list(range(len(in_shape)))
        temp = [1 if i in axes else in_shape[i] for i in list(range(len(in_shape)))]
        grad = out_grad
        if out_node.shape != temp:
            grad = Reshape(temp)(grad)
        return (BroadcastTo(in_shape)(grad), )

# VIEWS


class Reshape(Op):
    def __init__(self, new_shape):
        self.new_shape = list(new_shape)

    def compute(self, a_ndarr):
        return a_ndarr.reshape(self.new_shape)

    def gradient(self, out_grad, out_node):
        return (Reshape(list(out_node._inputs[0].shape))(out_grad),)



class Transpose(Op):
    def __init__(self, axes=None):
        self.axes = list(axes) if axes is not None else None

    def compute(self, a_ndarr):
        rank = len(a_ndarr.shape)
        axes = self.axes if self.axes is not None else list(reversed(range(rank)))
        self._applied_axes = axes
        return a_ndarr.transpose(axes)

    def gradient(self, out_grad, out_node):
        axes = self._applied_axes
        inv = [0] * len(axes)
        for i, p in enumerate(axes):
            inv[p] = i
        return (Transpose(inv)(out_grad),)

class BroadcastTo(Op):
    def __init__(self, shape):
        self.shape = list(shape)

    def compute(self, a_ndarr):
        return a_ndarr.broadcast(self.shape)

    def gradient(self, out_grad, out_node):
        return (reduce_grad_to_shape(out_grad, out_node._inputs[0].shape),)


class MatMul(Op):
    def compute(self, a_ndarr, b_ndarr):
        return a_ndarr @ b_ndarr

    def gradient(self, out_grad, out_node):
        a, b = out_node._inputs[0], out_node._inputs[1]

        def swap_last_two(t):
            rank = len(t.shape)
            axes = list(range(rank))
            axes[-1], axes[-2] = axes[-2], axes[-1]
            return Transpose(axes)(t)

        grad_a = MatMul()(out_grad, swap_last_two(b))
        grad_b = MatMul()(swap_last_two(a), out_grad)
        return grad_a, grad_b


def reduce_grad_to_shape(grad, shape):
    """Reverse implicit broadcasting by summation over broadcasted dims"""
    shape = list(shape)
    gshape = list(grad.shape)
    if gshape == shape:
        return grad

    ndiff = len(gshape) - len(shape)
    if ndiff > 0:
        grad = Summation(axes=list(range(ndiff)))(grad)

    gshape = list(grad.shape)
    axes = [i for i in range(len(shape)) if shape[i] == 1 and gshape[i] != 1]
    if axes:
        grad = Summation(axes=axes, keepdims=True)(grad)
    
    
    # Remove keepdims if necessary
    if list(grad.shape) != shape:
        grad = Reshape(shape)(grad)
    return grad


