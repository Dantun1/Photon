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
        return out_grad, -1 * out_grad

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
        grad_b = (-out_node / b) * out_grad 
        
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

#Implement the unary ops

class Log(Op):
    ...