import numpy as np
from collections import deque

from .backend_dispatcher import get_backend
from .ops import *
from typing import Self

class Tensor:
    """
    Tensor Node class in the computational graph.

    Tracks its place in the computations + provides interface for gradient calculation and data access.

    Attributes:
        grad: Gradient tensor or None if isolated
        _data: Handle to backend NDArray object that manages the data
        _inputs: input nodes used to construct this tensor
        _op: operation used to construct this tensor
        _requires_grad: Whether gradient tracking is enabled
        _device: device holding the Tensor data
    """
    def __init__(self, data, device: str = "cpu", requires_grad: bool = True) -> None:
        """
        User facing init method to construct new tensors

        Parameters:
            data: tensor data
            device: device type string
            requires_grad: disable/enable gradient tracking for the comp graph
        """
        self._requires_grad = requires_grad

        # define the device + get backend
        self._device = device
        backend = get_backend(device)

        # (Clunky, refactor later)
        if isinstance(data, Tensor):
            # copy the data + shape via numpy (for now)
            raw_np_arr = data._data.numpy()
            # allocates new handle like torch
            self._data = backend.NDArray(raw_np_arr.flatten().tolist(),list(raw_np_arr.shape()))
        elif isinstance(data, (int,float)):
            # Convert to list and create new 1d tensor
            self._data = backend.NDArray([data])
        elif isinstance(data, list):
            self._data = backend.NDArray(data)
        elif isinstance(data, np.ndarray):
            self._data = backend.NDArray(data.flatten().tolist(),list(data.shape()))

        # Initialise as isolated node, no gradient
        self._inputs = []
        self._op = None
        self.grad = None

    @property
    def shape(self):
        return self._data.shape
    
    @property 
    def strides(self):
        return self._data.strides
    
    def _topo(self):
        """
        Returns the topo ordering of the computational graph 
        for the backward pass using iterative Kahn's algorithm.

        Wanted to avoid recursion for large networks but will benchmark later

        Returns:
            list: ordering of all nodes
        """
        dependencies = {}
        visited = set()
        stack = [self]
        
        while stack:
            curr = stack.pop()
            if curr not in visited:
                visited.add(curr)
            
                if curr not in dependencies:
                    dependencies[curr] = 0
                    
                # Increment deps of all inputs
                for node in curr._inputs:
                    if node not in dependencies:
                        dependencies[node] = 0
                    dependencies[node] += 1
                    stack.append(node)

        topo_order = []
        
        # The final node has 0 deps 
        queue = deque([self])
        
        # Pop off final node and decrement deps of inputs, 
        # Iteratively adding 0 dep nodes.
        while queue:
            curr = queue.popleft()
            topo_order.append(curr)
            
            for node in curr._inputs:
                dependencies[node] -= 1
        
                if dependencies[node] == 0:
                    queue.append(node)
                    
        return topo_order
    
    def backward(self):
        """
        Compute backward pass, populates gradients.
        """
        if not self._requires_grad:
            raise RuntimeError("Gradient computation disabled for this tensor")
        
        # Initialise output gradients as ones (via numpy for now)
        self.grad = Tensor(np.ones(self.shape), device=self._device, requires_grad=False)
        topo_order = self._topo()
        # Backprop gradients in topological order
        for node in topo_order:
            # Skip leaf nodes
            if node._op is None:
                continue 
            
            # Compute the grads to prop back
            input_grads = node._op.gradient(node.grad, node)
            
            # init or accumulate input grads
            for i, input_node in enumerate(node._inputs):
                if not input_node._requires_grad:
                    continue

                if input_node.grad is None:
                    input_node.grad = input_grads[i]
                else:
                    input_node.grad = input_node.grad + input_grads[i]
    def log(self):
        ...


    
    def __add__(self, other):
        if isinstance(other, (int,float)):
            return ScalarAdd(other)(self)
        elif isinstance(other, Tensor):
            return EwiseAdd()(self, other)
        
    def __sub__(self, other):
        if isinstance(other, (int,float)):
            return ScalarSub(other)(self)
        elif isinstance(other, Tensor):
            return EwiseSub()(self, other)
    
    def __mul__(self, other):
        if isinstance(other, (int,float)):
            return ScalarMul(other)(self)
        elif isinstance(other, Tensor):
            return EwiseMul()(self, other)
    
    def __truediv__(self, other):
        if isinstance(other, (int,float)):
            return ScalarDiv(other)(self)
        elif isinstance(other, Tensor):
            return EwiseDiv()(self, other)
        
    def __pow__(self, other):
        if isinstance(other, (int,float)):
            return ScalarPow(other)(self)
        elif isinstance(other, Tensor):
            return EwisePow()(self, other)
    

    
    @classmethod
    def _from_ndarray(cls, data, device: str = "cpu", requires_grad: bool = True, inputs: list[Self] = [], op: Op | None = None):
        """
        Alternate constructor taking in raw backend NDArray data, and comp graph info.
        """
        tens = cls.__new__(cls)
        tens._data = data       
        tens._device = device
        tens._requires_grad = requires_grad
        tens._inputs = inputs or []
        tens._op = op

        tens.grad = None

        return tens
    
    def __str__(self):
        # Printing offloaded to numpy for now
        if self._device == "cpu":
            return str(np.array(self._data))
        elif self._device == "cuda":
            return str(self._data.numpy())
        raise RuntimeError("Impossible to print array as handle is invalid")
    
    
    

    