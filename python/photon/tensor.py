import numpy as np

from .backend_dispatcher import get_backend
from .ops import Op

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

        self._data = backend.NDArray(data)

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
    
    def __str__(self):
        if self._device == "cpu":
            return str(np.array(self._data))
        elif self._device == "cuda":
            return str(self._data.numpy())
        raise RuntimeError("Impossible to print array as handle is invalid")
    

    