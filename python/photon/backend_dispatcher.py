import types
# CPU always loaded
from . import backend_cpu

# Try load the gpu backend, if not compiled (on non-gpu machine) catch error

HAS_GPU = False

try:
    from . import backend_gpu
    HAS_GPU = True
except ImportError:
    pass

def get_backend(device: str) -> types.ModuleType:
    """
    Parse and return the user's chosen backend device.

    CPU backend is unconditionally returned if requested as always compiled. 
    GPU backend checks for backend_gpu being available/compiled first.

    Parameters:
        device: string indicating cpu or gpu backend. "cpu" or "cuda"
    Returns:
        backend_cpu or backend_gpu module 
    """
    if device == "cpu":
        return backend_cpu
    elif device == "cuda":
        # Only return if gpu available
        if not HAS_GPU:
            raise RuntimeError("No gpu backend is available. Make sure NVCC 12.8+ is installed and GPU is available")
        return backend_gpu
    else:
        raise ValueError("Invalid backend device provided")
        
        

