import photon.backend_gpu as gpu 
import numpy as np
import numpy.testing as npt
import pytest

# GPU NDArray tests

def test_constructs_ndarray_with_data_and_shape():
    array = gpu.NDArray([1,2,3,4],[2,2])
    expected = np.array([1,2,3,4]).reshape([2,2])
    npt.assert_allclose(array.numpy(), expected) 

def test_constructs_ndarray_with_data():
    array = gpu.NDArray([1,2,3,4])
    
    npt.assert_allclose(array.numpy(), np.array([1,2,3,4]))

