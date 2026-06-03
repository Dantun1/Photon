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


import photon.backend_gpu as gpu
import numpy as np
import numpy.testing as npt
import pytest

# ---------------------------------------------------------
# Construction Tests
# ---------------------------------------------------------

def test_constructs_ndarray_with_data_and_shape():
    array = gpu.NDArray([1, 2, 3, 4], [2, 2])
    expected = np.array([1, 2, 3, 4], dtype=np.float32).reshape([2, 2])
    npt.assert_allclose(array.numpy(), expected)

def test_constructs_ndarray_with_data():
    array = gpu.NDArray([1, 2, 3, 4])
    expected = np.array([1, 2, 3, 4], dtype=np.float32)
    npt.assert_allclose(array.numpy(), expected)

def test_reshape():
    data = [1, 2, 3, 4, 5, 6]
    array = gpu.NDArray(data, [2, 3])
    
    reshaped = array.reshape([3, 2])
    expected = np.array(data, dtype=np.float32).reshape([3, 2])
    
    npt.assert_allclose(reshaped.numpy(), expected)

def test_transpose():
    data = [1, 2, 3, 4, 5, 6]
    array = gpu.NDArray(data, [2, 3])
    
    transposed = array.transpose([1, 0])
    expected = np.array(data, dtype=np.float32).reshape([2, 3]).transpose([1, 0])
    
    npt.assert_allclose(transposed.numpy(), expected)

def test_broadcast():
    data = [1, 2, 3]
    array = gpu.NDArray(data, [3])
    
    broadcasted = array.broadcast([2, 3])
    expected = np.broadcast_to(np.array(data, dtype=np.float32), (2, 3))
    
    npt.assert_allclose(broadcasted.numpy(), expected)

def test_make_compact():
    data = [1, 2, 3, 4, 5, 6]
    array = gpu.NDArray(data, [2, 3])
    
    transposed = array.transpose([1, 0])
    
    compact_array = transposed.make_compact()
    expected = np.array(data, dtype=np.float32).reshape([2, 3]).transpose([1, 0])
    
    npt.assert_allclose(compact_array.numpy(), expected)

def test_slice_basic():
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    array = gpu.NDArray(data, [3, 3])
    
    sliced = array[1:3, 1:3]
    expected = np.array(data, dtype=np.float32).reshape([3, 3])[1:3, 1:3]
    
    npt.assert_allclose(sliced.numpy(), expected)

def test_slice_with_step():
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    array = gpu.NDArray(data, [9])
    
    sliced = array[0:9:2]
    expected = np.array(data, dtype=np.float32)[0:9:2]
    
    npt.assert_allclose(sliced.numpy(), expected)

def test_setitem_scalar():
    array = gpu.NDArray([1, 2, 3, 4], [2, 2])
    
    array[0, :] = 10.0
    
    expected = np.array([[10., 10.], [3., 4.]], dtype=np.float32)
    npt.assert_allclose(array.numpy(), expected)

def test_setitem_scalar_on_slice():
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    array = gpu.NDArray(data, [3, 3])
    
    array[1:3, 1:3] = 99.0
    
    expected = np.array(data, dtype=np.float32).reshape([3, 3])
    expected[1:3, 1:3] = 99.0
    
    npt.assert_allclose(array.numpy(), expected)
