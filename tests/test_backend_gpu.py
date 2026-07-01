import photon.backend_gpu as gpu 
import numpy as np
import numpy.testing as npt
import pytest

# GPU NDArray tests
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

# Unary op tests 

@pytest.mark.parametrize("op_name, np_op", [
    ("neg", np.negative),
    ("exp", np.exp),
    ("log", np.log),
    ("sqrt", np.sqrt),
    ("sin", np.sin),
    ("cos", np.cos),
    ("tanh", np.tanh)
])
def test_unary_ops_contiguous(op_name, np_op):
    # contiguous kernel across all ops 
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    array = gpu.NDArray(data, [2, 3])
    
    result = getattr(array, op_name)() 
    expected = np_op(np.array(data, dtype=np.float32).reshape([2, 3]))
    
    npt.assert_allclose(result.numpy(), expected, rtol=1e-5)

def test_unary_ops_strided():
    # non contiguous kernel test 
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    array = gpu.NDArray(data, [3, 3])
    sliced = array[0:3:2, 0:3:2] # non contiguous layout, should call slow kernel path 
    
    result = sliced.exp()
    
    np_array = np.array(data, dtype=np.float32).reshape([3, 3])
    np_sliced = np_array[0:3:2, 0:3:2]
    expected = np.exp(np_sliced)
    
    npt.assert_allclose(result.numpy(), expected, rtol=1e-5)


# Scalar ops 

def test_scalar_math_basic():
    array = gpu.NDArray([1, 2, 3, 4], [2, 2])
    np_array = np.array([1, 2, 3, 4], dtype=np.float32).reshape([2, 2])
    
    npt.assert_allclose((array + 2.0).numpy(), np_array + 2.0)
    npt.assert_allclose((array * 3.0).numpy(), np_array * 3.0)
    npt.assert_allclose((array - 1.0).numpy(), np_array - 1.0)
    npt.assert_allclose((array / 2.0).numpy(), np_array / 2.0)
    npt.assert_allclose((array ** 2.0).numpy(), np_array ** 2.0)

def test_scalar_math_reverse():
    array = gpu.NDArray([1, 2, 3, 4], [2, 2])
    np_array = np.array([1, 2, 3, 4], dtype=np.float32).reshape([2, 2])
    
    npt.assert_allclose((10.0 - array).numpy(), 10.0 - np_array)
    npt.assert_allclose((12.0 / array).numpy(), 12.0 / np_array)
    npt.assert_allclose((2.0 ** array).numpy(), 2.0 ** np_array)

def test_scalar_math_strided():
    array = gpu.NDArray([1, 2, 3, 4, 5, 6, 7, 8, 9], [3, 3])
    sliced = array[1:3, 1:3]
    
    np_array = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.float32).reshape([3, 3])
    np_sliced = np_array[1:3, 1:3]
    
    npt.assert_allclose((sliced + 5.0).numpy(), np_sliced + 5.0)


# Ewise tests

def test_ewise_contiguous():
    a = gpu.NDArray([1, 2, 3, 4], [2, 2])
    b = gpu.NDArray([5, 6, 7, 8], [2, 2])
    
    np_a = np.array([1, 2, 3, 4], dtype=np.float32).reshape([2, 2])
    np_b = np.array([5, 6, 7, 8], dtype=np.float32).reshape([2, 2])
    
    npt.assert_allclose((a + b).numpy(), np_a + np_b)
    npt.assert_allclose((a * b).numpy(), np_a * np_b)

def test_ewise_broadcasting():
    a = gpu.NDArray([1, 2, 3, 4, 5, 6], [2, 3])
    b = gpu.NDArray([10, 20, 30], [3]) # should broadcast up to 2, 3
    
    np_a = np.array([1, 2, 3, 4, 5, 6], dtype=np.float32).reshape([2, 3])
    np_b = np.array([10, 20, 30], dtype=np.float32)
    
    npt.assert_allclose((a + b).numpy(), np_a + np_b)
    npt.assert_allclose((a - b).numpy(), np_a - np_b)

def test_ewise_strided():
    a = gpu.NDArray([1, 2, 3, 4, 5, 6, 7, 8, 9], [3, 3])
    b = gpu.NDArray([9, 8, 7, 6, 5, 4, 3, 2, 1], [3, 3])
    
    # view over non contiguous buffer, 3,1 strides instead of 2,1 
    slice_a = a[0:2, 0:2]
    slice_b = b[1:3, 1:3]
    
    np_a = np.arange(1, 10, dtype=np.float32).reshape(3, 3)[0:2, 0:2]
    np_b = np.arange(9, 0, -1, dtype=np.float32).reshape(3, 3)[1:3, 1:3]
    
    npt.assert_allclose((slice_a * slice_b).numpy(), np_a * np_b)
