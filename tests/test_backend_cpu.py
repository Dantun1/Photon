
import photon.backend_cpu as be
import numpy as np
import numpy.testing as npt
import pytest

"""
Testing suite for CPU backend functionality.
"""

# CompactArray tests

def test_creates_compact_array():
    arr = be.CompactArray([1.0,2.0,3.0,4.0,5.0])
    assert arr.size() == 5

def test_stores_compact_array_correctly():
    data = [1.0,2.0,3.0,4.0,5.0]
    arr = be.CompactArray(data)
    assert arr.data == data


# NDArray tests

def test_creates_2d_array():
    # Setup ndarray object 
    data = [1.0,2.0,3.0,4.0]
    arr = be.NDArray(data, [2,2])

    # Convert to numpy array (with buffer protocol), check allclose 
    actual = np.array(arr)
    expected = np.array(data).reshape(2,2)
    
    npt.assert_allclose(actual, expected)

SHAPE_CASES = [
    ([6], [2,3]),
    ([2,3], [6]),
    ([3,2],[2,3]),
    ([100], [5,2,2,5]),
    ([5,2,2,5], [100])
]

@pytest.mark.parametrize("start_shape, new_shape", SHAPE_CASES)
def test_reshapes_nd_array_valid_cases(start_shape, new_shape):
    # Setup 

    # Raw data list
    total_elems = np.prod(start_shape)
    data = np.arange(total_elems, dtype=np.float32).tolist()
    
    # Actual vs np reshape
    starting_actual = be.NDArray(data, start_shape)
    actual = starting_actual.reshape(new_shape)
    expected = np.array(data).reshape(new_shape)
    
    npt.assert_allclose(np.array(actual), expected)

def test_reshape_invalid_shapes():
    arr = be.NDArray([1.0, 2.0, 3.0, 4.0], [2, 2])
    with pytest.raises(ValueError): 
        arr.reshape([5]) 


TRANSPOSE_CASES = [
    ([2, 3], [1, 0]),
    ([4, 2, 3], [2, 0, 1]),
    ([4, 2, 3], [2, 1, 0]),
    ([4,2,5],[0,2,1])
]

@pytest.mark.parametrize("start_shape, dims", TRANSPOSE_CASES)
def test_transposes_nd_array_valid_dimensions(start_shape, dims):
    total_elems = np.prod(start_shape)
    data = np.arange(total_elems, dtype=np.float32).tolist()

    arr = be.NDArray(data, start_shape)
    actual = arr.transpose(dims)

    expected = np.array(data).reshape(start_shape).transpose(dims)
    
    npt.assert_allclose(np.array(actual), expected)

def test_compaction_after_transpose():
    # Setup
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    arr = be.NDArray(data, [2, 3])
    transposed = arr.transpose([1, 0])
    # Creates a new NDArray with new compacted handle.
    compacted = transposed.make_compact()

    expected = np.array(data).reshape(2, 3).transpose()

    npt.assert_allclose(np.array(compacted), expected)

def test_reshape_after_transpose():

    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    arr = be.NDArray(data, [2, 3])

    transposed = arr.transpose([1, 0])
    reshaped = transposed.reshape([3, 2])

    expected = np.array(data).reshape(2, 3).transpose().reshape(3, 2)

    npt.assert_allclose(np.array(reshaped), expected)   
