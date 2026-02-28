
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
    ([5,2,2,5], [100]),
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
    ([4,2,5],[0,2,1]),
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


SLICE_CASES = [
    ([4,4],[1,slice(1,3)]),
    ([4,4,4,4],[slice(1,3), slice(0,4), slice(0,4), slice(0,4)]),
    ([4,4,4,4],[slice(0,2)]),
    ([4,4,4,4],[0,0,0,1]),
]


@pytest.mark.parametrize("start_shape, slice_ranges", SLICE_CASES)
def test_nd_slicing_constant_step(start_shape, slice_ranges):
    elems = np.prod(start_shape)
    data = np.arange(elems, dtype=np.float32).tolist()
    arr = be.NDArray(data, start_shape)

    # Index as tuple of slices/ints for standard behaviour
    actual = arr[tuple(slice_ranges)]
    expected = np.array(data).reshape(start_shape)[tuple(slice_ranges)]

    npt.assert_allclose(np.array(actual), expected)


SET_SCALAR_CASES = [
    ([4,4],[1,slice(1,3)]),
    ([4,4,4,4],[slice(1,3), slice(0,4), slice(0,4), slice(0,4)]),
    ([4,4,4,4],[slice(0,2)]),
    ([4,4,4,4],[0,0,0,1]),
]


@pytest.mark.parametrize("start_shape, slice_ranges", SET_SCALAR_CASES)
def test_setitem_scalar(start_shape, slice_ranges):
    elems = np.prod(start_shape)
    data = np.arange(elems, dtype=np.float32).tolist()
    arr = be.NDArray(data, start_shape)

    arr[tuple(slice_ranges)] = 5

    # Expected result using numpy for verification
    expected_data = np.array(data).reshape(start_shape)
    expected_data[tuple(slice_ranges)] = 5

    npt.assert_allclose(np.array(arr), expected_data)


@pytest.mark.parametrize("t_shape, slices, s_data, s_shape", [
    ([4, 4], (slice(0, 2), slice(0, 2)), [10., 11., 12., 13.], [2, 2]),
    ([3, 3], (slice(None), slice(0, 2)), [1., 2., 3., 4., 5., 6.], [3, 2]),
    ([4, 4], (1, slice(None)), [9., 8., 7., 6.], [4,]),
])
def test_setitem_ewise_basic(t_shape, slices, s_data, s_shape):
    target_arr = be.NDArray(np.zeros(t_shape).flatten().tolist(), t_shape)
    source_arr = be.NDArray(s_data, s_shape)

    target_arr[slices] = source_arr

    expected = np.zeros(t_shape)
    expected[slices] = np.array(s_data).reshape(s_shape)
    
    npt.assert_allclose(np.array(target_arr), expected)

def test_setitem_ewise_transposed_source():
    # 4x4 zeros
    target = be.NDArray(np.zeros((4, 4)).flatten().tolist(), [4, 4])
    
    # transposed source: (3,2)
    s_data = [1., 2., 3., 4., 5., 6.]
    source = be.NDArray(s_data, [2, 3]).transpose([1, 0])
    
    # 3,2 slice of target
    target_slice = (slice(0, 3), slice(0, 2))

    # Perform assignment
    target[target_slice] = source
    
    # Expected
    expected = np.zeros((4, 4))
    expected[0:3, 0:2] = np.array(s_data).reshape(2, 3).T
    
    npt.assert_allclose(np.array(target), expected)


def test_setitem_ewise_broadcast_source():
    target = be.NDArray([0] * 32, [2, 4, 4])
    source = be.NDArray([1., 2., 3., 4.], [4])

    # Given a 4x4 view on this target, broadcast + set from 1d source
    target[:, 0, :] = source

    expected = np.zeros((2, 4, 4))
    expected[:, 0, :] = np.array([1., 2., 3., 4.])

    npt.assert_allclose(np.array(target), expected)

BROADCAST_CASES = [
    ([2, 4, 4], [4]),
    ([2, 3, 4], [4]),
    ([2, 3, 4], [3, 4]),
    ([2,3], [2,1])
]

@pytest.mark.parametrize("target_shape, source_shape", BROADCAST_CASES)
def test_broadcasting_basic(target_shape, source_shape):

    target = be.NDArray(np.arange(np.prod(target_shape), dtype=np.float32).tolist(), target_shape)
    source_data = np.arange(np.prod(source_shape), dtype=np.float32).tolist()
    source = be.NDArray(source_data, source_shape)

    broadcasted = source.broadcast(target_shape)

    expected = np.broadcast_to(np.array(source_data).reshape(source_shape), target_shape)

    npt.assert_allclose(np.array(broadcasted), expected)


MATMUL_CASES = [
    ([2,2],[2,2]),
    ([4,6,4,2,2],[4,6,4,2,2]),
    ([4,6,1,2,2],[4,6,4,2,2]),
    ([6,2],[2,8])
]

@pytest.mark.parametrize("mat_a_shape, mat_b_shape", MATMUL_CASES)
def test_matmul(mat_a_shape, mat_b_shape):
    a_data = np.arange(np.prod(mat_a_shape)).tolist()
    b_data = np.arange(np.prod(mat_b_shape)).tolist()

    a = be.NDArray(a_data, mat_a_shape)
    b = be.NDArray(b_data, mat_b_shape)

    result = a @ b

    expected = np.array(a_data).reshape(mat_a_shape) @ np.array(b_data).reshape(mat_b_shape)

    npt.assert_allclose(np.array(result),expected)
