#include <numeric>
#include <memory>
#include <vector>
#include <utility>
#include <functional>
#include <stdexcept>
#include <cmath>
#include <algorithm>

template <typename T>
__global__ 
void compaction_kernel(const T* src, T* dst, size_t size, typename NDArray<T>::TensorMeta meta, size_t offset){
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid >= size) return;
  
  size_t temp_idx = tid;
  size_t read_idx = offset;
  
  // compute index into read pointer using tid
  for (int dim = meta.rank - 1; dim >= 0; dim--){
    size_t dim_offset = temp_idx % meta.shape[dim];
    read_idx += dim_offset * meta.strides[dim];
    temp_idx /= meta.shape[dim];
  }
  // strided read, coalesced write
  dst[tid] = src[read_idx];
}


template <typename T>
NDArray<T> NDArray<T>::make_compact() const {
  const auto& shape = this->shape();
  const auto& strides = this->strides();
  size_t offset = this->offset();

  const auto old_handle = this->handle();
  size_t size = std::accumulate(shape.begin(),shape.end(), 1ULL, std::multiplies<T>());
  size_t rank = shape.size();

  auto new_handle = std::make_shared<CompactArray<T>>(size);

  TensorMeta meta;
  meta.rank = rank;
  for (size_t i = 0; i < rank; ++i) {
      meta.shape[i] = shape[i];
      meta.strides[i] = strides[i];
  }
  
  int threads = 256;
  int blocks = (size + threads - 1) / threads;
  // launch many blocks to maximise SM utilisation
  compaction_kernel<<<blocks, threads>>>(
      old_handle->d_ptr(), new_handle->d_ptr(),
      size, meta, offset);

  return NDArray<T>(new_handle, shape);
}
// Need kernels for these
// massively parallel setting

void setitem_scalar(const std::vector<Slice>& slice_ranges, T scalar) {
  return;
}

void setitem_ewise(const std::vector<Slice>& slice_ranges,  const NDArray<T> &source) {
  return;
}
/**
 * Metadata based view ops. No need to manipulate data
 *
 */

template <typename T>
NDArray<T> NDArray<T>::reshape(const DimVec& new_shape) const{
  
    // New shape must have same elements.
    size_t new_size = std::accumulate(new_shape.begin(), new_shape.end(), 1, std::multiplies<size_t>());
    size_t current_size = std::accumulate(_shape.begin(), _shape.end(), 1, std::multiplies<size_t>());

    if (new_size != current_size)
    {
        throw std::invalid_argument("New shape must have same number of elements as current shape");
    }

    // If contiguous, just change shape and strides, else make compact and then change shape and strides.

    NDArray<T> source = is_contiguous() ? *this : make_compact();

    DimVec new_strides(new_shape.size());
    size_t dim_stride = 1;
    for (int i = static_cast<int>(new_shape.size()) - 1; i >= 0; --i)
    {
        new_strides[i] = dim_stride;
        dim_stride *= new_shape[i];
    }
    return NDArray<T>(source.handle(), new_shape, new_strides, source.offset());
}

template <typename T>
NDArray<T> NDArray<T>::transpose(const DimVec &axes) const
{
    if (axes.size() != _shape.size())
    {
        throw std::invalid_argument("Invalid number of axes for transpose: must match number of dimensions");
    }
    DimVec new_shape(_shape.size());
    DimVec new_strides(_strides.size());
    for (size_t i = 0; i < axes.size(); i++)
    {
        if (axes[i] >= _shape.size() or axes[i] < 0)
        {
            throw std::invalid_argument("Invalid axis index for transpose: must be between 0 and number of dimensions");
        }
        // Permute the shape, swap the strides according to the new order.
        new_shape[i] = _shape[axes[i]];
        new_strides[i] = _strides[axes[i]];
    }
    // same handle + offset, just shape/stride change
    return NDArray<T>(_handle, new_shape, new_strides, _offset);
}


template <typename T>
NDArray<T> NDArray<T>::slice(const std::vector<Slice> &slice_ranges) const
{
    size_t new_offset = _offset;
    DimVec new_strides;
    DimVec new_shape;

    for (size_t i = 0; i < slice_ranges.size(); i++)
    {
        const auto &[start, stop, step, is_index] = slice_ranges[i];

        // shift offset (based on original strides)
        new_offset += start * _strides[i];

        // if just scalar value for dim, we don't include in the
        // shape/strides, just shift offset
        if (is_index)
        {
            continue;
        }
        int64_t num_elements = (std::abs(stop - start) + std::abs(step) - 1) / std::abs(step);
        new_shape.push_back(static_cast<size_t>(num_elements)); // round up division for shape dim
        new_strides.push_back(_strides[i] * step);
    }

    for (size_t i = slice_ranges.size(); i < _shape.size(); i++)
    {
        // dims not in slice_ranges are unchanged, just add to new shape and strides.
        new_shape.push_back(_shape[i]);
        new_strides.push_back(_strides[i]);
    }

    return NDArray<T>(_handle, new_shape, new_strides, new_offset);
}


template <typename T>
NDArray<T> NDArray<T>::broadcast(const DimVec &new_shape) const
{
    if (new_shape.size() < _shape.size())
    {
        throw std::invalid_argument("Cannot broadcast to fewer dimensions");
    }
    DimVec new_strides(new_shape.size(), 0);
    
    // Right align + broadcast loop
    for (int i = static_cast<int>(new_shape.size()) - 1, j = static_cast<int>(_shape.size()) - 1; i >= 0; i--, j--)
    {   
        // if i is larger, the prepended dimensions in new_shape will just have strides of 0 from init
        if (j >= 0)
        {   
            // Enforce either identical or dimension of 1 in dimension to broadcast
            if (_shape[j] == new_shape[i])
            {
                new_strides[i] = _strides[j];
            }
            else if (_shape[j] == 1)
            {
                new_strides[i] = 0;
            }
            else
            {
                throw std::invalid_argument("Cannot broadcast: incompatible shapes");
            }
        }
    }

    return NDArray<T>(_handle, new_shape, new_strides, _offset);
}
