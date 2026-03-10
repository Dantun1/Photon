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
  size_t size = old_handle->size();
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
