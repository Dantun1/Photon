template <typename T, typename Op>
__global__
void unary_op_contiguous_kernel(const T* src, T* dst, size_t size, Op op){
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid >= size) return;

  dst[tid] = op(src[tid]);
}


template <typename T, typename Op>
__global__
void unary_op_kernel(const T* src, T* dst, size_t size, TensorMeta meta, Op op){
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid >= size) return;

  size_t read_idx = meta.offset;
  size_t temp_idx = tid;
  for (int dim = meta.rank - 1; dim >= 0; dim--) {
    size_t dim_offset = temp_idx % meta.shape[dim];
    read_idx += dim_offset * meta.strides[dim];
    // skip the current dim to access offset into next dim
    temp_idx /= meta.shape[dim];
  }
  
  dst[tid] = op(src[read_idx]);
}

template <typename T>
template <typename Op>
NDArray<T> NDArray<T>::unary_dispatch(Op op) const 
{
  // New buffer 
  NDArray<T> new_array{_shape};
  size_t size = std::accumulate(_shape.begin(), _shape.end(), 1ULL, std::multiplies<size_t>());
  TensorMeta tmeta = meta();
  
  // call kernels depending on contiguity
  int threads = 256;
  int blocks = (size + threads - 1) / threads;
    
  if (is_contiguous()) {
      unary_op_contiguous_kernel<<<blocks, threads>>>(
          // pre increment with any offset 
          handle()->d_ptr() + tmeta.offset,
          new_array.handle()->d_ptr(),
          size,
          op
      );
  } else {
      unary_op_kernel<<<blocks, threads>>>(
          handle()->d_ptr(),
          new_array.handle()->d_ptr(), 
          size,
          tmeta,
          op
      );
  }
  return new_array;
}

template <typename T>
NDArray<T> NDArray<T>::neg() const
{
  return unary_dispatch([] __device__ (T scalar) { return -scalar; });
}
template <typename T>
NDArray<T> NDArray<T>::exp() const
{
  // cuda builtins for math functors
  return unary_dispatch([] __device__ (T scalar) { return ::exp(scalar); });
}
template <typename T>
NDArray<T> NDArray<T>::log() const
{
    return unary_dispatch([] __device__ (T scalar) { return ::log(scalar); });
}

template <typename T>
NDArray<T> NDArray<T>::sqrt() const
{
    return unary_dispatch([] __device__ (T scalar) { return ::sqrt(scalar); });
}

template <typename T>
NDArray<T> NDArray<T>::sin() const
{
    return unary_dispatch([] __device__ (T scalar) { return ::sin(scalar); });
}

template <typename T>
NDArray<T> NDArray<T>::cos() const
{
    return unary_dispatch([] __device__ (T scalar) { return ::cos(scalar); });
}

template <typename T>
NDArray<T> NDArray<T>::tanh() const
{
    return unary_dispatch([] __device__ (T scalar) { return ::tanh(scalar); });
}
