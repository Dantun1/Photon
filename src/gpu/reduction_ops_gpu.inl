// reduce functor methods for passing to generic kernels, inlined
template <typename T>
struct MaxOp {
  __host__ __device__ __forceinline__ T operator()(T a, T b) const {
    return a > b ? a : b;
  }

  __host__ __device__ __forceinline__ T identity() const {
    return cuda::std::numeric_limits<T>::lowest();
  }
};

template <typename T>
struct MinOp {
  __host__ __device__ __forceinline__ T operator()(T a, T b) const {
    return a < b ? a : b;
  }

  __host__ __device__ __forceinline__ T identity() const {
    return cuda::std::numeric_limits<T>::max();
  }
};

template <typename T>
struct SumOp {
  __host__ __device__ __forceinline__ T operator()(T a, T b) const {
    return a + b;
  }

  __host__ __device__ __forceinline__ T identity() const {
    return static_cast<T>(0);
  }
};

// CAS loop for atomic max/min of floating point types as not natively supported by CUDA
template <typename T>
void __device__ __forceinline__ customAtomicMax(T* dst_addr, T value) {
  if constexpr (std::is_integral_v<T>) {
    atomicMax(dst_addr, value);
  } else if constexpr (sizeof(T) == 4) {
    // ptr cast for CAS compatibility
    unsigned int* dst_ui = reinterpret_cast<unsigned int*>(dst_addr);
    unsigned int old = *dst_ui, assumed_val;
    do {
      // what we think is there 
      assumed_val = old;
      // cast to T for correct comparison 
      T old_val = reinterpret_cast<T&>(assumed_val);
      T next_val = old_val > value ? old_val : value;
      // CAS with integral type 
      old = atomicCAS(dst_ui, assumed_val, reinterpret_cast<unsigned int&>(next_val));
    } while (assumed_val != old);
    // reinterpret as integer then hit compare and swap loop 
  } else if constexpr (sizeof(T) == 8) {
    unsigned long long* dst_ull = reinterpret_cast<unsigned long long*>(dst_addr);
    unsigned long long old = *dst_ull, assumed_val;
    // done until successfully update old value with correct next.
    do {
      assumed_val = old;
      T old_val = reinterpret_cast<T&>(assumed_val);
      T next_val = old_val > value ? old_val : value;
      old = atomicCAS(dst_ull, assumed_val, reinterpret_cast<unsigned long long&>(next_val));
    } while (assumed_val != old);
  }
}

template <typename T>
void __device__ __forceinline__ customAtomicMin(T* dst_addr, T value) {
  if constexpr (std::is_integral_v<T>) {
    atomicMin(dst_addr, value);
  } else if constexpr (sizeof(T) == 4) {
    // ptr cast for CAS compatibility
    unsigned int* dst_ui = reinterpret_cast<unsigned int*>(dst_addr);
    unsigned int old = *dst_ui, assumed_val;
    do {
      assumed_val = old;
      T old_val = reinterpret_cast<T&>(assumed_val);
      T next_val = old_val < value ? old_val : value;
      old = atomicCAS(dst_ui, assumed_val, reinterpret_cast<unsigned int&>(next_val));
    } while (assumed_val != old);
    // reinterpret as integer then hit compare and swap loop 
  } else if constexpr (sizeof(T) == 8) {
    unsigned long long* dst_ull = reinterpret_cast<unsigned long long*>(dst_addr);
    unsigned long long old = *dst_ull, assumed_val;
    // done until successfully update old value with correct next.
    do {
      assumed_val = old;
      T old_val = reinterpret_cast<T&>(assumed_val);
      T next_val = old_val < value ? old_val : value;
      old = atomicCAS(dst_ull, assumed_val, reinterpret_cast<unsigned long long&>(next_val));
    } while (assumed_val != old);
  }
}

// -- reduction kernels -- 


template <typename T, typename ReduceOp, typename AtomicOp>
__global__
void batched_reduction_kernel(
  const T* src, 
  T* dst, 
  size_t M, size_t N, 
  ReduceOp op, AtomicOp at_op
  ) 
{
  // dynamic buffer for initial reduced x values from global memory 
  extern __shared__ T input_segment[];

  
  //Each block reduces a row until all M are covered
  for (size_t by = blockIdx.y; by < M; by += gridDim.y) {
    // Start idx of non-reduced dimension being reduced 
    size_t row_start = by * N;

    // row-relative initial index within block 
    size_t col_idx  = (blockIdx.x * blockDim.x) + threadIdx.x;
    
    T init_val = op.identity();
    // dynamic coarsening
    for (; col_idx < N; col_idx += gridDim.x * blockDim.x) {
      // reduce all valid values blockDim.x apart from init point
      init_val = op(init_val, src[row_start + col_idx]);
    }
    input_segment[threadIdx.x] = init_val;

    // strides reduce so neighbouring threads remain active for minimal control divergence
    for (size_t stride = blockDim.x / 2; stride >= 1; stride /=2) {
      __syncthreads();
      if (threadIdx.x < stride) {
        input_segment[threadIdx.x] = op(input_segment[threadIdx.x], input_segment[threadIdx.x + stride]);
      }
    } 

    if (threadIdx.x == 0) {
      at_op(&dst[by],input_segment[0]);
    }
  }
}




template <typename T>
template <typename ReduceOp, typename AtomicOp>
NDArray<T> NDArray<T>::reduction_dispatch(const DimVec& axes,  ReduceOp op, AtomicOp at_op, bool keepdims) const {

  DimVec permuted_axes = _shape; // will hold permuation needed for batched reduction s.t. (NR,R) shape after transpose
  DimVec new_shape; // will hold shape for resulting NDArray, just (NR,) and 1s if keepdims
                   
  size_t M = 1; // product of non-reduced dims for grid y axis 
  size_t N = 1; // product of reduced dims for grid x axis
  
  std::vector<bool> removed(_shape.size());
  for (auto ax : axes) {
    if (ax >= _shape.size()) 
      throw std::invalid_argument("invalid dim provided, must be within range of tensor dims");
    removed[ax] = true;
  }

  bool current_shape_compatible = true; // if reduced dims are already at end, we don't need to transpose current array
  bool dim_removed_yet = false; // flag to check if we have a reduced dim followed by a non reduced dim e.g. NR -> R -> NR then incompatible 

  size_t left = 0;
  size_t right = _shape.size()-1;
  for (size_t i = 0; i < _shape.size(); ++i) {
    if (removed[i]) {
      if (keepdims) new_shape.push_back(1);
      permuted_axes[right--] = i;

      N *= _shape[i];
      dim_removed_yet = true;
    }
    else {
      new_shape.push_back(_shape[i]);
      permuted_axes[left++] = i;

      M *= _shape[i];
      if (dim_removed_yet) current_shape_compatible = false;
    }
  } 
  
  // -- Setup src / dst NDArrays -- 

  NDArray<T> dst_ndarray{new_shape};
  size_t dst_size = std::accumulate(new_shape.begin(), new_shape.end(), 1ULL, std::multiplies<size_t>());
  if (dst_size > 0) {
      int init_threads = 256;
      int init_blocks = (dst_size + init_threads - 1) / init_threads;
      setitem_scalar_contiguous_kernel<<<init_blocks, init_threads>>>(
          op.identity(),
          dst_ndarray.handle()->d_ptr(), 
          dst_size
      );
  }
  NDArray<T> src_ndarray = current_shape_compatible ? *this : transpose(permuted_axes);
  // ensure compact always for now, otherwise need separate kernel for non compacted reduction
  if (!src_ndarray.is_contiguous()) {
    src_ndarray = src_ndarray.make_compact();
  }

  // -- Optimal kernel dispatch-- 
  const DeviceManager& device_info = DeviceManager::get();

  constexpr int threads_pb = kernelcfg::REDUCTION_THREADS_PB;
  size_t shared_mem_bytes = threads_pb * sizeof(T);

  // cache max blocks per sm for this instantation
  static int max_blocks_per_sm = 0;
  if (max_blocks_per_sm == 0) {
    auto reduce_ptr = batched_reduction_kernel<T, ReduceOp, AtomicOp>;
    cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &max_blocks_per_sm,
      reduce_ptr,
      threads_pb,
      shared_mem_bytes
      );

    if (err != cudaSuccess) {
      throw std::runtime_error("Can't calculate max blocks that can run concurrently");
    }
  }


  constexpr int waves = kernelcfg::REDUCTION_WAVES;
  // avoid overserialisation, but ensure high occupancy
  size_t max_optimal_blocks = max_blocks_per_sm * device_info.num_sms * waves;
  

  size_t blocks_y = std::min<size_t>(M, max_optimal_blocks);
  size_t blocks_x = 1;

  // distribute remaining blocks 
  if (blocks_y < max_optimal_blocks) {
    size_t blocks_left = max_optimal_blocks / blocks_y;

    size_t min_elements_per_block = threads_pb * kernelcfg::REDUCTION_MIN_ELEMS_PER_THREAD;
    size_t max_useful_blocks = (N + min_elements_per_block - 1) / min_elements_per_block;

    // cap to max useful blocks for the reduction dims. At least 1 though.
    blocks_x = std::min<size_t>(blocks_left, max_useful_blocks);
    blocks_x = std::max<size_t>(1, blocks_x);
  }

  dim3 grid(blocks_x,blocks_y, 1);
  dim3 block(threads_pb,1, 1);
  
  batched_reduction_kernel<<<grid,block,shared_mem_bytes>>>(
      src_ndarray.handle()->d_ptr(),
      dst_ndarray.handle()->d_ptr(), 
      M, N, 
      op, at_op
      );
  return dst_ndarray.reshape(new_shape);
}

template <typename T>
NDArray<T> NDArray<T>::sum(const DimVec& axes, bool keepdims) const {

  return reduction_dispatch(
      axes,
      SumOp<T>{},
      [] __device__ (T* output, T val) {atomicAdd(output, val);},
      keepdims
      );
}

template <typename T>
NDArray<T> NDArray<T>::max(const DimVec& axes, bool keepdims) const {

  return reduction_dispatch(
      axes,
      MaxOp<T>{},
      [] __device__ (T* output, T val) {customAtomicMax(output, val);},
      keepdims
      );
}


template <typename T>
NDArray<T> NDArray<T>::min(const DimVec& axes, bool keepdims) const {
  return reduction_dispatch(
      axes,
      MinOp<T>{},
      [] __device__ (T* output, T val) {customAtomicMin(output, val);},
      keepdims
      );
}
