
// Initial implementation for GPU: uses explicit memory on GPU
// rather than unified memory - i want to profile the performance gaps.
template <typename T>
CompactArray<T>::CompactArray(size_t sz): size{sz}{
  // allocate global memory array of specified size
  cudaError_t err = cudaMalloc(&d_ptr, sizeof(T) * sz);

  if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
}


template <typename T>
CompactArray<T>::~CompactArray(){
  // allocate global memory array of specified size
  if (d_ptr != nullptr) cudaFree(d_ptr);
}


// Implement upload/download, get the array working to / from gpu.

template <typename T>
void CompactArray<T>::upload(const T* h_ptr, size_t count) {
  cudaError_t err = cudaMemcpy(d_ptr, h_ptr, count * sizeof(T), cudaMemcpyHostToDevice);

  if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
}

template <typename T>
void CompactArray<T>::download(T* h_ptr, size_t count) const{
  cudaError_t err = cudaMemcpy(h_ptr, d_ptr, count * sizeof(T), cudaMemcpyDeviceToHost);

  if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
}

// test function
template <typename T>
__global__
void test_add_kernel(T* data,T scalar, size_t sz){
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i<sz){
    data[i] += scalar;
  }
}



template <typename T>
void CompactArray<T>::add_device(T scalar) {
  test_add_kernel<<<(size+255)/256,256>>>(d_ptr, scalar, size);
}
  
