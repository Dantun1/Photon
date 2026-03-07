#include <vector>
#include <stdexcept>
// This version manually allocates/frees from global memory
// rather than unified memory - i want to profile the performance bottlenecks.

template <typename T>
CompactArray<T>::CompactArray(size_t sz): _size{sz}{
  // allocate global memory array of specified size
  cudaError_t err = cudaMalloc(&_d_ptr, sizeof(T) * _size);

  if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
}


template <typename T>
CompactArray<T>::CompactArray(std::vector<T> data): _size{data.size()}{
  // allocate global memory array of specified size
  cudaError_t err = cudaMalloc(&_d_ptr, sizeof(T) * _size);

  if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));

  upload(data.data(), _size);
}

template <typename T>
size_t CompactArray<T>::size() const
{
    return _size;
}

template <typename T>
CompactArray<T>::~CompactArray(){
  // allocate global memory array of specified size
  if (_d_ptr != nullptr) cudaFree(_d_ptr);
}


// Implement upload/download, get the array working to / from gpu.

template <typename T>
void CompactArray<T>::upload(const T* h_ptr, size_t count) {
  cudaError_t err = cudaMemcpy(_d_ptr, h_ptr, count * sizeof(T), cudaMemcpyHostToDevice);

  if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
}

template <typename T>
void CompactArray<T>::download(T* h_ptr, size_t count) const{
  cudaError_t err = cudaMemcpy(h_ptr, _d_ptr, count * sizeof(T), cudaMemcpyDeviceToHost);

  if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
}

// This pointer should never be called in host code
template <typename T>
const T* CompactArray<T>::d_ptr() const {
    return _d_ptr;
}

template <typename T>
T* CompactArray<T>::d_ptr() {
    return _d_ptr;
}


