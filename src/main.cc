#include <vector>
#include <numeric> 
#include <iostream>
#include <backend_gpu.cuh>

int main() {
    size_t N = 1000;
    
    std::vector<float> data{1,2,3,4};
    DimVec shape{2,2};
    NDArray<float> tensor{data, shape};

      std::cout << "Shape: [";
      for (size_t dim : tensor.shape()) {
          std::cout << dim << " ";
      }
      std::cout << "]" << std::endl;

      std::cout << "Strides: [";
      for (size_t stride : tensor.strides()) {
          std::cout << stride << " ";
      }
      std::cout << "]" << std::endl;



    return 0;
}
