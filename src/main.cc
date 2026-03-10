#include <vector>
#include <numeric> 
#include <iostream>
#include <backend_gpu.cuh>

int main() {
    std::cout << "Starting main" << std::endl;
    std::vector<float> data{1,2,3,4};
    DimVec shape{2,2};
    NDArray<float> tensor{data, shape};
    
    std::cout << "Created tensor. Handle pointer: " << tensor.handle().get() << std::endl;

    auto new_tensor = tensor.make_compact();
    
    cudaDeviceSynchronize();
    std::cout << "Sync complete" << std::endl;

    return 0;
}
