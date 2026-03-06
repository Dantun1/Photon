#include <vector>
#include <numeric> 
#include <iostream>
#include <backend_gpu.cuh>

int main() {
    size_t N = 1000;
    CompactArray<int> vec{N};

    std::vector<int> h_buf(N);
    
    std::iota(h_buf.begin(), h_buf.end(), 0);

    vec.upload(h_buf.data(), N);

    vec.add_device(10); 

    vec.download(h_buf.data(), N);

    std::cout << "vec[0]: " << h_buf[0] << " (Expected 10)" << std::endl;
    std::cout << "vec[1]: " << h_buf[1] << " (Expected 11)" << std::endl;

    return 0;
}
