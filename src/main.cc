#include <backend_cpu.hpp>

/**
 * Rough testing file, unit tests in python
 */

using Tensor = NDArray<float>;

int main()
{
    Tensor tensor{{1, 2, 3, 4}, {2, 2}};
    tensor.print();
    if (tensor.is_contiguous())
    {
        std::cout << "Tensor is contiguous." << std::endl;
    }
    return 0;
}
