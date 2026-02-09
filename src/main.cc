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

    Tensor mini_tens{std::vector<float>{1, 2, 3, 4, 5, 6}};
    mini_tens.print();
    mini_tens.reshape({3, 2}).print();

    return 0;
}
