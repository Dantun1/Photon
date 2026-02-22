#include <vector>
#include <iostream>

/*
 * Implementation of CompactArray methods.
 */

template <typename T>
CompactArray<T>::CompactArray(size_t size) : data(size) {}

template <typename T>
CompactArray<T>::CompactArray(const std::vector<T> &input) : data(input) {}

template <typename T>
CompactArray<T>::CompactArray(std::vector<T> &&input) : data{std::move(input)} {}

template <typename T>
size_t CompactArray<T>::size() const
{
    return data.size();
}

template <typename T>
void CompactArray<T>::print() const
{
    for (const auto &value : data)
    {
        std::cout << value << " ";
    }
    std::cout << std::endl;
}

template <typename T>
T *CompactArray<T>::ptr()
{
    return data.data();
}

template <typename T>
const T *CompactArray<T>::ptr() const
{
    return data.data();
}

