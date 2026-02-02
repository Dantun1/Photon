#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <backend_cpu.hpp>

namespace py = pybind11;

PYBIND11_MODULE(backend_cpu, m)
{
    py::class_<CompactArray>(m, "CompactArray")
        .def(py::init<const std::vector<float> &>())
        .def_readonly("data", &CompactArray::data)
        .def("size", &CompactArray::size)
        .def("print", &CompactArray::print);
}