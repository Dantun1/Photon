#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <backend_gpu.cuh>

namespace py = pybind11;

auto process_slices(const NDArray<float> &self, const py::object &index)
{
    // populate slice_ranges then call slice func with it to return the new view.
    std::vector<NDArray<float>::Slice> slice_ranges;
    const auto &shape = self.shape();

    // Tuple of slices, single slice, or single index.
    if (py::isinstance<py::tuple>(index))
    {
        auto tup = index.cast<py::tuple>();
        if (tup.size() > shape.size())
            throw py::index_error("Too many indices for array");
        for (size_t i = 0; i < tup.size(); ++i)
        {
            if (py::isinstance<py::slice>(tup[i]))
            {
                auto s = tup[i].cast<py::slice>();
                size_t start, stop, step, slicelength;
                s.compute(self.shape()[i], &start, &stop, &step, &slicelength);
                slice_ranges.push_back({(int64_t)start, (int64_t)stop, (int64_t)step, false});
            }
            else
            {
                int64_t idx = tup[i].cast<int64_t>();
                if (idx < 0)
                    idx += shape[i];
                if (idx < 0 || idx >= (int64_t)shape[i])
                    throw py::index_error("Index out of bounds");
                slice_ranges.push_back({idx, 0, 0, true});
            }
        }
    }
    else if (py::isinstance<py::slice>(index))
    {
        auto s = index.cast<py::slice>();
        size_t start, stop, step, slicelength;
        s.compute(self.shape()[0], &start, &stop, &step, &slicelength);
        slice_ranges.push_back({(int64_t)start, (int64_t)stop, (int64_t)step, false});
    }
    else
    {
        int64_t idx = index.cast<int64_t>();
        // if negative, convert to positive.
        if (idx < 0)
            idx += shape[0];
        // if still negative or too big, throw OOB error.
        if (idx < 0 || idx >= (int64_t)shape[0])
            throw py::index_error("Index out of bounds");
        slice_ranges.push_back({idx, 0, 0, true});
    }
    return slice_ranges;
}

PYBIND11_MODULE(backend_gpu, m) {
    py::class_<NDArray<float>>(m, "NDArray")
         // Constructors
        .def(py::init<std::vector<float>, DimVec>())
        .def(py::init<std::vector<float>>())
        // Metadata
        .def_property_readonly("shape", &NDArray<float>::shape)
        .def_property_readonly("strides", &NDArray<float>::strides)
        .def_property_readonly("offset", &NDArray<float>::offset)
        // Compaction
        .def("make_compact", &NDArray<float>::make_compact)
        .def("reshape", &NDArray<float>::reshape)
        .def("transpose", &NDArray<float>::transpose)
        .def("broadcast", &NDArray<float>::broadcast)
        .def("__getitem__", [](const NDArray<float> &self, py::object index)
             { 
                auto slice_ranges = process_slices(self, index);
                return self.slice(slice_ranges); })
        .def("numpy", [](const NDArray<float> &array) {

            NDArray<float> compact_array = array.is_contiguous() ? array : array.make_compact();

            std::vector<py::ssize_t> py_shape(compact_array.shape().begin(), compact_array.shape().end());
            std::vector<py::ssize_t> py_strides(compact_array.strides().size());
            
            for (size_t i = 0; i < compact_array.strides().size(); ++i) {
                py_strides[i] = compact_array.strides()[i] * sizeof(float);
            }

            py::array_t<float> result(py_shape, py_strides);

            py::buffer_info buf = result.request();
            float* h_ptr = static_cast<float*>(buf.ptr);

            size_t elements = compact_array.handle()->size();
            compact_array.handle()->download(h_ptr, elements);

            return result;
        });
}
