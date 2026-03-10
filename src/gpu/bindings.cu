
//
//         // get shape and stride (bytes) data for python
//         std::vector<py::ssize_t> py_shape(array.shape().begin(), array.shape().end());
//         std::vector<py::ssize_t> py_strides(array.strides().size());
//         for (size_t i = 0; i < array.strides().size(); ++i) {
//             py_strides[i] = array.strides()[i] * sizeof(float);
//         }
//         py::array_t<float> result(py_shape, py_strides);
//
//         // Create empty array/buffer
//         py::buffer_info buf = result.request();
//         float* h_ptr = static_cast<float*>(buf.ptr);
//
//         // download the compacted array directly from the GPU
//         size_t elements = array.handle()->size()    
//         array.handle()->download(h_ptr, elements);
//
//         return result;
//         })
//
//
// }
