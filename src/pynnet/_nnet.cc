
#include <Python.h>  // NOLINT(build/include_alpha)
#include <boost/numpy.hpp>
#include <boost/python.hpp>
#include <boost/python/raw_function.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

#include <string>  // NOLINT(build/include_order)
#include <vector>  // NOLINT(build/include_order)
#include <iostream>

#include "nnet/nnet-nnet.h"

namespace bp = boost::python;
namespace bn = boost::numpy;

namespace kaldi {
namespace nnet1 {

bn::ndarray blob_wrapper(Blob<float> * const self) {
    std::vector<int> shape = self->shape();
    bn::ndarray arr = bn::from_data(self->data(), bn::dtype::get_builtin<float>(), self->shape(), self->stride(), bp::object());
    /*
    if (shape.size() == 2) {
        std::cout << "Converting Blobs...Rows:" << self->shape()[0] << " Columns:" << self->shape()[1] << " Stride:" << self->stride()[0] << ' ' << self->stride()[1] << std::endl;
        std::cout << "Example: " << self->data()[self->shape()[0] * self->shape()[1]-1] << ' ' << self->data()[400] << ' ' << self->data()[400 * shape[1]] << std::endl;
        std::cout << "Example: " << bp::extract<float>(arr[bp::make_tuple(-1,-1)]) << ' ' << bp::extract<float>(arr[bp::make_tuple(0, 400)]) << ' ' << bp::extract<float>(arr[bp::make_tuple(400,0)]) <<std::endl;
    }
    else {
        std::cout << "Converting Blobs...Column size:" << self->shape()[0] << " Stride:" << self->stride()[0] << std::endl;
        std::cout << "Example: " << self->data()[self->shape()[0]-1] << ' ' << self->data()[400]  << std::endl;
        std::cout << "Example: " << bp::extract<float>(arr[bp::make_tuple(-1)]) << ' ' << bp::extract<float>(arr[bp::make_tuple(400)])  <<std::endl;
    }
    */
    return arr;
}

// Wrapper to overloaded functions

void (Nnet::*read_str)(const std::string &) = &Nnet::Read;
typedef void (Nnet::*WriteFunc)(const std::string &, bool) const;
WriteFunc write_str = &Nnet::Write;


BOOST_PYTHON_MODULE(_nnet) {
    bn::initialize();
    bp::class_<Nnet>("Nnet", bp::init<>())
        .def("read", read_str)
        .def("write", write_str)
        .add_property("layers", bp::make_function(
            &Nnet::Components, bp::return_internal_reference<>()));

    bp::class_<Component>("Layer", bp::no_init)
        .add_property("type", &Component::Type)
        .def("get_params", &Component::Params)
        .def("set_params", &Component::SetParams);

    bp::class_<Blob<float> >("Blob", bp::no_init)
        .add_property("data", &blob_wrapper);

    bp::class_<std::vector<Blob<float>*  > >("BlobVec")
        .def(bp::vector_indexing_suite<std::vector<Blob<float>* >, true>());

    bp::class_<std::vector<Component* > >("LayerVec")
        .def(bp::vector_indexing_suite<std::vector<Component* >, true>());
}

}
}
