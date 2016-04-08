
#include <Python.h>  // NOLINT(build/include_alpha)
#include <boost/numpy.hpp>
#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

#include <string>  // NOLINT(build/include_order)
#include <vector>  // NOLINT(build/include_order)

#include "nnet/nnet-nnet.h"

namespace bp = boost::python;
namespace bn = boost::numpy;

namespace kaldi {
namespace nnet1 {

bn::ndarray blob_wrapper(Blob<float> * const self) {
    bn::ndarray arr = bn::from_data(self->data(), bn::dtype::get_builtin<float>(), self->shape(), self->stride(), bp::object());
    return arr;
}

// Wrapper to overloaded functions

void (Nnet::*read_str)(const std::string &) = &Nnet::Read;
typedef void (Nnet::*WriteFunc)(const std::string &, bool) const;
WriteFunc write_str = &Nnet::Write;


BOOST_PYTHON_MODULE(_nnet) {
    bn::initialize();
    bp::class_<Nnet>("Nnet")
        .def("read", read_str)
        .def("write", write_str)
        .add_property("layers", bp::make_function(
            &Nnet::Components, bp::return_internal_reference<>()));

    bp::class_<Component>("Layer")
        .add_property("type", &Component::Type)
        .def("get_params", bp::make_function(
            &Component::Params, bp::return_internal_reference<>()));
        /*
        .def("set_params", bp::make_function(
            &Component::SetParams, bp::return_internal_reference<>())); */

    bp::class_<Blob<float> >("Blob")
        .add_property("data", bp::make_function(
            &blob_wrapper, bp::return_internal_reference<>()));

    bp::class_<std::vector<Blob<float>*  > >("BlobVec")
        .def(bp::vector_indexing_suite<std::vector<Blob<float>* >, true>());

    bp::class_<std::vector<Component* > >("LayerVec")
        .def(bp::vector_indexing_suite<std::vector<Component* >, true>());
}

}
}
