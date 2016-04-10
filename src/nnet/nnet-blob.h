
#ifndef NNET_BLOB
#define NNET_BLOB

#include "matrix/kaldi-matrix.h"
#include "matrix/kaldi-vector.h"
#include <vector>
#include <iostream>

namespace kaldi {
namespace nnet1 {
template<typename Real>
class Blob {
    public:
        Blob(CuMatrix<Real> &M);
        Blob(CuVector<Real> &V);
        ~Blob(){ delete []data_;}
        Real* data() {return data_;}
        inline std::vector<int>& shape() { return shape_;}
        inline std::vector<int>& stride() {return stride_;}
        Matrix<Real> ToMatrix();
        Vector<Real> ToVector();
    private:
        Real* data_;
        std::vector<int> shape_;
        std::vector<int> stride_;

};


template<typename Real>
Blob<Real>::Blob(CuMatrix<Real> &M) {
    Matrix<Real> *Mt_ptr = new Matrix<Real>(M);

    data_ = Mt_ptr->Data();
    shape_.push_back(Mt_ptr->NumRows());
    shape_.push_back(Mt_ptr->NumCols());
    stride_.push_back(Mt_ptr->Stride() * sizeof(Real));
    stride_.push_back(sizeof(Real));
    /*
    std::cout << "Creating Blobs...Rows:" << Mt_ptr->NumRows() << " Columns:" << Mt_ptr->NumCols() << " Stride:" << Mt_ptr->Stride() << std::endl;
    std::cout << "Example: " << data_[shape_[0] * shape_[1]-1] << std::endl;
    */
}

template<typename Real>
Blob<Real>::Blob(CuVector<Real> &V) {
    Vector<Real> *Vt = new Vector<Real>(V);
    data_ = Vt->Data();
    shape_.push_back(Vt->Dim());
    stride_.push_back(sizeof(Real));
}

template<typename Real>
Matrix<Real> Blob<Real>::ToMatrix() {
    KALDI_ASSERT(shape_.size() == 2);
    KALDI_ASSERT(stride_[1] == sizeof(Real));
    Matrix<Real> mat(data_, shape_[1], shape_[0], stride_[0] / sizeof(Real));
    return mat;
}
template<typename Real>
Vector<Real> Blob<Real>::ToVector(){
    KALDI_ASSERT(shape_.size() == 1);
    KALDI_ASSERT(stride_[0] == sizeof(Real));
    Vector<Real> vec(data_, shape_[0]);
    return vec;
}

// Instantiate
template class Blob<float>;

}
}

#endif
