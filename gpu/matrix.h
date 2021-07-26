#ifndef MATRIX_H_
#define MATRIX_H_

#include <unsupported/Eigen/CXX11/Tensor>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <cuComplex.h>
using namespace tensorflow;

template <typename Device, typename T>
struct MatrixFunctor {
  void operator()(const Device& d, const double*, const double*, const double*, const double*, const T*, const T*, double*, const int);
};

#if GOOGLE_CUDA
// Partially specialize functor for GpuDevice.
template <typename T>
struct MatrixFunctor<Eigen::GpuDevice, T> {
  void operator()(const Eigen::GpuDevice& d, const double*, const double*, const double*, const double*, const T*, const T*, double*, const int);
};
#endif

#include <thrust/complex.h>

#define COMPLEX_TYPE complex128//thrust::complex<double>

#endif
