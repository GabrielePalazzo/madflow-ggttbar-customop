#ifndef MATRIX_H_
#define MATRIX_H_

#include <unsupported/Eigen/CXX11/Tensor>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
using namespace tensorflow;

template <typename Device>
struct MatrixFunctor {
  void operator()(const Device& d, const double*, const double*, const double*, const double*, const complex128*, const complex128*, double*, const int);
};

#if GOOGLE_CUDA
// Partially specialize functor for GpuDevice.
template <>
struct MatrixFunctor<Eigen::GpuDevice> {
  void operator()(const Eigen::GpuDevice& d, const double*, const double*, const double*, const double*, const complex128*, const complex128*, double*, const int);
};
#endif

#endif
