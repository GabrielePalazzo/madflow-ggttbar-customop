#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
//#include "tensorflow/cc/ops/math_ops.h"

#include <vector>

using namespace tensorflow;

// mdl_MT,mdl_WT,GC_10,GC_11

REGISTER_OP("Matrix")
    .Input("all_ps: double")
    .Input("hel: double")
    .Input("mdl_mt: double")
    .Input("mdl_wt: double")
    .Input("gc_10: complex128")
    .Input("gc_11: complex128")
    .Output("zeroed: double")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });

Tensor matrix();
    
class MatrixOp : public OpKernel {
 public:
  explicit MatrixOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& all_ps = context->input(0);
    auto all_ps_flat = all_ps.flat<double>();

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, all_ps.shape(),
                                                     &output_tensor));
    auto output_flat = output_tensor->flat<double>();

    matrix();
  }
};

Tensor matrix() {
    int ngraphs = 3;
    int nwavefuncs = 5;
    int ncolor = 2;
    double ZERO = 0.;
    
    Tensor denom = Tensor(DT_COMPLEX128);
    denom.vec<complex128>()(0) = 3;
    denom.vec<complex128>()(1) = 3;
    
    Tensor cf = Tensor(DT_COMPLEX128, TensorShape ({2,2}));
    cf.vec<complex128>()(0) = 16;
    cf.vec<complex128>()(1) = -2;
    cf.vec<complex128>()(2) = -2;
    cf.vec<complex128>()(3) = 16;
    
    return Tensor(); // dummy tensor
}

REGISTER_KERNEL_BUILDER(Name("Matrix").Device(DEVICE_CPU), MatrixOp);


