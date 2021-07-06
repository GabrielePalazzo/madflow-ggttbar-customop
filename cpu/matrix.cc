#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/math_ops.h"

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

int nevents = 2;
Tensor matrix(const double*, const double*, const double, const double, const complex128*, const complex128*);
Tensor vxxxxx(double*, double, double, double);
    
class MatrixOp : public OpKernel {
 public:
  explicit MatrixOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& all_ps_tensor = context->input(0);
    auto all_ps = all_ps_tensor.flat<double>().data();
    
    const Tensor& hel_tensor = context->input(1);
    auto hel = hel_tensor.flat<double>().data();
    
    const Tensor& mdl_MT_tensor = context->input(2);
    auto mdl_MT = mdl_MT_tensor.flat<double>().data();
    
    const Tensor& mdl_WT_tensor = context->input(3);
    auto mdl_WT = mdl_WT_tensor.flat<double>().data();
    
    const Tensor& GC_10_tensor = context->input(4);
    auto GC_10 = GC_10_tensor.flat<complex128>().data();
    
    const Tensor& GC_11_tensor = context->input(5);
    auto GC_11 = GC_11_tensor.flat<complex128>().data();

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, all_ps_tensor.shape(),
                                                     &output_tensor));
    auto output_flat = output_tensor->flat<double>();

    matrix(all_ps, hel, *mdl_MT, *mdl_WT, GC_10, GC_11);
  }
};

Tensor matrix(const double* all_ps, const double* hel, const double mdl_MT, const double mdl_WT, const complex128* GC_10, const complex128* GC_11) {
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
    
    // Begin code
    
    for (int i = 0; i < nevents; i++) {
        double all_ps_0[4];
        double all_ps_1[4];
        double all_ps_2[4];
        double all_ps_3[4];
        for (int j = 0; j < 4; j++) {
            all_ps_0[j] = all_ps[16 * i + j];
            all_ps_1[j] = all_ps[16 * i + j + 4];
            all_ps_2[j] = all_ps[16 * i + j + 8];
            all_ps_3[j] = all_ps[16 * i + j + 12];
        } 
        auto w0 = vxxxxx(all_ps_0, ZERO, hel[0], -1);
        //Tensor w0 = vxxxxx(all_ps[:,0],ZERO,hel[0],-1)
    }
    
    return Tensor(); // dummy tensor
}

Tensor complex_tf(Scope& scope, double real, double imag) {
     /*Complex(
  const ::tensorflow::Scope & scope,
  ::tensorflow::Input real,
  ::tensorflow::Input imag
)*/
    return Tensor(); // dummy tensor
//    return tensorflow::ops::Complex(scope, real, imag);
}

Tensor vxxxxx(double* p, double vmass, double nhel, double nsv) {
    //Scope root = Scope::NewRootScope();
    //Tensor v0 = tensorflow::ops::ExpandDims(root, complex_tf(root, 1, 1), 1);
    //v0 = tf.expand_dims(complex_tf(p[:, 0] * nsv, p[:, 3] * nsv), 1)
    //v1 = tf.expand_dims(complex_tf(p[:, 1] * nsv, p[:, 2] * nsv), 1)
    //pt2 = p[:, 1] ** 2 + p[:, 2] ** 2
    //pp = tfmath.minimum(p[:, 0], tfmath.sqrt(pt2 + p[:, 3] ** 2))
    //pt = tfmath.minimum(pp, tfmath.sqrt(pt2))
    
    double pt2 = p[1] * p[1] + p[2] * p[2];
    return Tensor(); // dummy tensor
}

REGISTER_KERNEL_BUILDER(Name("Matrix").Device(DEVICE_CPU), MatrixOp);


