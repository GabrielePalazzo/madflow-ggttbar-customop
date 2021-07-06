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
double SQH = sqrt(0.5);
Tensor matrix(const double*, const double*, const double, const double, const complex128*, const complex128*);
Tensor vxxxxx(double*, double, double, double);
std::vector<complex128> _vx_BRST_check(double* p, double vmass);
std::vector<complex128> _vx_no_BRST_check(double *p, double vmass, double nhel, double nsv, double hel0, double nsvahl, double pp, double pt);
std::vector<complex128> _vx_BRST_check_massless(double* p);
std::vector<complex128> _vx_BRST_check_massive(double* p, double vmass);
std::vector<complex128> _vx_no_BRST_check_massive(double* p, double vmass, double nhel, double hel0, double nsvahl, double pp, double pt);
std::vector<complex128> _vx_no_BRST_check_massive_pp_zero(double nhel, double nsvahl);
std::vector<complex128> _vx_no_BRST_check_massive_pp_nonzero(double* p, double vmass, double nhel, double hel0, double nsvahl, double pp, double pt);
std::vector<complex128>  _vx_no_BRST_check_massive_pp_nonzero_pt_nonzero(double* p, double nhel, double hel0, double nsvahl, double pp, double pt, double emp);
std::vector<complex128> _vx_no_BRST_check_massive_pp_nonzero_pt_zero(double* p, double nhel, double nsvahl);
std::vector<complex128> _vx_no_BRST_check_massless(double* p, double nhel, double nsv);
double signvec(double x, double y);
    
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
        auto w1 = vxxxxx(all_ps_1, ZERO, hel[1], -1);
        //Tensor w0 = vxxxxx(all_ps[:,0],ZERO,hel[0],-1)
        //w1 = vxxxxx(all_ps[:,1],ZERO,hel[1],float_me(-1))
        //w2 = oxxxxx(all_ps[:,2],mdl_MT,hel[2],float_me(+1))
        //w3 = ixxxxx(all_ps[:,3],mdl_MT,hel[3],float_me(-1))
        //w4= VVV1P0_1(w0,w1,GC_10,ZERO,ZERO)
    }
    
    return Tensor(); // dummy tensor
}

std::vector<complex128> _vx_BRST_check(double* p, double vmass) {
    if (vmass == 0) {
        return _vx_BRST_check_massless(p);
    }
    else {
        return _vx_BRST_check_massive(p, vmass);
    }
}

std::vector<complex128>  _vx_no_BRST_check(double* p, double vmass, double nhel, double nsv, double hel0, double nsvahl, double pp, double pt) {
    if (vmass != 0) {
        return _vx_no_BRST_check_massive(
                            p, vmass, nhel, hel0, nsvahl, pp, pt
                                                );
    }
    else {
        return _vx_no_BRST_check_massless(p, nhel, nsv);
    }
}

std::vector<complex128> _vx_BRST_check_massless(double* p) {
    std::vector<complex128> r(p, p + 4);
    for (int i = 0; i < 4; i++) {
        r[i] = p[i]/p[0];
    }
    return r;
}

std::vector<complex128> _vx_BRST_check_massive(double* p, double vmass) {
    std::vector<complex128> r(p, p + 4);
    for (int i = 0; i < 4; i++) {
        r[i] = p[i]/vmass;
    }
    return r;
}

std::vector<complex128> _vx_no_BRST_check_massive(double* p, double vmass, double nhel, double hel0, double nsvahl, double pp, double pt) {
    if (pp == 0) {
        return _vx_no_BRST_check_massive_pp_zero(nhel, nsvahl);
    }
    else {
        return _vx_no_BRST_check_massive_pp_nonzero(
                        p, vmass, nhel, hel0, nsvahl, pp, pt
                                                    );
    }
}

std::vector<complex128> _vx_no_BRST_check_massive_pp_zero(double nhel, double nsvahl) {
    double hel0 = 1.0 - abs(nhel);
    std::vector<complex128> v(4, complex128(1,0));
    v[1] = complex128(-nhel * SQH, 0.0);
    v[2] = complex128(0.0, nsvahl * SQH);
    v[3] = complex128(hel0, 0.0);
    return v;
}

std::vector<complex128> _vx_no_BRST_check_massive_pp_nonzero(double* p, double vmass, double nhel, double hel0, double nsvahl, double pp, double pt) {
    double emp = p[0] / (vmass * pp);
    complex128 v2 = complex128(hel0 * pp / vmass, 0.0);
    complex128 v5 = complex128(hel0 * p[3] * emp + nhel * pt / pp * SQH, 0.0);
    
    std::vector<complex128> v34(2, complex128(1,0));
    if (pt != 0) {
        v34 = _vx_no_BRST_check_massive_pp_nonzero_pt_nonzero(p, nhel, hel0, nsvahl, pp, pt, emp);
    }
    else {
        v34 = _vx_no_BRST_check_massive_pp_nonzero_pt_zero(p, nhel, nsvahl);
    }
    std::vector<complex128> ret(4, complex128(1,0));
    ret[0] = v2;
    ret[1] = v34[0];
    ret[2] = v34[1];
    ret[3] = v5;
    return ret;
}

std::vector<complex128> _vx_no_BRST_check_massive_pp_nonzero_pt_nonzero(double* p, double nhel, double hel0, double nsvahl, double pp, double pt, double emp) {
    std::vector<complex128> v(2, complex128(0,0));
    double pzpt = p[3] / (pp * pt) * SQH * nhel;
    v[0] = complex128(hel0 * p[1] * emp - p[1] * pzpt, -nsvahl * p[2] / pt * SQH);
    v[1] = complex128(hel0 * p[2] * emp - p[2] * pzpt, nsvahl * p[1] / pt * SQH);
    return v;
}

std::vector<complex128> _vx_no_BRST_check_massive_pp_nonzero_pt_zero(double* p, double nhel, double nsvahl) {
    std::vector<complex128> v(2, complex128(0,0));
    v[0] = complex128(-nhel * SQH, 0);
    v[1] = complex128(0.0, nsvahl * signvec(SQH, p[3]));
    return v;
}

std::vector<complex128> _vx_no_BRST_check_massless(double* p, double nhel, double nsv) {
}/*
    """
    Parameters
    ----------
        p: tf.Tensor, four-momenta of shape=(None,4)
        nhel: tf.Tensor, boson helicity of shape=()
        nsv: tf.Tensor, final|initial state of shape=()

    Returns
    -------
        tf.Tensor, of shape=(None,4) and dtype DTYPECOMPLEX
    """
    pp = p[:, 0]
    pt = tfmath.sqrt(p[:, 1] ** 2 + p[:, 2] ** 2)
    v2 = tf.expand_dims(tf.zeros_like(p[:,0], dtype=DTYPECOMPLEX), 1)
    v5 = tf.expand_dims(complex_tf(nhel * pt / pp * SQH, 0.0), 1)
    cond = tf.expand_dims(pt != 0, 1)
    v34 = tf.where(cond,
                   _vx_no_BRST_check_massless_pt_nonzero(p, nhel, nsv, pp, pt),
                   _vx_no_BRST_check_massless_pt_zero(p, nhel, nsv))
    return tf.concat([v2, v34, v5], axis=1)*/

double signvec(double x, double y) {
    int sign = 0;
    y >= 0 ? sign = 1 : sign = -1;
    return x * sign;
}

Tensor vxxxxx(double* p, double vmass, double nhel, double nsv) {
    //Scope root = Scope::NewRootScope();
    //Tensor v0 = tensorflow::ops::ExpandDims(root, complex_tf(root, 1, 1), 1);
    //v0 = tf.expand_dims(complex_tf(p[:, 0] * nsv, p[:, 3] * nsv), 1)
    //v1 = tf.expand_dims(complex_tf(p[:, 1] * nsv, p[:, 2] * nsv), 1)
    //pt2 = p[:, 1] ** 2 + p[:, 2] ** 2
    //pp = tfmath.minimum(p[:, 0], tfmath.sqrt(pt2 + p[:, 3] ** 2))
    //pt = tfmath.minimum(pp, tfmath.sqrt(pt2))
    //BRST = nhel == 4
    //v = tf.cond(BRST,
    //            lambda: _vx_BRST_check(p, vmass),
    //            lambda: _vx_no_BRST_check(
    //                       p, vmass, nhel, nsv, hel0, nsvahl, pp, pt
    //                                 )
    //           )
    //eps = tf.concat([v0, v1, v], axis=1)
    //return tf.transpose(eps)
    
    complex128 v0 = complex128(p[0] * nsv, p[3] * nsv);
    complex128 v1 = complex128(p[1] * nsv, p[2] * nsv);
    
    double pt2 = p[1] * p[1] + p[2] * p[2];
    double pp = std::min(p[0], sqrt(pt2 + p[3] * p[3]));
    double pt = std::min(pp, sqrt(pt2));
    
    double hel0 = 1 - abs(nhel);
    double nsvahl = nsv * abs(nhel);
    
    std::vector<complex128> v;
    
    if (nhel == 4) {
        v = _vx_BRST_check(p, vmass);
    }
    else {
        v = _vx_no_BRST_check(p, vmass, nhel, nsv, hel0, nsvahl, pp, pt);
    }
    
    return Tensor(); // dummy tensor
}

REGISTER_KERNEL_BUILDER(Name("Matrix").Device(DEVICE_CPU), MatrixOp);


