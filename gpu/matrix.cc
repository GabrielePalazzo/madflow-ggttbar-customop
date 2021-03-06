#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/math_ops.h"

#include <omp.h>
#include <iomanip>

#include "matrix.h"

using namespace tensorflow;


using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

// mdl_MT,mdl_WT,GC_10,GC_11

REGISTER_OP("Matrix")
    .Attr("T: numbertype")
    .Input("all_ps: double")
    .Input("hel: double")
    .Input("mdl_mt: double")
    .Input("mdl_wt: double")
    .Input("gc_10: T")
    .Input("gc_11: T")
    .Output("matrix_element: double")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->MakeShape({c->Dim(c->input(0), 0)}));
      return Status::OK();
    });

//int nevents = 2;
double SQH = sqrt(0.5); // tf.math.sqrt(0.5) == 0.70710676908493;
complex128 CZERO = complex128(0.0, 0.0);
template <typename T>
void matrix(const double*, const double*, const double*, const double*, const T*, const T*, double*, const int);
template <typename T>
void vxxxxx(const double* p, double fmass, double nhel, double nsf, T*);
template <typename T>
void ixxxxx(const double* p, double fmass, double nhel, double nsf, T*);
template <typename T>
void oxxxxx(const double* p, double fmass, double nhel, double nsf, T*);
template <typename T>
void _ix_massive(const double* p, double fmass, double nsf, double nh, T* v);
template <typename T>
void _ix_massless(const double* p, double nhel, double nsf, double nh, T* v);
template <typename T>
void _ox_massless(const double* p, double nhel, double nsf, double nh, T* v);
template <typename T>
void _ox_massive(const double* p, double fmass, double nhel, double nsf, double nh, T* v);
template <typename T>
void _ix_massless_sqp0p3_zero(const double* p, double nhel, T& val);
template <typename T>
void _ix_massless_sqp0p3_nonzero(const double* p, double nh, double sqp0p3, T& val);
template <typename T>
void _ix_massive_pp_nonzero(const double* p, double fmass, double nsf, double nh, int ip, int im, double pp, T* v);
template <typename T>
void _ix_massless_nh_one(T* chi, T* v);
template <typename T>
void _ix_massless_nh_not_one(T* chi, T* v);
template <typename T>
void _ox_massive_pp_zero(double fmass, double nsf, int ip, int im, T* v);
template <typename T>
void _ox_massive_pp_nonzero(const double* p, double fmass, double nsf, double nh, double pp, T* v);
template <typename T>
void _vx_BRST_check(const double* p, double vmass, T* v);
template <typename T>
void _vx_no_BRST_check(const double *p, double vmass, double nhel, double nsv, double hel0, double nsvahl, double pp, double pt, T* v);
template <typename T>
void _vx_BRST_check_massless(const double* p, T* v);
template <typename T>
void _vx_BRST_check_massive(const double* p, double vmass, T* v);
template <typename T>
void _vx_no_BRST_check_massive(const double* p, double vmass, double nhel, double hel0, double nsvahl, double pp, double pt, T* v);
template <typename T>
void _vx_no_BRST_check_massive_pp_zero(double nhel, double nsvahl, T* v);
template <typename T>
void _vx_no_BRST_check_massive_pp_nonzero(const double* p, double vmass, double nhel, double hel0, double nsvahl, double pp, double pt, T* v);
template <typename T>
void _vx_no_BRST_check_massive_pp_nonzero_pt_nonzero(const double* p, double nhel, double hel0, double nsvahl, double pp, double pt, double emp, T* v);
template <typename T>
void _vx_no_BRST_check_massive_pp_nonzero_pt_zero(const double* p, double nhel, double nsvahl, T* v);
template <typename T>
void _vx_no_BRST_check_massless(const double* p, double nhel, double nsv, T* v);
template <typename T>
void _vx_no_BRST_check_massless_pt_nonzero(const double* p, double nhel, double nsv, double pp, double pt, T* v);
template <typename T>
void _vx_no_BRST_check_massless_pt_zero(const double* p, double nhel, double nsv, T* v);

template <typename T>
void VVV1P0_1(T* V2, T* V3, const T COUP, double M1, double W1, T*);
template <typename T>
void FFV1_0(T* F1, T* F2, T* V3, const T COUP, T& amp);
template <typename T>
void FFV1_1(T* F2, T* V3, const T COUP, double M1, double W1, T*);
template <typename T>
void FFV1_2(T* F1, T* V3, const T COUP, double M1, double W1, T*);

double sign(double x, double y);
double signvec(double x, double y);

template <typename T>
struct MatrixFunctor<CPUDevice, T> {
  void operator()(const CPUDevice& d, const double* all_ps, const double* hel, const double* mdl_MT, const double* mdl_WT, const T* GC_10, const T* GC_11, 
            double* output_flat, const int nevents) {
      matrix(all_ps, hel, mdl_MT, mdl_WT, GC_10, GC_11, output_flat, nevents);
  }
};
    
template <typename Device, typename T>
class MatrixOp : public OpKernel {
 public:
  explicit MatrixOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& all_ps_tensor = context->input(0);    // Shape: [nevents, 4, 4]
    auto all_ps = all_ps_tensor.flat<double>().data();
    
    const Tensor& hel_tensor = context->input(1);    // Shape: [4]
    auto hel = hel_tensor.flat<double>().data();
    
    const Tensor& mdl_MT_tensor = context->input(2);    // Shape: []
    auto mdl_MT = mdl_MT_tensor.flat<double>().data();
    
    const Tensor& mdl_WT_tensor = context->input(3);    // Shape: []
    auto mdl_WT = mdl_WT_tensor.flat<double>().data();
    
    const Tensor& GC_10_tensor = context->input(4);    // Shape: [nevents]
    auto GC_10 = GC_10_tensor.flat<T>().data();
    
    const Tensor& GC_11_tensor = context->input(5);    // Shape: [nevents]
    auto GC_11 = GC_11_tensor.flat<T>().data();

    const int nevents = all_ps_tensor.shape().dim_size(0);
    
    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({nevents}),
                                                     &output_tensor));
    auto output_flat = output_tensor->flat<double>();
    
    
    MatrixFunctor<Device, COMPLEX_TYPE>()(
        context->eigen_device<Device>(), all_ps, hel, mdl_MT, mdl_WT, GC_10, GC_11, output_flat.data(), nevents);
    //matrix(all_ps, hel, mdl_MT, mdl_WT, GC_10, GC_11, output_flat.data(), nevents);
  }
};
/*
// Register the CPU kernels.
REGISTER_KERNEL_BUILDER(Name("Matrix").Device(DEVICE_CPU).TypeConstraint<T>("T"), MatrixOp<CPUDevice>);

// Register the GPU kernels.
#ifdef GOOGLE_CUDA
extern template class MatrixFunctor<GPUDevice>;
REGISTER_KERNEL_BUILDER(Name("Matrix").Device(DEVICE_GPU).TypeConstraint<T>("T"), MatrixOp<GPUDevice>);
#endif  // GOOGLE_CUDA
*/

// Register the CPU kernels.
#define REGISTER_CPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("Matrix").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      MatrixOp<CPUDevice, T>);
REGISTER_CPU(COMPLEX_TYPE);

// Register the GPU kernels.
#ifdef GOOGLE_CUDA
#define REGISTER_GPU(T)                                          \
  /* Declare explicit instantiations in kernel_example.cu.cc. */ \
  extern template class MatrixFunctor<GPUDevice, T>;            \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("Matrix").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      MatrixOp<GPUDevice, T>);
REGISTER_GPU(COMPLEX_TYPE);
#endif  // GOOGLE_CUDA


template <typename T>
void matrix(const double* all_ps, const double* hel, const double* mdl_MT, const double* mdl_WT, const T* GC_10, const T* GC_11, 
            double* output_flat, const int nevents) {
    int ngraphs = 3;
    int nwavefuncs = 5;
    int ncolor = 2;
    double ZERO = 0.;
    
    T denom[2];
    denom[0] = 3;
    denom[1] = 3;
    
    T cf[4];
    cf[0] = 16;
    cf[1] = -2;
    cf[2] = -2;
    cf[3] = 16;
    
    // Begin code
    #pragma omp parallel for
    for (int i = 0; i < nevents; i++) {
        T w0[6], w1[6], w2[6], w3[6], w4[6];
        vxxxxx(all_ps+(16*i), ZERO, hel[0], -1, w0);
        vxxxxx(all_ps+(16*i+4), ZERO, hel[1], -1, w1);
        oxxxxx(all_ps+(16*i+8), mdl_MT[0], hel[2], +1, w2);
        ixxxxx(all_ps+(16*i+12), mdl_MT[0], hel[3], -1, w3);
        VVV1P0_1(w0, w1, GC_10[i], ZERO, ZERO, w4);
        
        // Amplitude(s) for diagram number 1
        
        T amp0;
        FFV1_0(w3, w2, w4, GC_11[i], amp0);
        FFV1_1(w2, w0, GC_11[i], mdl_MT[0], mdl_WT[0], w4);
        
        // Amplitude(s) for diagram number 2
        
        T amp1;
        FFV1_0(w3, w4, w1, GC_11[i], amp1);
        FFV1_2(w3, w0, GC_11[i], mdl_MT[0], mdl_WT[0], w4);
        
        // Amplitude(s) for diagram number 3
        
        T amp2;
        FFV1_0(w4, w2, w1, GC_11[i], amp2);
        
        T jamp[2] = {T(0, 1) * amp0 - amp1, -T(0, 1) * amp0 - amp2};
        
        T ret(0, 0);
        for (int a = 0; a < 2; a++) {
            for (int b = 0; b < 2; b++) {
                // ret = tf.einsum("ae, ab, be -> e", jamp, cf, tf.math.conj(jamp)/tf.reshape(denom, (ncolor, 1)))
                ret += (jamp[a] * cf[a * 2 + b] * std::conj(jamp[b])) / denom[0];
            }
        }
        output_flat[i] = ret.real();
    }
}

template <typename T>
void _ix_massive(const double* p, double fmass, double nsf, double nh, T* v) {
    double pp = std::min(p[0], sqrt(p[1] * p[1] + p[2] * p[2] + p[3] * p[3]));
    
    int ip = (int)(1 + nh) / 2;
    int im = (int)(1 - nh) / 2;
    
    if (pp == 0) {
        _ox_massive_pp_zero(fmass, nsf, im, ip, v);
    }
    else {
        _ix_massive_pp_nonzero(p, fmass, nsf, nh, ip, im, pp, v);
    }
}

template <typename T>
void _ix_massless(const double* p, double nhel, double nsf, double nh, T* v) {
    double sqp0p3 = sqrt(std::max(p[0] + p[3], 0.0)) * nsf;
    
    T chi1;
    if (sqp0p3 == 0) {
        _ix_massless_sqp0p3_zero(p, nhel, chi1);
    }
    else {
        _ix_massless_sqp0p3_nonzero(p, nh, sqp0p3, chi1);
    }
    
    T chi[] = {T(sqp0p3, 0.0), chi1};
    
    if (nh == 1) {
        _ix_massless_nh_one(chi, v);
    }
    else {
        _ix_massless_nh_not_one(chi, v);
    }
}

template <typename T>
void _ox_massless(const double* p, double nhel, double nsf, double nh, T* v) {
    double sqp0p3 = sqrt(std::max(p[0] + p[3], 0.0)) * nsf;
    double mult[] = {1, 1, -1, 1};
    
    T chi0;
    if (sqp0p3 == 0) {
        _ix_massless_sqp0p3_zero(p, nhel, chi0);
    }
    else {
        double prod[4];
        for (int i = 0; i < 4; i++)
            prod[i] = p[i] * mult[i];
        _ix_massless_sqp0p3_nonzero(prod, nh, sqp0p3, chi0);
    }
    
    T chi[2];
    chi[0] = chi0;
    chi[1] = T(sqp0p3, 0.0);
    
    if (nh == 1) {
        _ix_massless_nh_not_one(chi, v);
    }
    else {
        _ix_massless_nh_one(chi, v);
    }
}

template <typename T>
void _ox_massive(const double* p, double fmass, double nhel, double nsf, double nh, T* v) {
    double pp = std::min(p[0], sqrt(p[1] * p[1] + p[2] * p[2] + p[3] * p[3]));
    
    int ip = -((int)(1 - nh) / 2) * (int)nhel;
    int im =  ((int)(1 + nh) / 2) * (int)nhel;
    
    if (pp == 0) {
        _ox_massive_pp_zero(fmass, nsf, ip, im, v);
    }
    else {
        _ox_massive_pp_nonzero(p, fmass, nsf, nh, pp, v);
    }
}

template <typename T>
void _ox_massive_pp_zero(double fmass, double nsf, int ip, int im, T* v) {
    double sqm[2];
    sqm[0] = sqrt(fmass);
    sqm[1] = sign(sqm[0], fmass);
    
    v[0] = T((double)im * sqm[abs(im)], 0.0);
    v[1] = T((double)ip * nsf * sqm[abs(im)], 0.0);
    v[2] = T((double)im * nsf * sqm[abs(ip)], 0.0);
    v[3] = T((double)ip * sqm[abs(ip)], 0.0);
}

template <typename T>
void _ox_massive_pp_nonzero(const double* p, double fmass, double nsf, double nh, double pp, T* v) {
    double sf[] = {(1 + nsf + (1 - nsf) * nh) * 0.5, (1 + nsf - (1 - nsf) * nh) * 0.5};
    double omega[] = {sqrt(p[0] + pp), fmass / (sqrt(p[0] + pp))};
    
    int ip = (int) (1 + nh) / 2;
    int im = (int) (1 - nh) / 2;
    
    double sfomeg[] = {sf[0] * omega[ip], sf[1] * omega[im]};
    
    double pp3 = std::max(pp + p[3], 0.0);
    T chi1;
    if (pp3 == 0) {
        chi1 = T(-nh, 0);
    }
    else {
        chi1 = T(nh * p[1] / sqrt(2.0 * pp * pp3), -p[2] / sqrt(2.0 * pp * pp3));
    }
    T chi2(sqrt(pp3 * 0.5 / pp), 0.0);
    T chi[] = {chi2, chi1};
    
    v[0] = T(sfomeg[1], 0.0) * chi[im];
    v[1] = T(sfomeg[1], 0.0) * chi[ip];
    v[2] = T(sfomeg[0], 0.0) * chi[im];
    v[3] = T(sfomeg[0], 0.0) * chi[ip];
}

template <typename T>
void _ix_massless_sqp0p3_zero(const double* p, double nhel, T& val) {
    val = complex128(-nhel * sqrt(2.0 * p[0]), 0.0);
}

template <typename T>
void _ix_massless_sqp0p3_nonzero(const double* p, double nh, double sqp0p3, T& val) {
    val = complex128(nh * p[1] / sqp0p3, p[2] / sqp0p3);
}

template <typename T>
void _ix_massive_pp_nonzero(const double* p, double fmass, double nsf, double nh, int ip, int im, double pp, T* v) {
    double sf[] = {(1 + nsf + (1 - nsf) * nh) * 0.5, (1 + nsf - (1 - nsf) * nh) * 0.5};
    double omega[] = {sqrt(p[0] + pp), fmass / (sqrt(p[0] + pp))};
    
    double sfomeg[] = {sf[0] * omega[ip], sf[1] * omega[im]};
    
    double pp3 = std::max(pp + p[3], 0.0);
    T chi1;
    if (pp3 == 0) {
        chi1 = T(-nh, 0);
    }
    else {
        chi1 = T(nh * p[1] / sqrt(2.0 * pp * pp3), p[2] / sqrt(2.0 * pp * pp3));
    }
    T chi2(sqrt(pp3 * 0.5 / pp), 0.0);
    T chi[] = {chi2, chi1};
    
    v[0] = T(sfomeg[0], 0.0) * chi[im];
    v[1] = T(sfomeg[0], 0.0) * chi[ip];
    v[2] = T(sfomeg[1], 0.0) * chi[im];
    v[3] = T(sfomeg[1], 0.0) * chi[ip];
}

template <typename T>
void _ix_massless_nh_one(T* chi, T* v) {
    v[2] = chi[0];
    v[3] = chi[1];
    v[0] = CZERO;
    v[1] = CZERO;
}

template <typename T>
void _ix_massless_nh_not_one(T* chi, T* v) {
    v[0] = chi[1];
    v[1] = chi[0];
    v[2] = CZERO;
    v[3] = CZERO;
}

template <typename T>
void _vx_BRST_check(const double* p, double vmass, T* v) {
    if (vmass == 0) {
        _vx_BRST_check_massless(p, v);
    }
    else {
        _vx_BRST_check_massive(p, vmass, v);
    }
}

template <typename T>
void _vx_no_BRST_check(const double* p, double vmass, double nhel, double nsv, double hel0, double nsvahl, double pp, double pt, T* v) {
    if (vmass != 0) {
        _vx_no_BRST_check_massive(
                            p, vmass, nhel, hel0, nsvahl, pp, pt, v
                                                );
    }
    else {
        _vx_no_BRST_check_massless(p, nhel, nsv, v);
    }
}

template <typename T>
void _vx_BRST_check_massless(const double* p, T* v) {
    for (int i = 0; i < 4; i++) {
        v[i] = p[i]/p[0];
    }
}

template <typename T>
void _vx_BRST_check_massive(const double* p, double vmass, T* v) {
    for (int i = 0; i < 4; i++) {
        v[i] = p[i]/vmass;
    }
}

template <typename T>
void _vx_no_BRST_check_massive(const double* p, double vmass, double nhel, double hel0, double nsvahl, double pp, double pt, T* v) {
    if (pp == 0) {
        _vx_no_BRST_check_massive_pp_zero(nhel, nsvahl, v);
    }
    else {
        _vx_no_BRST_check_massive_pp_nonzero(
                        p, vmass, nhel, hel0, nsvahl, pp, pt, v
                                                    );
    }
}

template <typename T>
void _vx_no_BRST_check_massive_pp_zero(double nhel, double nsvahl, T* v) {
    double hel0 = 1.0 - abs(nhel);
    v[0] = T(1, 0);
    v[1] = T(-nhel * SQH, 0.0);
    v[2] = T(0.0, nsvahl * SQH);
    v[3] = T(hel0, 0.0);
}

template <typename T>
void _vx_no_BRST_check_massive_pp_nonzero(const double* p, double vmass, double nhel, double hel0, double nsvahl, double pp, double pt, T* v) {
    double emp = p[0] / (vmass * pp);
    T v2 = T(hel0 * pp / vmass, 0.0);
    T v5 = T(hel0 * p[3] * emp + (nhel * pt) / (pp * SQH), 0.0);
    
    T v34[2];
    if (pt != 0) {
        _vx_no_BRST_check_massive_pp_nonzero_pt_nonzero(p, nhel, hel0, nsvahl, pp, pt, emp, v34);
    }
    else {
        _vx_no_BRST_check_massive_pp_nonzero_pt_zero(p, nhel, nsvahl, v34);
    }
    v[0] = v2;
    v[1] = v34[0];
    v[2] = v34[1];
    v[3] = v5;
}

template <typename T>
void _vx_no_BRST_check_massive_pp_nonzero_pt_nonzero(const double* p, double nhel, double hel0, double nsvahl, double pp, double pt, double emp, T* v) {
    double pzpt = p[3] / (pp * pt) * SQH * nhel;
    v[0] = T(hel0 * p[1] * emp - p[1] * pzpt, -nsvahl * p[2] / pt * SQH);
    v[1] = T(hel0 * p[2] * emp - p[2] * pzpt, nsvahl * p[1] / pt * SQH);
}

template <typename T>
void _vx_no_BRST_check_massive_pp_nonzero_pt_zero(const double* p, double nhel, double nsvahl, T* v) {
    v[0] = T(-nhel * SQH, 0);
    v[1] = T(0.0, nsvahl * signvec(SQH, p[3]));
}

template <typename T>
void _vx_no_BRST_check_massless(const double* p, double nhel, double nsv, T* v) {
    double pp = p[0];
    double pt = sqrt(p[1] * p[1] + p[2] * p[2]);
    
    T v2 = T(0, 0);
    T v5 = T(nhel * pt / pp * SQH, 0);
    
    T v34[2];
    if (pt != 0) {
        _vx_no_BRST_check_massless_pt_nonzero(p, nhel, nsv, pp, pt, v34);
    }
    else {
        _vx_no_BRST_check_massless_pt_zero(p, nhel, nsv, v34);
    }
    
    v[0] = v2;
    v[1] = v34[0];
    v[2] = v34[1];
    v[3] = v5;
}

template <typename T>
void _vx_no_BRST_check_massless_pt_nonzero(const double* p, double nhel, double nsv, double pp, double pt, T* v) {
    double pzpt = p[3] / (pp * pt) * SQH * nhel;
    
    v[0] = T(-p[1] * pzpt, -nsv * p[2] / pt * SQH);
    v[1] = T(-p[2] * pzpt, nsv * p[1] / pt * SQH);
}

template <typename T>
void _vx_no_BRST_check_massless_pt_zero(const double* p, double nhel, double nsv, T* v) {
    v[0] = T(-nhel * SQH, 0);
    v[1] = T(0, nsv * signvec(SQH, p[3]));
}

double sign(double x, double y) {
    int sign = 0;
    y >= 0 ? sign = 1 : sign = -1;
    return x * sign;
}

double signvec(double x, double y) {
    return sign(x, y);
}

template <typename T>
void vxxxxx(const double* p, double vmass, double nhel, double nsv, T* ret) {
    T v0 = T(p[0] * nsv, p[3] * nsv);
    T v1 = T(p[1] * nsv, p[2] * nsv);
    
    double pt2 = p[1] * p[1] + p[2] * p[2];
    double pp = std::min(p[0], sqrt(pt2 + p[3] * p[3]));
    double pt = std::min(pp, sqrt(pt2));
    
    double hel0 = 1 - abs(nhel);
    double nsvahl = nsv * abs(nhel);
    
    T v[4];
    
    if (nhel == 4) {
        _vx_BRST_check(p, vmass, v);
    }
    else {
        _vx_no_BRST_check(p, vmass, nhel, nsv, hel0, nsvahl, pp, pt, v);
    }
    
    ret[0] = v0;
    ret[1] = v1;
    for (int i = 0; i < 4; i++)
        ret[i+2] = v[i];
}

template <typename T>
void ixxxxx(const double* p, double fmass, double nhel, double nsf, T* ret) {
    T v0 = T(-p[0] * nsf, -p[3] * nsf);
    T v1 = T(-p[1] * nsf, -p[2] * nsf);
    
    double nh = nhel * nsf;
    
    T v[4];
    
    if (fmass != 0) {
        _ix_massive(p, fmass, nsf, nh, v);
    }
    else {
        _ix_massless(p, nhel, nsf, nh, v);
    }
    
    ret[0] = v0;
    ret[1] = v1;
    for (int i = 0; i < 4; i++)
        ret[i+2] = v[i];
}

template <typename T>
void oxxxxx(const double* p, double fmass, double nhel, double nsf, T* ret) {
    T v0 = T(p[0] * nsf, p[3] * nsf);
    T v1 = T(p[1] * nsf, p[2] * nsf);
    
    double nh = nhel * nsf;
    
    T v[4];
    
    if (fmass != 0) {
        _ox_massive(p, fmass, nhel, nsf, nh, v);
    }
    else {
        _ox_massless(p, nhel, nsf, nh, v);
    }
    
    ret[0] = v0;
    ret[1] = v1;
    for (int i = 0; i < 4; i++)
        ret[i+2] = v[i];
}

template <typename T>
void VVV1P0_1(T* V2, T* V3, const T COUP, double M1_double, double W1_double, T* V1) {
    
    // V2 -> 6-component vector
    // V3 -> 6-component vector
    
    T cI(0, 1);
    T M1 = M1_double;
    T W1 = W1_double;
    //complex128 COUP = COUP_comp;
    
    T P2[4];
    P2[0] = T(V2[0].real(), 0.0);
    P2[1] = T(V2[1].real(), 0.0);
    P2[2] = T(V2[1].imag(), 0.0);
    P2[3] = T(V2[0].imag(), 0.0);
    
    T P3[4];
    P3[0] = T(V3[0].real(), 0.0);
    P3[1] = T(V3[1].real(), 0.0);
    P3[2] = T(V3[1].imag(), 0.0);
    P3[3] = T(V3[0].imag(), 0.0);
    
    V1[0] = V2[0] + V3[0];
    V1[1] = V2[1] + V3[1];
    
    T P1[4];
    P1[0] = T(-V1[0].real(), 0.0);
    P1[1] = T(-V1[1].real(), 0.0);
    P1[2] = T(-V1[1].imag(), 0.0);
    P1[3] = T(-V1[0].imag(), 0.0);
    
    T TMP0 = (V3[2]*P1[0] - V3[3]*P1[1] - V3[4]*P1[2] - V3[5]*P1[3]);
    T TMP1 = (V3[2]*P2[0] - V3[3]*P2[1] - V3[4]*P2[2] - V3[5]*P2[3]);
    T TMP2 = (P1[0]*V2[2] - P1[1]*V2[3] - P1[2]*V2[4] - P1[3]*V2[5]);
    T TMP3 = (V2[2]*P3[0] - V2[3]*P3[1] - V2[4]*P3[2] - V2[5]*P3[3]);
    T TMP4 = (V3[2]*V2[2] - V3[3]*V2[3] - V3[4]*V2[4] - V3[5]*V2[5]);
    
    T denom = COUP/(P1[0]*P1[0] - P1[1]*P1[1] - P1[2]*P1[2] - P1[3]*P1[3] - M1 * (M1 -cI* W1));
    
    V1[2]= denom * (TMP4 * (-cI*(P2[0]) + cI*(P3[0])) + (V2[2]*(-cI*(TMP0) + cI*(TMP1)) + V3[2]*(cI*(TMP2) - cI*(TMP3))));
    V1[3]= denom * (TMP4 * (-cI*(P2[1]) + cI*(P3[1])) + (V2[3]*(-cI*(TMP0) + cI*(TMP1)) + V3[3]*(cI*(TMP2) - cI*(TMP3))));
    V1[4]= denom * (TMP4 * (-cI*(P2[2]) + cI*(P3[2])) + (V2[4]*(-cI*(TMP0) + cI*(TMP1)) + V3[4]*(cI*(TMP2) - cI*(TMP3))));
    V1[5]= denom * (TMP4 * (-cI*(P2[3]) + cI*(P3[3])) + (V2[5]*(-cI*(TMP0) + cI*(TMP1)) + V3[5]*(cI*(TMP2) - cI*(TMP3))));
}

template <typename T>
void FFV1_0(T* F1, T* F2, T* V3, const T COUP, T& amp) {
    T cI(0, 1);
    //complex128 COUP = COUP_comp;
    T TMP5 = (F1[2]*(F2[4]*(V3[2]+V3[5])+F2[5]*(V3[3]+cI*(V3[4])))+(F1[3]*(F2[4]*(V3[3]-cI*(V3[4]))+F2[5]*(V3[2]-V3[5]))+(F1[4]*(F2[2]*(V3[2]-V3[5])-F2[3]*(V3[3]+cI*(V3[4])))+F1[5]*(F2[2]*(-V3[3]+cI*(V3[4]))+F2[3]*(V3[2]+V3[5])))));
    amp = COUP*-cI * TMP5;
}

template <typename T>
void FFV1_1(T* F2, T* V3, const T COUP, double M1_double, double W1_double, T* F1) {
    T cI(0, 1);
    T M1 = M1_double;
    T W1 = W1_double;
    //complex128 COUP = COUP_comp;
    
    F1[0] = F2[0] + V3[0];
    F1[1] = F2[1] + V3[1];
    
    T P1[4];
    P1[0] = T(-F1[0].real(), 0.0);
    P1[1] = T(-F1[1].real(), 0.0);
    P1[2] = T(-F1[1].imag(), 0.0);
    P1[3] = T(-F1[0].imag(), 0.0);
    
    T denom = COUP/(P1[0]*P1[0] - P1[1]*P1[1] - P1[2]*P1[2] - P1[3]*P1[3] - M1 * (M1 -cI* W1));
    
    F1[2]= denom*cI*(F2[2]*(P1[0]*(-V3[2]+V3[5])+(P1[1]*(V3[3]-cI*(V3[4]))+(P1[2]*(cI*(V3[3])+V3[4])+P1[3]*(-V3[2]+V3[5]))))+(F2[3]*(P1[0]*(V3[3]+cI*(V3[4]))+(P1[1]*(-1./1.)*(V3[2]+V3[5])+(P1[2]*(-1./1.)*(cI*(V3[2]+V3[5]))+P1[3]*(V3[3]+cI*(V3[4])))))+M1*(F2[4]*(V3[2]+V3[5])+F2[5]*(V3[3]+cI*(V3[4])))));
    F1[3]= denom*(-cI)*(F2[2]*(P1[0]*(-V3[3]+cI*(V3[4]))+(P1[1]*(V3[2]-V3[5])+(P1[2]*(-cI*(V3[2])+cI*(V3[5]))+P1[3]*(V3[3]-cI*(V3[4])))))+(F2[3]*(P1[0]*(V3[2]+V3[5])+(P1[1]*(-1./1.)*(V3[3]+cI*(V3[4]))+(P1[2]*(cI*(V3[3])-V3[4])-P1[3]*(V3[2]+V3[5]))))+M1*(F2[4]*(-V3[3]+cI*(V3[4]))+F2[5]*(-V3[2]+V3[5]))));
    F1[4]= denom*(-cI)*(F2[4]*(P1[0]*(V3[2]+V3[5])+(P1[1]*(-V3[3]+cI*(V3[4]))+(P1[2]*(-1./1.)*(cI*(V3[3])+V3[4])-P1[3]*(V3[2]+V3[5]))))+(F2[5]*(P1[0]*(V3[3]+cI*(V3[4]))+(P1[1]*(-V3[2]+V3[5])+(P1[2]*(-cI*(V3[2])+cI*(V3[5]))-P1[3]*(V3[3]+cI*(V3[4])))))+M1*(F2[2]*(-V3[2]+V3[5])+F2[3]*(V3[3]+cI*(V3[4])))));
    F1[5]= denom*cI*(F2[4]*(P1[0]*(-V3[3]+cI*(V3[4]))+(P1[1]*(V3[2]+V3[5])+(P1[2]*(-1./1.)*(cI*(V3[2]+V3[5]))+P1[3]*(-V3[3]+cI*(V3[4])))))+(F2[5]*(P1[0]*(-V3[2]+V3[5])+(P1[1]*(V3[3]+cI*(V3[4]))+(P1[2]*(-cI*(V3[3])+V3[4])+P1[3]*(-V3[2]+V3[5]))))+M1*(F2[2]*(-V3[3]+cI*(V3[4]))+F2[3]*(V3[2]+V3[5]))));
    
    //return F1;
}

template <typename T>
void FFV1_2(T* F1, T* V3, const T COUP, double M2_double, double W2_double, T* F2) {
    T cI(0, 1);
    T M2 = M2_double;
    T W2 = W2_double;
    //complex128 COUP = COUP_comp;
    
    F2[0] = F1[0] + V3[0];
    F2[1] = F1[1] + V3[1];
    
    T P2[4];
    P2[0] = T(-F2[0].real(), 0.0);
    P2[1] = T(-F2[1].real(), 0.0);
    P2[2] = T(-F2[1].imag(), 0.0);
    P2[3] = T(-F2[0].imag(), 0.0);
    
    T denom = COUP/(P2[0]*P2[0] - P2[1]*P2[1] - P2[2]*P2[2] - P2[3]*P2[3] - M2 * (M2 -cI* W2));
    
    F2[2]= denom*cI*(F1[2]*(P2[0]*(V3[2]+V3[5])+(P2[1]*(-1./1.)*(V3[3]+cI*(V3[4]))+(P2[2]*(cI*(V3[3])-V3[4])-P2[3]*(V3[2]+V3[5]))))+(F1[3]*(P2[0]*(V3[3]-cI*(V3[4]))+(P2[1]*(-V3[2]+V3[5])+(P2[2]*(cI*(V3[2])-cI*(V3[5]))+P2[3]*(-V3[3]+cI*(V3[4])))))+M2*(F1[4]*(V3[2]-V3[5])+F1[5]*(-V3[3]+cI*(V3[4])))));
    F2[3]= denom*(-cI)*(F1[2]*(P2[0]*(-1./1.)*(V3[3]+cI*(V3[4]))+(P2[1]*(V3[2]+V3[5])+(P2[2]*(cI*(V3[2]+V3[5]))-P2[3]*(V3[3]+cI*(V3[4])))))+(F1[3]*(P2[0]*(-V3[2]+V3[5])+(P2[1]*(V3[3]-cI*(V3[4]))+(P2[2]*(cI*(V3[3])+V3[4])+P2[3]*(-V3[2]+V3[5]))))+M2*(F1[4]*(V3[3]+cI*(V3[4]))-F1[5]*(V3[2]+V3[5]))));
    F2[4]= denom*(-cI)*(F1[4]*(P2[0]*(-V3[2]+V3[5])+(P2[1]*(V3[3]+cI*(V3[4]))+(P2[2]*(-cI*(V3[3])+V3[4])+P2[3]*(-V3[2]+V3[5]))))+(F1[5]*(P2[0]*(V3[3]-cI*(V3[4]))+(P2[1]*(-1./1.)*(V3[2]+V3[5])+(P2[2]*(cI*(V3[2]+V3[5]))+P2[3]*(V3[3]-cI*(V3[4])))))+M2*(F1[2]*(-1./1.)*(V3[2]+V3[5])+F1[3]*(-V3[3]+cI*(V3[4])))));
    F2[5]= denom*cI*(F1[4]*(P2[0]*(-1./1.)*(V3[3]+cI*(V3[4]))+(P2[1]*(V3[2]-V3[5])+(P2[2]*(cI*(V3[2])-cI*(V3[5]))+P2[3]*(V3[3]+cI*(V3[4])))))+(F1[5]*(P2[0]*(V3[2]+V3[5])+(P2[1]*(-V3[3]+cI*(V3[4]))+(P2[2]*(-1./1.)*(cI*(V3[3])+V3[4])-P2[3]*(V3[2]+V3[5]))))+M2*(F1[2]*(V3[3]+cI*(V3[4]))+F1[3]*(V3[2]-V3[5]))));
}


REGISTER_OP("Vxxxxx")
    .Input("all_ps: double")
    .Input("zero: double")
    .Input("hel: double")
    .Input("m1: double")
    .Input("num: int32")
    .Input("correct_shape: complex128")
    .Output("vx: complex128")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(5));
      return Status::OK();
    });


class VxxxxxOp : public OpKernel {
 public:
  explicit VxxxxxOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& all_ps_tensor = context->input(0);
    auto all_ps = all_ps_tensor.flat<double>().data();
    
    const Tensor& zero_tensor = context->input(1);
    auto zero = zero_tensor.flat<double>().data();
    
    const Tensor& hel_tensor = context->input(2);
    auto hel = hel_tensor.flat<double>().data();
    
    const Tensor& m1_tensor = context->input(3);
    auto m1 = m1_tensor.flat<double>().data();
    
    const Tensor& num_tensor = context->input(4);
    auto num = num_tensor.flat<int>().data();
    
    const Tensor& correct_shape = context->input(5);

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, correct_shape.shape(),
                                                     &output_tensor));
    auto output_flat = output_tensor->flat<complex128>();
    
    // Begin code
    
    int nevents = all_ps_tensor.shape().dim_size(0);
    int output_slice_size = 6;
    complex128 jamp[output_slice_size * nevents];
    
    for (int i = 0; i < nevents; i++) {
        complex128 w0[6];
        vxxxxx(all_ps+(16*i + 4*num[0]), *zero, *hel, *m1, w0);
        
        for (int j = 0; j < output_slice_size; j++) {
            jamp[j * nevents + i] = w0[j];
        }
    }
    
    for (int i = 0; i < output_slice_size * nevents; i++) {
      output_flat(i) = jamp[i];
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("Vxxxxx").Device(DEVICE_CPU), VxxxxxOp);

REGISTER_OP("Oxxxxx")
    .Input("all_ps: double")
    .Input("zero: double")
    .Input("hel: double")
    .Input("m1: double")
    .Input("correct_shape: complex128")
    .Output("vx: complex128")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(4));
      return Status::OK();
    });


class OxxxxxOp : public OpKernel {
 public:
  explicit OxxxxxOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& all_ps_tensor = context->input(0);
    auto all_ps = all_ps_tensor.flat<double>().data();
    
    const Tensor& zero_tensor = context->input(1);
    auto zero = zero_tensor.flat<double>().data();
    
    const Tensor& hel_tensor = context->input(2);
    auto hel = hel_tensor.flat<double>().data();
    
    const Tensor& m1_tensor = context->input(3);
    auto m1 = m1_tensor.flat<double>().data();
    
    const Tensor& correct_shape = context->input(4);

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, correct_shape.shape(),
                                                     &output_tensor));
    auto output_flat = output_tensor->flat<complex128>();
    
    // Begin code
    
    int nevents = all_ps_tensor.shape().dim_size(0);
    int output_slice_size = 6;
    complex128 jamp[output_slice_size * nevents];
    
    for (int i = 0; i < nevents; i++) {
        complex128 w0[6];
        oxxxxx(all_ps+(16*i+8), *zero, *hel, *m1, w0);
        
        for (int j = 0; j < output_slice_size; j++) {
            jamp[j * nevents + i] = w0[j];
        }
    }
    
    for (int i = 0; i < output_slice_size * nevents; i++) {
      output_flat(i) = jamp[i];
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("Oxxxxx").Device(DEVICE_CPU), OxxxxxOp);

REGISTER_OP("Ixxxxx")
    .Input("all_ps: double")
    .Input("zero: double")
    .Input("hel: double")
    .Input("m1: double")
    .Input("correct_shape: complex128")
    .Output("vx: complex128")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(4));
      return Status::OK();
    });


class IxxxxxOp : public OpKernel {
 public:
  explicit IxxxxxOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& all_ps_tensor = context->input(0);
    auto all_ps = all_ps_tensor.flat<double>().data();
    
    const Tensor& zero_tensor = context->input(1);
    auto zero = zero_tensor.flat<double>().data();
    
    const Tensor& hel_tensor = context->input(2);
    auto hel = hel_tensor.flat<double>().data();
    
    const Tensor& m1_tensor = context->input(3);
    auto m1 = m1_tensor.flat<double>().data();
    
    const Tensor& correct_shape = context->input(4);

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, correct_shape.shape(),
                                                     &output_tensor));
    auto output_flat = output_tensor->flat<complex128>();
    
    // Begin code
    
    int nevents = all_ps_tensor.shape().dim_size(0);
    int output_slice_size = 6;
    complex128 jamp[output_slice_size * nevents];
    
    for (int i = 0; i < nevents; i++) {
        complex128 w0[6];
        ixxxxx(all_ps+(16*i+12), *zero, *hel, *m1, w0);
        
        for (int j = 0; j < output_slice_size; j++) {
            jamp[j * nevents + i] = w0[j];
        }
    }
    
    for (int i = 0; i < output_slice_size * nevents; i++) {
      output_flat(i) = jamp[i];
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("Ixxxxx").Device(DEVICE_CPU), IxxxxxOp);

REGISTER_OP("Vxnobrstcheck")
    .Input("all_ps: double")
    .Input("vmass: double")
    .Input("nhel: double")
    .Input("nsv: double")
    .Input("hel0: double")
    .Input("nsvalh: double")
    .Input("pp: double")
    .Input("pt: double")
    .Input("correct_shape: complex128")
    .Output("vx: complex128")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(8));
      return Status::OK();
    });


class VxnobrstcheckOp : public OpKernel {
 public:
  explicit VxnobrstcheckOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& all_ps_tensor = context->input(0);
    auto all_ps = all_ps_tensor.flat<double>().data();
    
    const Tensor& vmass_tensor = context->input(1);
    auto vmass = vmass_tensor.flat<double>().data();
    
    const Tensor& nhel_tensor = context->input(2);
    auto nhel = nhel_tensor.flat<double>().data();
    
    const Tensor& nsv_tensor = context->input(3);
    auto nsv = nsv_tensor.flat<double>().data();
    
    const Tensor& hel0_tensor = context->input(4);
    auto hel0 = hel0_tensor.flat<double>().data();
    
    const Tensor& nsvalh_tensor = context->input(5);
    auto nsvahl = nsvalh_tensor.flat<double>().data();
    
    const Tensor& pp_tensor = context->input(6);
    auto pp = pp_tensor.flat<double>().data();
    
    const Tensor& pt_tensor = context->input(7);
    auto pt = pt_tensor.flat<double>().data();
    
    const Tensor& correct_shape = context->input(8);

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, correct_shape.shape(),
                                                     &output_tensor));
    auto output_flat = output_tensor->flat<complex128>();
    
    // Begin code
    
    int nevents = all_ps_tensor.shape().dim_size(0);
    int output_slice_size = 4;
    complex128 jamp[output_slice_size * nevents];
    
    for (int i = 0; i < nevents; i++) {
        complex128 w0[6];
        _vx_no_BRST_check(all_ps+(16*i), *vmass, *nhel, *nsv, *hel0, *nsvahl, *pp, *pt, w0);
        
        for (int j = 0; j < output_slice_size; j++) {
            jamp[i * output_slice_size + j] = w0[j];
        }
    }
    
    for (int i = 0; i < output_slice_size * nevents; i++) {
      output_flat(i) = jamp[i];
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("Vxnobrstcheck").Device(DEVICE_CPU), VxnobrstcheckOp);

REGISTER_OP("Vvv1p01")
    .Input("all_ps: double")
    .Input("vmass: double")
    .Input("v2: complex128")
    .Input("v3: complex128")
    .Input("coup: complex128")
    .Input("m1: double")
    .Input("w1: double")
    .Input("mdl_mt: double")
    .Input("correct_shape: complex128")
    .Output("vx: complex128")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(8));
      return Status::OK();
    });

class Vvv1p01Op : public OpKernel {
 public:
  explicit Vvv1p01Op(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& all_ps_tensor = context->input(0);
    auto all_ps = all_ps_tensor.flat<double>().data();
    
    const Tensor& hel_tensor = context->input(1);
    auto hel = hel_tensor.flat<double>().data();
    
    const Tensor& V2_tensor = context->input(2);
    auto V2_v = V2_tensor.flat<complex128>().data();
    
    const Tensor& V3_tensor = context->input(3);
    auto V3_v = V3_tensor.flat<complex128>().data();
    
    const Tensor& COUP_tensor = context->input(4);
    auto COUP = COUP_tensor.flat<complex128>().data();
    
    const Tensor& M1_tensor = context->input(5);
    auto M1 = M1_tensor.flat<double>().data();
    
    const Tensor& W1_tensor = context->input(6);
    auto W1 = W1_tensor.flat<double>().data();
    
    const Tensor& mdl_MT_tensor = context->input(7);
    auto mdl_MT = mdl_MT_tensor.flat<double>().data();
    
    const Tensor& correct_shape = context->input(8);

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, correct_shape.shape(),
                                                     &output_tensor));
    auto output_flat = output_tensor->flat<complex128>();
    
    // Begin code
    
    int nevents = all_ps_tensor.shape().dim_size(0);
    int output_slice_size = 6;
    complex128 jamp[output_slice_size * nevents];
    
    for (int i = 0; i < nevents; i++) {
        complex128 V2[output_slice_size];
        complex128 V3[output_slice_size];
        
        for (int j = 0; j < output_slice_size; j++) {
            V2[j] = V2_v[j * nevents + i];
            V3[j] = V3_v[j * nevents + i];
        }
        complex128 w4[6];
        VVV1P0_1(V2, V3, COUP[i], *M1, *W1, w4);
        
        for (int j = 0; j < output_slice_size; j++) {
            jamp[j * nevents + i] = w4[j];
        }
    }
    
    for (int i = 0; i < output_slice_size * nevents; i++) {
      output_flat(i) = jamp[i];
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("Vvv1p01").Device(DEVICE_CPU), Vvv1p01Op);

REGISTER_OP("Ffv10")
    .Input("all_ps: double")
    .Input("vmass: double")
    .Input("w3: complex128")
    .Input("w2: complex128")
    .Input("w4: complex128")
    .Input("coup0: complex128")
    .Input("coup1: complex128")
    .Input("mdl_mt: double")
    .Input("correct_shape: complex128")
    .Output("vx: complex128")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(8));
      return Status::OK();
    });

//camp0 = MatrixOp.ffv10(all_ps, hel, w3, w2, w4, GC_10, GC_11, mdl_MT, pw4)
class Ffv10Op : public OpKernel {
 public:
  explicit Ffv10Op(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& all_ps_tensor = context->input(0);
    auto all_ps = all_ps_tensor.flat<double>().data();
    
    const Tensor& hel_tensor = context->input(1);
    auto hel = hel_tensor.flat<double>().data();
    
    const Tensor& w3_tensor = context->input(2);
    auto w3_v = w3_tensor.flat<complex128>().data();
    
    const Tensor& w2_tensor = context->input(3);
    auto w2_v = w2_tensor.flat<complex128>().data();
    
    const Tensor& w4_tensor = context->input(4);
    auto w4_v = w4_tensor.flat<complex128>().data();
    
    const Tensor& COUP0_tensor = context->input(5);
    auto COUP0 = COUP0_tensor.flat<complex128>().data();
    
    const Tensor& COUP1_tensor = context->input(6);
    auto COUP1 = COUP1_tensor.flat<complex128>().data();
    
    const Tensor& mdl_MT_tensor = context->input(7);
    auto mdl_MT = mdl_MT_tensor.flat<double>().data();
    
    const Tensor& correct_shape = context->input(8);

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, correct_shape.shape(),
                                                     &output_tensor));
    auto output_flat = output_tensor->flat<complex128>();
    
    // Begin code
    
    int nevents = all_ps_tensor.shape().dim_size(0);
    int output_slice_size = 1;
    int vector_slice_size = 6;
    std::vector<complex128> jamp(output_slice_size * nevents, complex128(0,0));
    
    double ZERO = 0;
    
    for (int i = 0; i < nevents; i++) {
        complex128 w2[vector_slice_size];
        complex128 w3[vector_slice_size];
        complex128 w4[vector_slice_size];
        
        for (int j = 0; j < vector_slice_size; j++) {
            w2[j] = w2_v[j * nevents + i];
            w3[j] = w3_v[j * nevents + i];
            w4[j] = w4_v[j * nevents + i];
        }
        complex128 amp0;
        FFV1_0(w3, w2, w4, COUP1[i], amp0);
        
        for (int j = 0; j < output_slice_size; j++) {
            jamp[j * nevents + i] = amp0;
        }
    }
    
    for (int i = 0; i < output_slice_size * nevents; i++) {
      output_flat(i) = jamp[i];
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("Ffv10").Device(DEVICE_CPU), Ffv10Op);

REGISTER_OP("Ffv11")
    .Input("all_ps: double")
    .Input("vmass: double")
    .Input("w2: complex128")
    .Input("w0: complex128")
    .Input("coup0: complex128")
    .Input("coup1: complex128")
    .Input("mdl_mt: double")
    .Input("mdl_wt: double")
    .Input("correct_shape: complex128")
    .Output("vx: complex128")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(8));
      return Status::OK();
    });

//pw4 = MatrixOp.ffv11(all_ps, hel, w2, w0, GC_10, GC_11, mdl_MT, mdl_WT, pamp0)
class Ffv11Op : public OpKernel {
 public:
  explicit Ffv11Op(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& all_ps_tensor = context->input(0);
    auto all_ps = all_ps_tensor.flat<double>().data();
    
    const Tensor& hel_tensor = context->input(1);
    auto hel = hel_tensor.flat<double>().data();
    
    const Tensor& w2_tensor = context->input(2);
    auto w2_v = w2_tensor.flat<complex128>().data();
    
    const Tensor& w0_tensor = context->input(3);
    auto w0_v = w0_tensor.flat<complex128>().data();
    
    const Tensor& COUP0_tensor = context->input(4);
    auto COUP0 = COUP0_tensor.flat<complex128>().data();
    
    const Tensor& COUP1_tensor = context->input(5);
    auto COUP1 = COUP1_tensor.flat<complex128>().data();
    
    const Tensor& mdl_MT_tensor = context->input(6);
    auto mdl_MT = mdl_MT_tensor.flat<double>().data();
    
    const Tensor& mdl_WT_tensor = context->input(7);
    auto mdl_WT = mdl_WT_tensor.flat<double>().data();
    
    const Tensor& correct_shape = context->input(8);

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, correct_shape.shape(),
                                                     &output_tensor));
    auto output_flat = output_tensor->flat<complex128>();
    
    // Begin code
    
    int nevents = all_ps_tensor.shape().dim_size(0);
    int output_slice_size = 6;
    int vector_slice_size = 6;
    std::vector<complex128> jamp(output_slice_size * nevents, complex128(0,0));
    
    double ZERO = 0;
    
    for (int i = 0; i < nevents; i++) {
        complex128 w2[vector_slice_size];
        complex128 w0[vector_slice_size];
        
        for (int j = 0; j < vector_slice_size; j++) {
            w2[j] = w2_v[j * nevents + i];
            w0[j] = w0_v[j * nevents + i];
        }
        complex128 w4[6];
        FFV1_1(w2, w0, COUP1[i], *mdl_MT, *mdl_WT, w4);
        //auto w4 = FFV1_1(w2, w0, *COUP1, *mdl_MT, *mdl_WT);
        
        for (int j = 0; j < output_slice_size; j++) {
            jamp[j * nevents + i] = w4[j];
        }
    }
    
    for (int i = 0; i < output_slice_size * nevents; i++) {
      output_flat(i) = jamp[i];
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("Ffv11").Device(DEVICE_CPU), Ffv11Op);

REGISTER_OP("Ffv12")
    .Input("all_ps: double")
    .Input("vmass: double")
    .Input("w2: complex128")
    .Input("w3: complex128")
    .Input("coup0: complex128")
    .Input("coup1: complex128")
    .Input("mdl_mt: double")
    .Input("mdl_wt: double")
    .Input("correct_shape: complex128")
    .Output("vx: complex128")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(8));
      return Status::OK();
    });

//pw4 = MatrixOp.ffv12(all_ps, hel, cw3, cw0, GC_10, GC_11, mdl_MT, mdl_WT, w4)
class Ffv12Op : public OpKernel {
 public:
  explicit Ffv12Op(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& all_ps_tensor = context->input(0);
    auto all_ps = all_ps_tensor.flat<double>().data();
    
    const Tensor& hel_tensor = context->input(1);
    auto hel = hel_tensor.flat<double>().data();
    
    const Tensor& w3_tensor = context->input(2);
    auto w3_v = w3_tensor.flat<complex128>().data();
    
    const Tensor& w0_tensor = context->input(3);
    auto w0_v = w0_tensor.flat<complex128>().data();
    
    const Tensor& COUP0_tensor = context->input(4);
    auto COUP0 = COUP0_tensor.flat<complex128>().data();
    
    const Tensor& COUP1_tensor = context->input(5);
    auto COUP1 = COUP1_tensor.flat<complex128>().data();
    
    const Tensor& mdl_MT_tensor = context->input(6);
    auto mdl_MT = mdl_MT_tensor.flat<double>().data();
    
    const Tensor& mdl_WT_tensor = context->input(7);
    auto mdl_WT = mdl_WT_tensor.flat<double>().data();
    
    const Tensor& correct_shape = context->input(8);

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, correct_shape.shape(),
                                                     &output_tensor));
    auto output_flat = output_tensor->flat<complex128>();
    
    // Begin code
    
    int nevents = all_ps_tensor.shape().dim_size(0);
    int output_slice_size = 6;
    int vector_slice_size = 6;
    std::vector<complex128> jamp(output_slice_size * nevents, complex128(0,0));
    
    double ZERO = 0;

    for (int i = 0; i < nevents; i++) {
        complex128 w3[vector_slice_size];
        complex128 w0[vector_slice_size];
        
        for (int j = 0; j < vector_slice_size; j++) {
            w3[j] = w3_v[j * nevents + i];
            w0[j] = w0_v[j * nevents + i];
        }
        complex128 w4[6];
        FFV1_2(w3, w0, COUP1[i], *mdl_MT, *mdl_WT, w4);
        //auto w4 = FFV1_2(w3, w0, *COUP1, *mdl_MT, *mdl_WT);
        
        for (int j = 0; j < output_slice_size; j++) {
            jamp[j * nevents + i] = w4[j];
        }
    }
    
    for (int i = 0; i < output_slice_size * nevents; i++) {
      output_flat(i) = jamp[i];
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("Ffv12").Device(DEVICE_CPU), Ffv12Op);

REGISTER_OP("Stacktest")
    .Input("amp0: complex128")
    .Input("amp1: complex128")
    .Input("amp2: complex128")
    .Input("correct_shape: complex128")
    .Output("vx: complex128")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(3));
      return Status::OK();
    });

//jamp = MatrixOp.stack(amp0, amp1, amp2, jamp)
class StackOp : public OpKernel {
 public:
  explicit StackOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    
    const Tensor& amp0_tensor = context->input(0);
    auto amp0 = amp0_tensor.flat<complex128>().data();
    
    const Tensor& amp1_tensor = context->input(1);
    auto amp1 = amp1_tensor.flat<complex128>().data();
    
    const Tensor& amp2_tensor = context->input(2);
    auto amp2 = amp2_tensor.flat<complex128>().data();
    
    const Tensor& correct_shape = context->input(3);

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, correct_shape.shape(),
                                                     &output_tensor));
    auto output_flat = output_tensor->flat<complex128>();
    
    // Begin code
    
    int nevents = amp0_tensor.shape().dim_size(0);
    int output_slice_size = 2;
    complex128 jamp[output_slice_size * nevents];
    
    for (int i = 0; i < nevents; i++) {
        jamp[i + 0 * nevents] =  complex128(0, 1) * amp0[i] - amp1[i];
        jamp[i + 1 * nevents] = -complex128(0, 1) * amp0[i] - amp2[i];
    }
    
    for (int i = 0; i < output_slice_size * nevents; i++) {
      output_flat(i) = jamp[i];
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("Stacktest").Device(DEVICE_CPU), StackOp);

/*
End
*/

