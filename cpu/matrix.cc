#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/math_ops.h"

#include <vector>

using namespace tensorflow;

#define USE_PYTHON 1

// mdl_MT,mdl_WT,GC_10,GC_11

REGISTER_OP("Matrix")
    .Input("all_ps: double")
    .Input("hel: double")
    .Input("mdl_mt: double")
    .Input("mdl_wt: double")
    .Input("gc_10: complex128")
    .Input("gc_11: complex128")
    .Input("correct_shape: double")
    .Output("matrix_element: double")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(6));
      return Status::OK();
    });

int nevents = 2;
double SQH = sqrt(0.5);
complex128 CZERO = complex128(0.0, 0.0);
std::vector<double> matrix(const double*, const double*, const double, const double, const complex128, const complex128);
std::vector<complex128> vxxxxx(double* p, double fmass, double nhel, double nsf);
std::vector<complex128> ixxxxx(double* p, double fmass, double nhel, double nsf);
std::vector<complex128> oxxxxx(double* p, double fmass, double nhel, double nsf);
std::vector<complex128> _ix_massive(double* p, double fmass, double nsf, double nh);
std::vector<complex128> _ix_massless(double* p, double nhel, double nsf, double nh);
std::vector<complex128> _ox_massless(double* p, double nhel, double nsf, double nh);
std::vector<complex128> _ox_massive(double* p, double fmass, double nhel, double nsf, double nh);
complex128 _ix_massless_sqp0p3_zero(double* p, double nhel);
complex128 _ix_massless_sqp0p3_nonzero(double* p, double nh, double sqp0p3);
std::vector<complex128> _ix_massive_pp_nonzero(double* p, double fmass, double nsf, double nh, int ip, int im, double pp);
std::vector<complex128> _ix_massless_nh_one(complex128* chi);
std::vector<complex128> _ix_massless_nh_not_one(complex128* chi);
std::vector<complex128> _ox_massive_pp_zero(double fmass, double nsf, int ip, int im);
std::vector<complex128> _ox_massive_pp_nonzero(double* p, double fmass, double nsf, double nh, double pp);
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
std::vector<complex128> _vx_no_BRST_check_massless_pt_nonzero(double* p, double nhel, double nsv, double pp, double pt);
std::vector<complex128> _vx_no_BRST_check_massless_pt_zero(double* p, double nhel, double nsv);

std::vector<complex128> VVV1P0_1(std::vector<complex128> V2, std::vector<complex128> V3, const complex128 COUP, double M1, double W1);
complex128 FFV1_0(std::vector<complex128> F1, std::vector<complex128> F2, std::vector<complex128> V3, const complex128 COUP);
std::vector<complex128> FFV1_1(std::vector<complex128> F2, std::vector<complex128> V3, const complex128 COUP, double M1, double W1);
std::vector<complex128> FFV1_2(std::vector<complex128> F1, std::vector<complex128> V3, const complex128 COUP, double M1, double W1);

double sign(double x, double y);
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
    
    const Tensor& correct_shape = context->input(6);

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, correct_shape.shape(),
                                                     &output_tensor));
    auto output_flat = output_tensor->flat<double>();

    auto matrixElement = matrix(all_ps, hel, *mdl_MT, *mdl_WT, *GC_10, *GC_11);
    
    for (int i = 0; i < nevents; i++) {
      output_flat(i) = matrixElement[i];
    }
  }
};

std::vector<double> matrix(const double* all_ps, const double* hel, const double mdl_MT, const double mdl_WT, const complex128 GC_10, const complex128 GC_11) {
    int ngraphs = 3;
    int nwavefuncs = 5;
    int ncolor = 2;
    double ZERO = 0.;
    
    std::vector<complex128> denom(2, complex128(0,0));
    denom[0] = 3;
    denom[1] = 3;
    
    std::vector<complex128> cf(4, complex128(0,0));
    cf[0] = 16;
    cf[1] = -2;
    cf[2] = -2;
    cf[3] = 16;
    
    // Begin code
    
    std::vector<complex128> jamp(2 * nevents, complex128(0,0));
    std::vector<complex128> ret(2, complex128(0,0));
    
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
        auto w2 = oxxxxx(all_ps_2, mdl_MT, hel[2], +1);
        auto w3 = ixxxxx(all_ps_3, mdl_MT, hel[3], -1);
        auto w4 = VVV1P0_1(w0, w1, GC_10, ZERO, ZERO);
        
        // Amplitude(s) for diagram number 1
        
        auto amp0 = FFV1_0(w3, w2, w4, GC_11);
        w4 = FFV1_1(w2, w0, GC_11, mdl_MT, mdl_WT);
        
        // Amplitude(s) for diagram number 2
        
        auto amp1 = FFV1_0(w3,w4,w1,GC_11);
        w4= FFV1_2(w3, w0, GC_11, mdl_MT, mdl_WT);
        
        // Amplitude(s) for diagram number 3
        
        auto amp2 = FFV1_0(w4, w2, w1, GC_11);
        
        jamp[i + 0 * nevents] =  complex128(0, 1) * amp0 - amp1;
        jamp[i + 1 * nevents] = -complex128(0, 1) * amp0 - amp2;
    }
    /*
    std::vector<complex128> ret(2, complex128(0,0));*/
    for (int e = 0; e < nevents; e++) {
        ret[e] = 0;
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                // ret = tf.einsum("ie, ij, je -> e", jamp, cf, tf.math.conj(jamp)/tf.reshape(denom, (ncolor, 1)))
                ret[e] += (jamp[i * nevents + e] * cf[i * 2+ j]) * (std::conj(jamp[j * nevents + e]) / denom[e]);
            }
        }
    }
    
    std::vector<double> ret_re(2, double(0));
    for (int e = 0; e < nevents; e++) {
        ret_re[e] = ret[e].real();
    }
    return ret_re;
}

std::vector<complex128> _ix_massive(double* p, double fmass, double nsf, double nh) {
    double pp = std::min(p[0], sqrt(p[1] * p[1] + p[2] * p[2] + p[3] * p[3]));
    
    int ip = (int)(1 + nh) / 2;
    int im = (int)(1 - nh) / 2;
    
    if (pp == 0) {
        return _ox_massive_pp_zero(fmass, nsf, im, ip);
    }
    else {
        return _ix_massive_pp_nonzero(p, fmass, nsf, nh, ip, im, pp);
    }
}

std::vector<complex128> _ix_massless(double* p, double nhel, double nsf, double nh) {
    double sqp0p3 = sqrt(std::max(p[0] + p[3], 0.0)) * nsf;
    
    complex128 chi1;
    if (sqp0p3 == 0) {
        chi1 = _ix_massless_sqp0p3_zero(p, nhel);
    }
    else {
        chi1 = _ix_massless_sqp0p3_nonzero(p, nh, sqp0p3);
    }
    
    complex128 chi[] = {complex128(sqp0p3, 0.0), chi1};
    
    if (nh == 1) {
        return _ix_massless_nh_one(chi);
    }
    else {
        return _ix_massless_nh_not_one(chi);
    }
}

std::vector<complex128> _ox_massless(double* p, double nhel, double nsf, double nh) {
    double sqp0p3 = sqrt(std::max(p[0] + p[3], 0.0)) * nsf;
    double mult[] = {1, 1, -1, 1};
    
    complex128 chi0;
    if (sqp0p3 == 0) {
        chi0 = _ix_massless_sqp0p3_zero(p, nhel);
    }
    else {
        double prod[4];
        for (int i = 0; i < 4; i++)
            prod[i] = p[i] * mult[i];
        chi0 = _ix_massless_sqp0p3_nonzero(prod, nh, sqp0p3);
    }
    
    complex128 chi[2];
    chi[0] = chi0;
    chi[1] = complex128(sqp0p3, 0.0);
    
    if (nh == 1) {
        return _ix_massless_nh_not_one(chi);
    }
    else {
        return _ix_massless_nh_one(chi);
    }
}

std::vector<complex128> _ox_massive(double* p, double fmass, double nhel, double nsf, double nh) {
    double pp = std::min(p[0], sqrt(p[1] * p[1] + p[2] * p[2] + p[3] * p[3]));
    
    int ip = -((int)(1 - nh) / 2) * (int)nhel;
    int im =  ((int)(1 + nh) / 2) * (int)nhel;
    
    if (pp == 0) {
        return _ox_massive_pp_zero(fmass, nsf, ip, im);
    }
    else {
        return _ox_massive_pp_nonzero(p, fmass, nsf, nh, pp);
    }
}

std::vector<complex128> _ox_massive_pp_zero(double fmass, double nsf, int ip, int im) {
    double sqm[2];
    sqm[0] = sqrt(fmass);
    sqm[1] = sign(sqm[0], fmass);
    
    std::vector<complex128> v(4, complex128(0,0));
    
    v[0] = complex128((double)im * sqm[abs(im)], 0.0);
    v[1] = complex128((double)ip * nsf * sqm[abs(im)], 0.0);
    v[2] = complex128((double)im * nsf * sqm[abs(ip)], 0.0);
    v[3] = complex128((double)ip * sqm[abs(ip)], 0.0);
    
    return v;
}

std::vector<complex128> _ox_massive_pp_nonzero(double* p, double fmass, double nsf, double nh, double pp) {
    double sf[] = {(1 + nsf + (1 - nsf) * nh) * 0.5, (1 + nsf - (1 - nsf) * nh) * 0.5};
    double omega[] = {sqrt(p[0] + pp), fmass / (sqrt(p[0] + pp))};
    
    int ip = (int) (1 + nh) / 2;
    int im = (int) (1 - nh) / 2;
    
    double sfomeg[] = {sf[0] * omega[ip], sf[1] * omega[im]};
    
    double pp3 = std::max(pp + p[3], 0.0);
    complex128 chi1;
    if (pp3 == 0) {
        chi1 = complex128(-nh, 0);
    }
    else {
        chi1 = complex128(nh * p[1] / sqrt(2.0 * pp * pp3), -p[2] / sqrt(2.0 * pp * pp3));
    }
    complex128 chi2(sqrt(pp3 * 0.5 / pp), 0.0);
    complex128 chi[] = {chi2, chi1};
    
    std::vector<complex128> v(4, complex128(0,0));
    
    v[0] = complex128(sfomeg[1], 0.0) * chi[im];
    v[1] = complex128(sfomeg[1], 0.0) * chi[ip];
    v[2] = complex128(sfomeg[0], 0.0) * chi[im];
    v[3] = complex128(sfomeg[0], 0.0) * chi[ip];
    
    return v;
}

complex128 _ix_massless_sqp0p3_zero(double* p, double nhel) {
    return complex128(-nhel * sqrt(2.0 * p[0]), 0.0);
}

complex128 _ix_massless_sqp0p3_nonzero(double* p, double nh, double sqp0p3) {
    return complex128(nh * p[1] / sqp0p3, p[2] / sqp0p3);
}

std::vector<complex128> _ix_massive_pp_nonzero(double* p, double fmass, double nsf, double nh, int ip, int im, double pp) {
    double sf[] = {(1 + nsf + (1 - nsf) * nh) * 0.5, (1 + nsf - (1 - nsf) * nh) * 0.5};
    double omega[] = {sqrt(p[0] + pp), fmass / (sqrt(p[0] + pp))};
    
    double sfomeg[] = {sf[0] * omega[ip], sf[1] * omega[im]};
    
    double pp3 = std::max(pp + p[3], 0.0);
    complex128 chi1;
    if (pp3 == 0) {
        chi1 = complex128(-nh, 0);
    }
    else {
        chi1 = complex128(nh * p[1] / sqrt(2.0 * pp * pp3), p[2] / sqrt(2.0 * pp * pp3));
    }
    complex128 chi2(sqrt(pp3 * 0.5 / pp), 0.0);
    complex128 chi[] = {chi2, chi1};
    
    std::vector<complex128> v(4, complex128(0,0));
    
    v[0] = complex128(sfomeg[0], 0.0) * chi[im];
    v[1] = complex128(sfomeg[0], 0.0) * chi[ip];
    v[2] = complex128(sfomeg[1], 0.0) * chi[im];
    v[3] = complex128(sfomeg[1], 0.0) * chi[ip];
    
    return v;
}

std::vector<complex128> _ix_massless_nh_one(complex128* chi) {
    std::vector<complex128> v(4, complex128(0,0));
    
    v[2] = chi[0];
    v[3] = chi[1];
    v[0] = CZERO;
    v[1] = CZERO;
    return v;
}

std::vector<complex128> _ix_massless_nh_not_one(complex128* chi) {
    std::vector<complex128> v(4, complex128(0,0));
    
    v[0] = chi[1];
    v[1] = chi[0];
    v[2] = CZERO;
    v[3] = CZERO;
    return v;
}

std::vector<complex128> _vx_BRST_check(double* p, double vmass) {
    if (vmass == 0) {
        return _vx_BRST_check_massless(p);
    }
    else {
        return _vx_BRST_check_massive(p, vmass);
    }
}

std::vector<complex128> _vx_no_BRST_check(double* p, double vmass, double nhel, double nsv, double hel0, double nsvahl, double pp, double pt) {
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
    double pp = p[0];
    double pt = sqrt(p[1] * p[1] + p[2] * p[2]);
    std::vector<complex128> v(4, complex128(0,0));
    complex128 v2 = complex128(0, 0);
    complex128 v5 = complex128(nhel * pt / pp * SQH, 0);
    
    std::vector<complex128> v34(2, complex128(0,0));
    if (pt != 0) {
        v34 = _vx_no_BRST_check_massless_pt_nonzero(p, nhel, nsv, pp, pt);
    }
    else {
        v34 = _vx_no_BRST_check_massless_pt_zero(p, nhel, nsv);
    }
    
    v[0] = v2;
    v[1] = v34[0];
    v[2] = v34[1];
    v[3] = v5;
    
    return v;
}

std::vector<complex128> _vx_no_BRST_check_massless_pt_nonzero(double* p, double nhel, double nsv, double pp, double pt) {
    double pzpt = p[3] / (pp * pt) * SQH * nhel;
    
    std::vector<complex128> v(2, complex128(0,0));
    v[0] = complex128(-p[1] * pzpt, -nsv * p[2] / pt * SQH);
    v[1] = complex128(-p[2] * pzpt, nsv * p[1] / pt * SQH);
    
    return v;
}

std::vector<complex128> _vx_no_BRST_check_massless_pt_zero(double* p, double nhel, double nsv) {
    std::vector<complex128> v(2, complex128(0,0));
    v[0] = complex128(-nhel * SQH, 0);
    v[1] = complex128(0, nsv * signvec(SQH, p[3]));
    
    return v;
}

double sign(double x, double y) {
    int sign = 0;
    y >= 0 ? sign = 1 : sign = -1;
    return x * sign;
}

double signvec(double x, double y) {
    return sign(x, y);
}

std::vector<complex128> vxxxxx(double* p, double vmass, double nhel, double nsv) {
    complex128 v0 = complex128(p[0] * nsv, p[3] * nsv);
    complex128 v1 = complex128(p[1] * nsv, p[2] * nsv);
    
    double pt2 = p[1] * p[1] + p[2] * p[2];
    double pp = std::min(p[0], sqrt(pt2 + p[3] * p[3]));
    double pt = std::min(pp, sqrt(pt2));
    
    double hel0 = 1 - abs(nhel);
    double nsvahl = nsv * abs(nhel);
    
    std::vector<complex128> v(4, complex128(0,0));
    std::vector<complex128> ret(6, complex128(0,0));
    
    if (nhel == 4) {
        v = _vx_BRST_check(p, vmass);
    }
    else {
        v = _vx_no_BRST_check(p, vmass, nhel, nsv, hel0, nsvahl, pp, pt);
    }
    
    ret[0] = v0;
    ret[1] = v1;
    for (int i = 0; i < 4; i++)
        ret[i+2] = v[i];
    
    return ret;
}

std::vector<complex128> ixxxxx(double* p, double fmass, double nhel, double nsf) {
    complex128 v0 = complex128(-p[0] * nsf, -p[3] * nsf);
    complex128 v1 = complex128(-p[1] * nsf, -p[2] * nsf);
    
    double nh = nhel * nsf;
    
    std::vector<complex128> v(4, complex128(0,0));
    std::vector<complex128> ret(6, complex128(0,0));
    
    if (fmass != 0) {
        v = _ix_massive(p, fmass, nsf, nh);
    }
    else {
        v = _ix_massless(p, nhel, nsf, nh);
    }
    
    ret[0] = v0;
    ret[1] = v1;
    for (int i = 0; i < 4; i++)
        ret[i+2] = v[i];
    
    return ret;
}

std::vector<complex128> oxxxxx(double* p, double fmass, double nhel, double nsf) {
    complex128 v0 = complex128(p[0] * nsf, p[3] * nsf);
    complex128 v1 = complex128(p[1] * nsf, p[2] * nsf);
    
    double nh = nhel * nsf;
    
    std::vector<complex128> v(4, complex128(0,0));
    std::vector<complex128> ret(6, complex128(0,0));
    
    if (fmass != 0) {
        v = _ox_massive(p, fmass, nhel, nsf, nh);
    }
    else {
        v = _ox_massless(p, nhel, nsf, nh);
    }
    
    ret[0] = v0;
    ret[1] = v1;
    for (int i = 0; i < 4; i++)
        ret[i+2] = v[i];
    
    return ret;
}

std::vector<complex128> VVV1P0_1(std::vector<complex128> V2, std::vector<complex128> V3, const complex128 COUP_comp, double M1_double, double W1_double) {
    
    // V2 -> 6-component vector
    // V3 -> 6-component vector
    
    complex128 cI(0, 1);
    complex128 M1 = M1_double;
    complex128 W1 = W1_double;
    complex128 COUP = COUP_comp;
    
    std::vector<complex128> P2(4, complex128(0,0));
    P2[0] = complex128(V2[0].real(), 0.0);
    P2[1] = complex128(V2[1].real(), 0.0);
    P2[2] = complex128(V2[1].imag(), 0.0);
    P2[3] = complex128(V2[0].imag(), 0.0);
    
    std::vector<complex128> P3(4, complex128(0,0));
    P3[0] = complex128(V3[0].real(), 0.0);
    P3[1] = complex128(V3[1].real(), 0.0);
    P3[2] = complex128(V3[1].imag(), 0.0);
    P3[3] = complex128(V3[0].imag(), 0.0);
    
    std::vector<complex128> V1(6, complex128(0,0));
    V1[0] = V2[0] + V3[0];
    V1[1] = V2[1] + V3[1];
    
    std::vector<complex128> P1(4, complex128(0,0));
    P1[0] = complex128(-V1[0].real(), 0.0);
    P1[1] = complex128(-V1[1].real(), 0.0);
    P1[2] = complex128(-V1[1].imag(), 0.0);
    P1[3] = complex128(-V1[0].imag(), 0.0);
    
    complex128 TMP0 = (V3[2]*P1[0] - V3[3]*P1[1] - V3[4]*P1[2] - V3[5]*P1[3]);
    complex128 TMP1 = (V3[2]*P2[0] - V3[3]*P2[1] - V3[4]*P2[2] - V3[5]*P2[3]);
    complex128 TMP2 = (P1[0]*V2[2] - P1[1]*V2[3] - P1[2]*V2[4] - P1[3]*V2[5]);
    complex128 TMP3 = (V2[2]*P3[0] - V2[3]*P3[1] - V2[4]*P3[2] - V2[5]*P3[3]);
    complex128 TMP4 = (V3[2]*V2[2] - V3[3]*V2[3] - V3[4]*V2[4] - V3[5]*V2[5]);
    
    complex128 denom = COUP/(P1[0]*P1[0] - P1[1]*P1[1] - P1[2]*P1[2] - P1[3]*P1[3] - M1 * (M1 -cI* W1));
    
    V1[2]= denom * (TMP4 * (-cI*(P2[0]) + cI*(P3[0])) + (V2[2]*(-cI*(TMP0) + cI*(TMP1)) + V3[2]*(cI*(TMP2) - cI*(TMP3))));
    V1[3]= denom * (TMP4 * (-cI*(P2[1]) + cI*(P3[1])) + (V2[3]*(-cI*(TMP0) + cI*(TMP1)) + V3[3]*(cI*(TMP2) - cI*(TMP3))));
    V1[4]= denom * (TMP4 * (-cI*(P2[2]) + cI*(P3[2])) + (V2[4]*(-cI*(TMP0) + cI*(TMP1)) + V3[4]*(cI*(TMP2) - cI*(TMP3))));
    V1[5]= denom * (TMP4 * (-cI*(P2[3]) + cI*(P3[3])) + (V2[5]*(-cI*(TMP0) + cI*(TMP1)) + V3[5]*(cI*(TMP2) - cI*(TMP3))));
    
    return V1;
}

complex128 FFV1_0(std::vector<complex128> F1, std::vector<complex128> F2, std::vector<complex128> V3, const complex128 COUP_comp) {
    complex128 cI(0, 1);
    complex128 COUP = COUP_comp;
    complex128 TMP5 = (F1[2]*(F2[4]*(V3[2]+V3[5])+F2[5]*(V3[3]+cI*(V3[4])))+(F1[3]*(F2[4]*(V3[3]-cI*(V3[4]))+F2[5]*(V3[2]-V3[5]))+(F1[4]*(F2[2]*(V3[2]-V3[5])-F2[3]*(V3[3]+cI*(V3[4])))+F1[5]*(F2[2]*(-V3[3]+cI*(V3[4]))+F2[3]*(V3[2]+V3[5])))));
    complex128 vertex = COUP*-cI * TMP5;
    return vertex;
}

std::vector<complex128> FFV1_1(std::vector<complex128> F2, std::vector<complex128> V3, const complex128 COUP_comp, double M1_double, double W1_double) {
    complex128 cI(0, 1);
    complex128 M1 = M1_double;
    complex128 W1 = W1_double;
    complex128 COUP = COUP_comp;
    
    std::vector<complex128> F1(6, complex128(0,0));
    F1[0] = F2[0] + V3[0];
    F1[1] = F2[1] + V3[1];
    
    std::vector<complex128> P1(4, complex128(0,0));
    P1[0] = complex128(-F1[0].real(), 0.0);
    P1[1] = complex128(-F1[1].real(), 0.0);
    P1[2] = complex128(-F1[1].imag(), 0.0);
    P1[3] = complex128(-F1[0].imag(), 0.0);
    
    complex128 denom = COUP/(P1[0]*P1[0] - P1[1]*P1[1] - P1[2]*P1[2] - P1[3]*P1[3] - M1 * (M1 -cI* W1));
    
    F1[2]= denom*cI*(F2[2]*(P1[0]*(-V3[2]+V3[5])+(P1[1]*(V3[3]-cI*(V3[4]))+(P1[2]*(cI*(V3[3])+V3[4])+P1[3]*(-V3[2]+V3[5]))))+(F2[3]*(P1[0]*(V3[3]+cI*(V3[4]))+(P1[1]*(-1./1.)*(V3[2]+V3[5])+(P1[2]*(-1./1.)*(cI*(V3[2]+V3[5]))+P1[3]*(V3[3]+cI*(V3[4])))))+M1*(F2[4]*(V3[2]+V3[5])+F2[5]*(V3[3]+cI*(V3[4])))));
    F1[3]= denom*(-cI)*(F2[2]*(P1[0]*(-V3[3]+cI*(V3[4]))+(P1[1]*(V3[2]-V3[5])+(P1[2]*(-cI*(V3[2])+cI*(V3[5]))+P1[3]*(V3[3]-cI*(V3[4])))))+(F2[3]*(P1[0]*(V3[2]+V3[5])+(P1[1]*(-1./1.)*(V3[3]+cI*(V3[4]))+(P1[2]*(cI*(V3[3])-V3[4])-P1[3]*(V3[2]+V3[5]))))+M1*(F2[4]*(-V3[3]+cI*(V3[4]))+F2[5]*(-V3[2]+V3[5]))));
    F1[4]= denom*(-cI)*(F2[4]*(P1[0]*(V3[2]+V3[5])+(P1[1]*(-V3[3]+cI*(V3[4]))+(P1[2]*(-1./1.)*(cI*(V3[3])+V3[4])-P1[3]*(V3[2]+V3[5]))))+(F2[5]*(P1[0]*(V3[3]+cI*(V3[4]))+(P1[1]*(-V3[2]+V3[5])+(P1[2]*(-cI*(V3[2])+cI*(V3[5]))-P1[3]*(V3[3]+cI*(V3[4])))))+M1*(F2[2]*(-V3[2]+V3[5])+F2[3]*(V3[3]+cI*(V3[4])))));
    F1[5]= denom*cI*(F2[4]*(P1[0]*(-V3[3]+cI*(V3[4]))+(P1[1]*(V3[2]+V3[5])+(P1[2]*(-1./1.)*(cI*(V3[2]+V3[5]))+P1[3]*(-V3[3]+cI*(V3[4])))))+(F2[5]*(P1[0]*(-V3[2]+V3[5])+(P1[1]*(V3[3]+cI*(V3[4]))+(P1[2]*(-cI*(V3[3])+V3[4])+P1[3]*(-V3[2]+V3[5]))))+M1*(F2[2]*(-V3[3]+cI*(V3[4]))+F2[3]*(V3[2]+V3[5]))));
    
    return F1;
}

std::vector<complex128> FFV1_2(std::vector<complex128> F1, std::vector<complex128> V3, const complex128 COUP_comp, double M2_double, double W2_double) {
    complex128 cI(0, 1);
    complex128 M2 = M2_double;
    complex128 W2 = W2_double;
    complex128 COUP = COUP_comp;
    
    std::vector<complex128> F2(6, complex128(0,0));
    F2[0] = F1[0] + V3[0];
    F2[1] = F1[1] + V3[1];
    
    std::vector<complex128> P2(4, complex128(0,0));
    P2[0] = complex128(-F2[0].real(), 0.0);
    P2[1] = complex128(-F2[1].real(), 0.0);
    P2[2] = complex128(-F2[1].imag(), 0.0);
    P2[3] = complex128(-F2[0].imag(), 0.0);
    
    complex128 denom = COUP/(P2[0]*P2[0] - P2[1]*P2[1] - P2[2]*P2[2] - P2[3]*P2[3] - M2 * (M2 -cI* W2));
    
    F2[2]= denom*cI*(F1[2]*(P2[0]*(V3[2]+V3[5])+(P2[1]*(-1./1.)*(V3[3]+cI*(V3[4]))+(P2[2]*(cI*(V3[3])-V3[4])-P2[3]*(V3[2]+V3[5]))))+(F1[3]*(P2[0]*(V3[3]-cI*(V3[4]))+(P2[1]*(-V3[2]+V3[5])+(P2[2]*(cI*(V3[2])-cI*(V3[5]))+P2[3]*(-V3[3]+cI*(V3[4])))))+M2*(F1[4]*(V3[2]-V3[5])+F1[5]*(-V3[3]+cI*(V3[4])))));
    F2[3]= denom*(-cI)*(F1[2]*(P2[0]*(-1./1.)*(V3[3]+cI*(V3[4]))+(P2[1]*(V3[2]+V3[5])+(P2[2]*(cI*(V3[2]+V3[5]))-P2[3]*(V3[3]+cI*(V3[4])))))+(F1[3]*(P2[0]*(-V3[2]+V3[5])+(P2[1]*(V3[3]-cI*(V3[4]))+(P2[2]*(cI*(V3[3])+V3[4])+P2[3]*(-V3[2]+V3[5]))))+M2*(F1[4]*(V3[3]+cI*(V3[4]))-F1[5]*(V3[2]+V3[5]))));
    F2[4]= denom*(-cI)*(F1[4]*(P2[0]*(-V3[2]+V3[5])+(P2[1]*(V3[3]+cI*(V3[4]))+(P2[2]*(-cI*(V3[3])+V3[4])+P2[3]*(-V3[2]+V3[5]))))+(F1[5]*(P2[0]*(V3[3]-cI*(V3[4]))+(P2[1]*(-1./1.)*(V3[2]+V3[5])+(P2[2]*(cI*(V3[2]+V3[5]))+P2[3]*(V3[3]-cI*(V3[4])))))+M2*(F1[2]*(-1./1.)*(V3[2]+V3[5])+F1[3]*(-V3[3]+cI*(V3[4])))));
    F2[5]= denom*cI*(F1[4]*(P2[0]*(-1./1.)*(V3[3]+cI*(V3[4]))+(P2[1]*(V3[2]-V3[5])+(P2[2]*(cI*(V3[2])-cI*(V3[5]))+P2[3]*(V3[3]+cI*(V3[4])))))+(F1[5]*(P2[0]*(V3[2]+V3[5])+(P2[1]*(-V3[3]+cI*(V3[4]))+(P2[2]*(-1./1.)*(cI*(V3[3])+V3[4])-P2[3]*(V3[2]+V3[5]))))+M2*(F1[2]*(V3[3]+cI*(V3[4]))+F1[3]*(V3[2]-V3[5]))));
    
    return F2;
}

REGISTER_KERNEL_BUILDER(Name("Matrix").Device(DEVICE_CPU), MatrixOp);


REGISTER_OP("Vxxxxx")
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
    
    const Tensor& correct_shape = context->input(4);

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, correct_shape.shape(),
                                                     &output_tensor));
    auto output_flat = output_tensor->flat<complex128>();
    
    // Begin code
    
    int output_slice_size = 6;
    std::vector<complex128> jamp(output_slice_size * nevents, complex128(0,0));
    
    for (int i = 0; i < nevents; i++) {
        double all_ps_0[4];
        for (int j = 0; j < 4; j++) {
            all_ps_0[j] = all_ps[16 * i + j];
        } 
        auto w0 = vxxxxx(all_ps_0, *zero, *hel, *m1);
        
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
    
    int output_slice_size = 6;
    std::vector<complex128> jamp(output_slice_size * nevents, complex128(0,0));
    
    for (int i = 0; i < nevents; i++) {
        double all_ps_2[4];
        for (int j = 0; j < 4; j++) {
            all_ps_2[j] = all_ps[16 * i + j + 8];
        } 
        auto w0 = oxxxxx(all_ps_2, *zero, *hel, *m1);
        
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
    
    int output_slice_size = 6;
    std::vector<complex128> jamp(output_slice_size * nevents, complex128(0,0));
    
    for (int i = 0; i < nevents; i++) {
        double all_ps_3[4];
        for (int j = 0; j < 4; j++) {
            all_ps_3[j] = all_ps[16 * i + j + 12];
        } 
        auto w0 = ixxxxx(all_ps_3, *zero, *hel, *m1);
        
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


//std::vector<complex128> _vx_no_BRST_check(double* p, double vmass, double nhel, double nsv, double hel0, double nsvahl, double pp, double pt)
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
    int output_slice_size = 4;
    std::vector<complex128> jamp(output_slice_size * nevents, complex128(0,0));
    
    for (int i = 0; i < nevents; i++) {
        double all_ps_0[4];
        for (int j = 0; j < 4; j++) {
            all_ps_0[j] = all_ps[16 * i + j];
        } 
        auto w0 = _vx_no_BRST_check(all_ps_0, *vmass, *nhel, *nsv, *hel0, *nsvahl, *pp, *pt);
        
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

//std::vector<complex128> VVV1P0_1(std::vector<complex128> V2, std::vector<complex128> V3, const complex128 COUP_comp, double M1_double, double W1_double)
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
    int output_slice_size = 6;
    std::vector<complex128> jamp(output_slice_size * nevents, complex128(0,0));
    
#if USE_PYTHON==1
    for (int i = 0; i < nevents; i++) {
        std::vector<complex128> V2(output_slice_size, complex128(0,0));
        std::vector<complex128> V3(output_slice_size, complex128(0,0));
        
        for (int j = 0; j < output_slice_size; j++) {
            V2[j] = V2_v[j * nevents + i];
            V3[j] = V3_v[j * nevents + i];
        }
        auto w4 = VVV1P0_1(V2, V3, *COUP, *M1, *W1);
        
        for (int j = 0; j < output_slice_size; j++) {
            jamp[j * nevents + i] = w4[j];
        }
    }
    
    for (int i = 0; i < output_slice_size * nevents; i++) {
      output_flat(i) = jamp[i];
    }
#else
    // Alternative way using all the Custom Op functions
    double ZERO = 0;
    
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
        auto w2 = oxxxxx(all_ps_2, ZERO, hel[2], +1);
        auto w3 = ixxxxx(all_ps_3, *mdl_MT, hel[3], -1);
        auto w4 = VVV1P0_1(w0, w1, *COUP, ZERO, ZERO);
        
        for (int j = 0; j < output_slice_size; j++) {
            jamp[j * nevents + i] = w4[j];
        }
    }
    
    for (int i = 0; i < output_slice_size * nevents; i++) {
      output_flat(i) = jamp[i];
    }
#endif
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
//complex128 FFV1_0(std::vector<complex128> F1, std::vector<complex128> F2, std::vector<complex128> V3, const complex128 COUP_comp)
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
    int output_slice_size = 1;
    int vector_slice_size = 6;
    std::vector<complex128> jamp(output_slice_size * nevents, complex128(0,0));
    
    double ZERO = 0;
#if USE_PYTHON==1
    for (int i = 0; i < nevents; i++) {
        std::vector<complex128> w2(vector_slice_size, complex128(0,0));
        std::vector<complex128> w3(vector_slice_size, complex128(0,0));
        std::vector<complex128> w4(vector_slice_size, complex128(0,0));
        
        for (int j = 0; j < vector_slice_size; j++) {
            w2[j] = w2_v[j * nevents + i];
            w3[j] = w3_v[j * nevents + i];
            w4[j] = w4_v[j * nevents + i];
        }
        auto amp0 = FFV1_0(w3, w2, w4, *COUP1);
        
        for (int j = 0; j < output_slice_size; j++) {
            jamp[j * nevents + i] = amp0;
        }
    }
    
    for (int i = 0; i < output_slice_size * nevents; i++) {
      output_flat(i) = jamp[i];
    }
#else
    // Alternative way using all the Custom Op functions
    
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
        auto w2 = oxxxxx(all_ps_2, ZERO, hel[2], +1);
        auto w3 = ixxxxx(all_ps_3, *mdl_MT, hel[3], -1);
        auto w4 = VVV1P0_1(w0, w1, *COUP0, ZERO, ZERO);
        
        // Amplitude(s) for diagram number 1
        
        auto amp0 = FFV1_0(w3, w2, w4, *COUP1);
        
        for (int j = 0; j < output_slice_size; j++) {
            jamp[j * nevents + i] = amp0;
        }
    }
    
    for (int i = 0; i < output_slice_size * nevents; i++) {
      output_flat(i) = jamp[i];
    }
#endif
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
//complex128 FFV1_0(std::vector<complex128> F1, std::vector<complex128> F2, std::vector<complex128> V3, const complex128 COUP_comp)
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
    int output_slice_size = 6;
    int vector_slice_size = 6;
    std::vector<complex128> jamp(output_slice_size * nevents, complex128(0,0));
    
    double ZERO = 0;
#if USE_PYTHON==1
    for (int i = 0; i < nevents; i++) {
        std::vector<complex128> w2(vector_slice_size, complex128(0,0));
        std::vector<complex128> w0(vector_slice_size, complex128(0,0));
        
        for (int j = 0; j < vector_slice_size; j++) {
            w2[j] = w2_v[j * nevents + i];
            w0[j] = w0_v[j * nevents + i];
        }
        auto w4 = FFV1_1(w2, w0, *COUP1, *mdl_MT, *mdl_WT);
        
        for (int j = 0; j < output_slice_size; j++) {
            jamp[j * nevents + i] = w4[j];
        }
    }
    
    for (int i = 0; i < output_slice_size * nevents; i++) {
      output_flat(i) = jamp[i];
    }
#else
    // Alternative way using all the Custom Op functions
    
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
        auto w2 = oxxxxx(all_ps_2, ZERO, hel[2], +1);
        auto w3 = ixxxxx(all_ps_3, *mdl_MT, hel[3], -1);
        auto w4 = VVV1P0_1(w0, w1, *COUP0, ZERO, ZERO);
        
        // Amplitude(s) for diagram number 1
        
        auto amp0 = FFV1_0(w3, w2, w4, *COUP1);
        w4 = FFV1_1(w2, w0, *COUP1, *mdl_MT, *mdl_WT);
        
        for (int j = 0; j < output_slice_size; j++) {
            jamp[j * nevents + i] = w4[j];
        }
    }
    
    for (int i = 0; i < output_slice_size * nevents; i++) {
      output_flat(i) = jamp[i];
    }
#endif
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
//complex128 FFV1_0(std::vector<complex128> F1, std::vector<complex128> F2, std::vector<complex128> V3, const complex128 COUP_comp)
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
    int output_slice_size = 6;
    int vector_slice_size = 6;
    std::vector<complex128> jamp(output_slice_size * nevents, complex128(0,0));
    
    double ZERO = 0;
#if USE_PYTHON==1
    for (int i = 0; i < nevents; i++) {
        std::vector<complex128> w3(vector_slice_size, complex128(0,0));
        std::vector<complex128> w0(vector_slice_size, complex128(0,0));
        
        for (int j = 0; j < vector_slice_size; j++) {
            w3[j] = w3_v[j * nevents + i];
            w0[j] = w0_v[j * nevents + i];
        }
        auto w4 = FFV1_2(w3, w0, *COUP1, *mdl_MT, *mdl_WT);
        
        for (int j = 0; j < output_slice_size; j++) {
            jamp[j * nevents + i] = w4[j];
        }
    }
    
    for (int i = 0; i < output_slice_size * nevents; i++) {
      output_flat(i) = jamp[i];
    }
#else
    // Alternative way using all the Custom Op functions
    
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
        auto w2 = oxxxxx(all_ps_2, ZERO, hel[2], +1);
        auto w3 = ixxxxx(all_ps_3, *mdl_MT, hel[3], -1);
        auto w4 = VVV1P0_1(w0, w1, *COUP0, ZERO, ZERO);
        
        // Amplitude(s) for diagram number 1
        
        auto amp0 = FFV1_0(w3, w2, w4, *COUP1);
        w4 = FFV1_2(w3, w0, *COUP1, *mdl_MT, *mdl_WT);
        
        for (int j = 0; j < output_slice_size; j++) {
            jamp[j * nevents + i] = w4[j];
        }
    }
    
    for (int i = 0; i < output_slice_size * nevents; i++) {
      output_flat(i) = jamp[i];
    }
#endif
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
//complex128 FFV1_0(std::vector<complex128> F1, std::vector<complex128> F2, std::vector<complex128> V3, const complex128 COUP_comp)
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
    int output_slice_size = 2;
    std::vector<complex128> jamp(output_slice_size * nevents, complex128(0,0));
    
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


