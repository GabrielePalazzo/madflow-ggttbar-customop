#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "matrix.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

#include <math.h>
#include <thrust/complex.h>

using namespace tensorflow;

using GPUDevice = Eigen::GpuDevice;

#define COMPLEX_TYPE thrust::complex<double>
#define COMPLEX_CONJUGATE thrust::conj

#define DEFAULT_BLOCK_SIZE 256//1024

__device__ double SQH;// = 0.70710676908493; // tf.math.sqrt(0.5) == 0.70710676908493;
__device__ COMPLEX_TYPE CZERO;// = COMPLEX_TYPE(0.0, 0.0);
__device__ void vxxxxx(const double* p, double fmass, double nhel, double nsf, COMPLEX_TYPE*);
__device__ void ixxxxx(const double* p, double fmass, double nhel, double nsf, COMPLEX_TYPE*);
__device__ void oxxxxx(const double* p, double fmass, double nhel, double nsf, COMPLEX_TYPE*);
__device__ void _ix_massive(const double* p, double fmass, double nsf, double nh, COMPLEX_TYPE* v);
__device__ void _ix_massless(const double* p, double nhel, double nsf, double nh, COMPLEX_TYPE* v);
__device__ void _ox_massless(const double* p, double nhel, double nsf, double nh, COMPLEX_TYPE* v);
__device__ void _ox_massive(const double* p, double fmass, double nhel, double nsf, double nh, COMPLEX_TYPE* v);
__device__ void _ix_massless_sqp0p3_zero(const double* p, double nhel, COMPLEX_TYPE& val);
__device__ void _ix_massless_sqp0p3_nonzero(const double* p, double nh, double sqp0p3, COMPLEX_TYPE& val);
__device__ void _ix_massive_pp_nonzero(const double* p, double fmass, double nsf, double nh, int ip, int im, double pp, COMPLEX_TYPE* v);
__device__ void _ix_massless_nh_one(COMPLEX_TYPE* chi, COMPLEX_TYPE* v);
__device__ void _ix_massless_nh_not_one(COMPLEX_TYPE* chi, COMPLEX_TYPE* v);
__device__ void _ox_massive_pp_zero(double fmass, double nsf, int ip, int im, COMPLEX_TYPE* v);
__device__ void _ox_massive_pp_nonzero(const double* p, double fmass, double nsf, double nh, double pp, COMPLEX_TYPE* v);
__device__ void _vx_BRST_check(const double* p, double vmass, COMPLEX_TYPE* v);
__device__ void _vx_no_BRST_check(const double *p, double vmass, double nhel, double nsv, double hel0, double nsvahl, double pp, double pt, COMPLEX_TYPE* v);
__device__ void _vx_BRST_check_massless(const double* p, COMPLEX_TYPE* v);
__device__ void _vx_BRST_check_massive(const double* p, double vmass, COMPLEX_TYPE* v);
__device__ void _vx_no_BRST_check_massive(const double* p, double vmass, double nhel, double hel0, double nsvahl, double pp, double pt, COMPLEX_TYPE* v);
__device__ void _vx_no_BRST_check_massive_pp_zero(double nhel, double nsvahl, COMPLEX_TYPE* v);
__device__ void _vx_no_BRST_check_massive_pp_nonzero(const double* p, double vmass, double nhel, double hel0, double nsvahl, double pp, double pt, COMPLEX_TYPE* v);
__device__ void _vx_no_BRST_check_massive_pp_nonzero_pt_nonzero(const double* p, double nhel, double hel0, double nsvahl, double pp, double pt, double emp, COMPLEX_TYPE* v);
__device__ void _vx_no_BRST_check_massive_pp_nonzero_pt_zero(const double* p, double nhel, double nsvahl, COMPLEX_TYPE* v);
__device__ void _vx_no_BRST_check_massless(const double* p, double nhel, double nsv, COMPLEX_TYPE* v);
__device__ void _vx_no_BRST_check_massless_pt_nonzero(const double* p, double nhel, double nsv, double pp, double pt, COMPLEX_TYPE* v);
__device__ void _vx_no_BRST_check_massless_pt_zero(const double* p, double nhel, double nsv, COMPLEX_TYPE* v);

__device__ void VVV1P0_1(COMPLEX_TYPE* V2, COMPLEX_TYPE* V3, const COMPLEX_TYPE COUP, double M1, double W1, COMPLEX_TYPE*);
__device__ void FFV1_0(COMPLEX_TYPE* F1, COMPLEX_TYPE* F2, COMPLEX_TYPE* V3, const COMPLEX_TYPE COUP, COMPLEX_TYPE& amp);
__device__ void FFV1_1(COMPLEX_TYPE* F2, COMPLEX_TYPE* V3, const COMPLEX_TYPE COUP, double M1, double W1, COMPLEX_TYPE*);
__device__ void FFV1_2(COMPLEX_TYPE* F1, COMPLEX_TYPE* V3, const COMPLEX_TYPE COUP, double M1, double W1, COMPLEX_TYPE*);

__device__ double signn(double x, double y);
__device__ double signvecc(double x, double y);
 

// Define the CUDA kernel.
__global__ void MatrixCudaKernel(const double* all_ps, const double* hel, const double* mdl_MT, const double* mdl_WT, const complex128* GC_10, const complex128* GC_11, 
            double* output_flat, const int nevents) {
    
    __shared__ COMPLEX_TYPE denom[2];
    denom[0] = COMPLEX_TYPE(3, 0);
    denom[1] = COMPLEX_TYPE(3, 0);
    
    __shared__ COMPLEX_TYPE cf[4];
    cf[0] = 16;
    cf[1] = -2;
    cf[2] = -2;
    cf[3] = 16;
    
    __shared__ double ZERO;
    ZERO = 0.;
    
        
    // Begin code
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nevents; i += blockDim.x * gridDim.x) {
        COMPLEX_TYPE w0[6], w1[6], w2[6], w3[6], w4[6];
        vxxxxx(all_ps+(16*i), ZERO, hel[0], -1, w0);
        vxxxxx(all_ps+(16*i+4), ZERO, hel[1], -1, w1);
        oxxxxx(all_ps+(16*i+8), mdl_MT[0], hel[2], +1, w2);
        ixxxxx(all_ps+(16*i+12), mdl_MT[0], hel[3], -1, w3);
        VVV1P0_1(w0, w1, GC_10[i], ZERO, ZERO, w4);
        
        // Amplitude(s) for diagram number 1
        
        COMPLEX_TYPE amp0;
        FFV1_0(w3, w2, w4, GC_11[i], amp0);
        FFV1_1(w2, w0, GC_11[i], mdl_MT[0], mdl_WT[0], w4);
        
        // Amplitude(s) for diagram number 2
        
        COMPLEX_TYPE amp1;
        FFV1_0(w3, w4, w1, GC_11[i], amp1);
        FFV1_2(w3, w0, GC_11[i], mdl_MT[0], mdl_WT[0], w4);
        
        // Amplitude(s) for diagram number 3
        
        COMPLEX_TYPE amp2;
        FFV1_0(w4, w2, w1, GC_11[i], amp2);
        
        COMPLEX_TYPE jamp[2] = {COMPLEX_TYPE(0, 1) * amp0 - amp1, -COMPLEX_TYPE(0, 1) * amp0 - amp2};
        
        COMPLEX_TYPE ret(0, 0);
        for (int a = 0; a < 2; a++) {
            for (int b = 0; b < 2; b++) {
                // ret = tf.einsum("ae, ab, be -> e", jamp, cf, tf.math.conj(jamp)/tf.reshape(denom, (ncolor, 1)))
                ret += (jamp[a] * cf[a * 2 + b] * COMPLEX_CONJUGATE(jamp[b])) / denom[b];
            }
        }
        output_flat[i] = ret.real();
    }
}

void MatrixFunctor<GPUDevice>::operator()(
    const GPUDevice& d, const double* all_ps, const double* hel, const double* mdl_MT, const double* mdl_WT, const complex128* GC_10, const complex128* GC_11, 
            double* output_flat, const int nevents) {
    // Launch the cuda kernel.
    //
    // See core/util/gpu_kernel_helper.h for example of computing
    // block count and thread_per_block count.
  
    int blockSize = DEFAULT_BLOCK_SIZE;
    int numBlocks = (nevents + blockSize - 1) / blockSize;
    
    //std::cout << blockSize << " " << numBlocks << std::endl;
    if (nevents < blockSize) {
      numBlocks = 1;
      blockSize = nevents;
    }
    
    //int ngraphs = 3;
    //int nwavefuncs = 5;
    //int ncolor = 2;
    //std::cout << "<<< " << numBlocks << ", " << blockSize << " >>>" << std::endl;
    
    MatrixCudaKernel<<<numBlocks, blockSize, 0, d.stream()>>>(all_ps, hel, mdl_MT, mdl_WT, GC_10, GC_11, output_flat, nevents);
}

// Explicitly instantiate functors for the types of OpKernels registered.
template struct MatrixFunctor<GPUDevice>;


__device__ void vxxxxx(const double* p, double vmass, double nhel, double nsv, COMPLEX_TYPE* ret) {
    COMPLEX_TYPE v0 = COMPLEX_TYPE(p[0] * nsv, p[3] * nsv);
    COMPLEX_TYPE v1 = COMPLEX_TYPE(p[1] * nsv, p[2] * nsv);
    
    double pt2 = p[1] * p[1] + p[2] * p[2];
    double pp = min(p[0], sqrt(pt2 + p[3] * p[3]));
    double pt = min(pp, sqrt(pt2));
    
    double hel0 = 1 - abs(nhel);
    double nsvahl = nsv * abs(nhel);
    
    COMPLEX_TYPE v[4];
    
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

__device__ void ixxxxx(const double* p, double fmass, double nhel, double nsf, COMPLEX_TYPE* ret) {
    COMPLEX_TYPE v0 = COMPLEX_TYPE(-p[0] * nsf, -p[3] * nsf);
    COMPLEX_TYPE v1 = COMPLEX_TYPE(-p[1] * nsf, -p[2] * nsf);
    
    double nh = nhel * nsf;
    
    COMPLEX_TYPE v[4];
    
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

__device__ void oxxxxx(const double* p, double fmass, double nhel, double nsf, COMPLEX_TYPE* ret) {
    COMPLEX_TYPE v0 = COMPLEX_TYPE(p[0] * nsf, p[3] * nsf);
    COMPLEX_TYPE v1 = COMPLEX_TYPE(p[1] * nsf, p[2] * nsf);
    
    double nh = nhel * nsf;
    
    COMPLEX_TYPE v[4];
    
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

// _vx_*

__device__ void _vx_BRST_check(const double* p, double vmass, COMPLEX_TYPE* v) {
    if (vmass == 0) {
        _vx_BRST_check_massless(p, v);
    }
    else {
        _vx_BRST_check_massive(p, vmass, v);
    }
}

__device__ void _vx_no_BRST_check(const double* p, double vmass, double nhel, double nsv, double hel0, double nsvahl, double pp, double pt, COMPLEX_TYPE* v) {
    if (vmass != 0) {
        _vx_no_BRST_check_massive(
                            p, vmass, nhel, hel0, nsvahl, pp, pt, v
                                                );
    }
    else {
        _vx_no_BRST_check_massless(p, nhel, nsv, v);
    }
}

__device__ void _vx_BRST_check_massless(const double* p, COMPLEX_TYPE* v) {
    for (int i = 0; i < 4; i++) {
        v[i] = p[i]/p[0];
    }
}

__device__ void _vx_BRST_check_massive(const double* p, double vmass, COMPLEX_TYPE* v) {
    for (int i = 0; i < 4; i++) {
        v[i] = p[i]/vmass;
    }
}

__device__ void _vx_no_BRST_check_massive(const double* p, double vmass, double nhel, double hel0, double nsvahl, double pp, double pt, COMPLEX_TYPE* v) {
    if (pp == 0) {
        _vx_no_BRST_check_massive_pp_zero(nhel, nsvahl, v);
    }
    else {
        _vx_no_BRST_check_massive_pp_nonzero(
                        p, vmass, nhel, hel0, nsvahl, pp, pt, v
                                                    );
    }
}

__device__ void _vx_no_BRST_check_massive_pp_zero(double nhel, double nsvahl, COMPLEX_TYPE* v) {
    double hel0 = 1.0 - abs(nhel);
    SQH = sqrt(0.5); // !!!!
    v[0] = COMPLEX_TYPE(1, 0);
    v[1] = COMPLEX_TYPE(-nhel * SQH, 0.0);
    v[2] = COMPLEX_TYPE(0.0, nsvahl * SQH);
    v[3] = COMPLEX_TYPE(hel0, 0.0);
}

__device__ void _vx_no_BRST_check_massive_pp_nonzero(const double* p, double vmass, double nhel, double hel0, double nsvahl, double pp, double pt, COMPLEX_TYPE* v) {
    double emp = p[0] / (vmass * pp);
    SQH = sqrt(0.5); // !!!!
    COMPLEX_TYPE v2 = COMPLEX_TYPE(hel0 * pp / vmass, 0.0);
    COMPLEX_TYPE v5 = COMPLEX_TYPE(hel0 * p[3] * emp + (nhel * pt) / (pp * SQH), 0.0);
    
    COMPLEX_TYPE v34[2];
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

__device__ void _vx_no_BRST_check_massive_pp_nonzero_pt_nonzero(const double* p, double nhel, double hel0, double nsvahl, double pp, double pt, double emp, COMPLEX_TYPE* v) {
    SQH = sqrt(0.5); // !!!!
    double pzpt = p[3] / (pp * pt) * SQH * nhel;
    v[0] = COMPLEX_TYPE(hel0 * p[1] * emp - p[1] * pzpt, -nsvahl * p[2] / pt * SQH);
    v[1] = COMPLEX_TYPE(hel0 * p[2] * emp - p[2] * pzpt, nsvahl * p[1] / pt * SQH);
}

__device__ void _vx_no_BRST_check_massive_pp_nonzero_pt_zero(const double* p, double nhel, double nsvahl, COMPLEX_TYPE* v) {
    SQH = sqrt(0.5); // !!!!
    v[0] = COMPLEX_TYPE(-nhel * SQH, 0);
    v[1] = COMPLEX_TYPE(0.0, nsvahl * signvecc(SQH, p[3]));
}

__device__ void _vx_no_BRST_check_massless(const double* p, double nhel, double nsv, COMPLEX_TYPE* v) {
    double pp = p[0];
    double pt = sqrt(p[1] * p[1] + p[2] * p[2]);
    SQH = sqrt(0.5); // !!!!
    
    COMPLEX_TYPE v2 = COMPLEX_TYPE(0, 0);
    COMPLEX_TYPE v5 = COMPLEX_TYPE(nhel * pt / pp * SQH, 0);
    
    COMPLEX_TYPE v34[2];
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

__device__ void _vx_no_BRST_check_massless_pt_nonzero(const double* p, double nhel, double nsv, double pp, double pt, COMPLEX_TYPE* v) {
    SQH = sqrt(0.5); // !!!!
    double pzpt = p[3] / (pp * pt) * SQH * nhel;
    
    v[0] = COMPLEX_TYPE(-p[1] * pzpt, -nsv * p[2] / pt * SQH);
    v[1] = COMPLEX_TYPE(-p[2] * pzpt, nsv * p[1] / pt * SQH);
}

__device__ void _vx_no_BRST_check_massless_pt_zero(const double* p, double nhel, double nsv, COMPLEX_TYPE* v) {
    SQH = sqrt(0.5); // !!!!
    v[0] = COMPLEX_TYPE(-nhel * SQH, 0);
    v[1] = COMPLEX_TYPE(0, nsv * signvecc(SQH, p[3]));
}

// _ix* / _ox*


__device__ void _ix_massive(const double* p, double fmass, double nsf, double nh, COMPLEX_TYPE* v) {
    double pp = min(p[0], sqrt(p[1] * p[1] + p[2] * p[2] + p[3] * p[3]));
    
    int ip = (int)(1 + nh) / 2;
    int im = (int)(1 - nh) / 2;
    
    if (pp == 0) {
        _ox_massive_pp_zero(fmass, nsf, im, ip, v);
    }
    else {
        _ix_massive_pp_nonzero(p, fmass, nsf, nh, ip, im, pp, v);
    }
}

__device__ void _ix_massless(const double* p, double nhel, double nsf, double nh, COMPLEX_TYPE* v) {
    double sqp0p3 = sqrt(max(p[0] + p[3], 0.0)) * nsf;
    
    COMPLEX_TYPE chi1;
    if (sqp0p3 == 0) {
        _ix_massless_sqp0p3_zero(p, nhel, chi1);
    }
    else {
        _ix_massless_sqp0p3_nonzero(p, nh, sqp0p3, chi1);
    }
    
    COMPLEX_TYPE chi[] = {COMPLEX_TYPE(sqp0p3, 0.0), chi1};
    
    if (nh == 1) {
        _ix_massless_nh_one(chi, v);
    }
    else {
        _ix_massless_nh_not_one(chi, v);
    }
}

__device__ void _ox_massless(const double* p, double nhel, double nsf, double nh, COMPLEX_TYPE* v) {
    double sqp0p3 = sqrt(max(p[0] + p[3], 0.0)) * nsf;
    double mult[] = {1, 1, -1, 1};
    
    COMPLEX_TYPE chi0;
    if (sqp0p3 == 0) {
        _ix_massless_sqp0p3_zero(p, nhel, chi0);
    }
    else {
        double prod[4];
        for (int i = 0; i < 4; i++)
            prod[i] = p[i] * mult[i];
        _ix_massless_sqp0p3_nonzero(prod, nh, sqp0p3, chi0);
    }
    
    COMPLEX_TYPE chi[2];
    chi[0] = chi0;
    chi[1] = COMPLEX_TYPE(sqp0p3, 0.0);
    
    if (nh == 1) {
        _ix_massless_nh_not_one(chi, v);
    }
    else {
        _ix_massless_nh_one(chi, v);
    }
}

__device__ void _ox_massive(const double* p, double fmass, double nhel, double nsf, double nh, COMPLEX_TYPE* v) {
    double pp = min(p[0], sqrt(p[1] * p[1] + p[2] * p[2] + p[3] * p[3]));
    
    int ip = -((int)(1 - nh) / 2) * (int)nhel;
    int im =  ((int)(1 + nh) / 2) * (int)nhel;
    
    if (pp == 0) {
        _ox_massive_pp_zero(fmass, nsf, ip, im, v);
    }
    else {
        _ox_massive_pp_nonzero(p, fmass, nsf, nh, pp, v);
    }
}

__device__ void _ox_massive_pp_zero(double fmass, double nsf, int ip, int im, COMPLEX_TYPE* v) {
    double sqm[2];
    sqm[0] = sqrt(fmass);
    sqm[1] = signn(sqm[0], fmass);
    
    v[0] = COMPLEX_TYPE((double)im * sqm[abs(im)], 0.0);
    v[1] = COMPLEX_TYPE((double)ip * nsf * sqm[abs(im)], 0.0);
    v[2] = COMPLEX_TYPE((double)im * nsf * sqm[abs(ip)], 0.0);
    v[3] = COMPLEX_TYPE((double)ip * sqm[abs(ip)], 0.0);
}

__device__ void _ox_massive_pp_nonzero(const double* p, double fmass, double nsf, double nh, double pp, COMPLEX_TYPE* v) {
    double sf[] = {(1 + nsf + (1 - nsf) * nh) * 0.5, (1 + nsf - (1 - nsf) * nh) * 0.5};
    double omega[] = {sqrt(p[0] + pp), fmass / (sqrt(p[0] + pp))};
    
    int ip = (int) (1 + nh) / 2;
    int im = (int) (1 - nh) / 2;
    
    double sfomeg[] = {sf[0] * omega[ip], sf[1] * omega[im]};
    
    double pp3 = max(pp + p[3], 0.0);
    COMPLEX_TYPE chi1;
    if (pp3 == 0) {
        chi1 = COMPLEX_TYPE(-nh, 0);
    }
    else {
        chi1 = COMPLEX_TYPE(nh * p[1] / sqrt(2.0 * pp * pp3), -p[2] / sqrt(2.0 * pp * pp3));
    }
    COMPLEX_TYPE chi2(sqrt(pp3 * 0.5 / pp), 0.0);
    COMPLEX_TYPE chi[] = {chi2, chi1};
    
    v[0] = COMPLEX_TYPE(sfomeg[1], 0.0) * chi[im];
    v[1] = COMPLEX_TYPE(sfomeg[1], 0.0) * chi[ip];
    v[2] = COMPLEX_TYPE(sfomeg[0], 0.0) * chi[im];
    v[3] = COMPLEX_TYPE(sfomeg[0], 0.0) * chi[ip];
}

__device__ void _ix_massless_sqp0p3_zero(const double* p, double nhel, COMPLEX_TYPE& val) {
    val = COMPLEX_TYPE(-nhel * sqrt(2.0 * p[0]), 0.0);
}

__device__ void _ix_massless_sqp0p3_nonzero(const double* p, double nh, double sqp0p3, COMPLEX_TYPE& val) {
    val = COMPLEX_TYPE(nh * p[1] / sqp0p3, p[2] / sqp0p3);
}

__device__ void _ix_massive_pp_nonzero(const double* p, double fmass, double nsf, double nh, int ip, int im, double pp, COMPLEX_TYPE* v) {
    double sf[] = {(1 + nsf + (1 - nsf) * nh) * 0.5, (1 + nsf - (1 - nsf) * nh) * 0.5};
    double omega[] = {sqrt(p[0] + pp), fmass / (sqrt(p[0] + pp))};
    
    double sfomeg[] = {sf[0] * omega[ip], sf[1] * omega[im]};
    
    double pp3 = max(pp + p[3], 0.0);
    COMPLEX_TYPE chi1;
    if (pp3 == 0) {
        chi1 = COMPLEX_TYPE(-nh, 0);
    }
    else {
        chi1 = COMPLEX_TYPE(nh * p[1] / sqrt(2.0 * pp * pp3), p[2] / sqrt(2.0 * pp * pp3));
    }
    COMPLEX_TYPE chi2(sqrt(pp3 * 0.5 / pp), 0.0);
    COMPLEX_TYPE chi[] = {chi2, chi1};
    
    v[0] = COMPLEX_TYPE(sfomeg[0], 0.0) * chi[im];
    v[1] = COMPLEX_TYPE(sfomeg[0], 0.0) * chi[ip];
    v[2] = COMPLEX_TYPE(sfomeg[1], 0.0) * chi[im];
    v[3] = COMPLEX_TYPE(sfomeg[1], 0.0) * chi[ip];
}

__device__ void _ix_massless_nh_one(COMPLEX_TYPE* chi, COMPLEX_TYPE* v) {
    CZERO = COMPLEX_TYPE(0.0, 0.0); // !!!!
    v[2] = chi[0];
    v[3] = chi[1];
    v[0] = CZERO;
    v[1] = CZERO;
}

__device__ void _ix_massless_nh_not_one(COMPLEX_TYPE* chi, COMPLEX_TYPE* v) {
    CZERO = COMPLEX_TYPE(0.0, 0.0); // !!!!
    v[0] = chi[1];
    v[1] = chi[0];
    v[2] = CZERO;
    v[3] = CZERO;
}

// sign

__device__ double signn(double x, double y) {
    int sign = 0;
    y >= 0 ? sign = 1 : sign = -1;
    return x * sign;
}

__device__ double signvecc(double x, double y) {
    return signn(x, y);
}

// V*

__device__ void VVV1P0_1(COMPLEX_TYPE* V2, COMPLEX_TYPE* V3, const COMPLEX_TYPE COUP, double M1_double, double W1_double, COMPLEX_TYPE* V1) {
    
    // V2 -> 6-component vector
    // V3 -> 6-component vector
    
    COMPLEX_TYPE cI(0, 1);
    COMPLEX_TYPE M1 = M1_double;
    COMPLEX_TYPE W1 = W1_double;
    //COMPLEX_TYPE COUP = COUP_comp;
    
    COMPLEX_TYPE P2[4];
    P2[0] = COMPLEX_TYPE(V2[0].real(), 0.0);
    P2[1] = COMPLEX_TYPE(V2[1].real(), 0.0);
    P2[2] = COMPLEX_TYPE(V2[1].imag(), 0.0);
    P2[3] = COMPLEX_TYPE(V2[0].imag(), 0.0);
    
    COMPLEX_TYPE P3[4];
    P3[0] = COMPLEX_TYPE(V3[0].real(), 0.0);
    P3[1] = COMPLEX_TYPE(V3[1].real(), 0.0);
    P3[2] = COMPLEX_TYPE(V3[1].imag(), 0.0);
    P3[3] = COMPLEX_TYPE(V3[0].imag(), 0.0);
    
    V1[0] = V2[0] + V3[0];
    V1[1] = V2[1] + V3[1];
    
    COMPLEX_TYPE P1[4];
    P1[0] = COMPLEX_TYPE(-V1[0].real(), 0.0);
    P1[1] = COMPLEX_TYPE(-V1[1].real(), 0.0);
    P1[2] = COMPLEX_TYPE(-V1[1].imag(), 0.0);
    P1[3] = COMPLEX_TYPE(-V1[0].imag(), 0.0);
    
    COMPLEX_TYPE TMP0 = (V3[2]*P1[0] - V3[3]*P1[1] - V3[4]*P1[2] - V3[5]*P1[3]);
    COMPLEX_TYPE TMP1 = (V3[2]*P2[0] - V3[3]*P2[1] - V3[4]*P2[2] - V3[5]*P2[3]);
    COMPLEX_TYPE TMP2 = (P1[0]*V2[2] - P1[1]*V2[3] - P1[2]*V2[4] - P1[3]*V2[5]);
    COMPLEX_TYPE TMP3 = (V2[2]*P3[0] - V2[3]*P3[1] - V2[4]*P3[2] - V2[5]*P3[3]);
    COMPLEX_TYPE TMP4 = (V3[2]*V2[2] - V3[3]*V2[3] - V3[4]*V2[4] - V3[5]*V2[5]);
    
    COMPLEX_TYPE denom = COUP/(P1[0]*P1[0] - P1[1]*P1[1] - P1[2]*P1[2] - P1[3]*P1[3] - M1 * (M1 -cI* W1));
    
    V1[2]= denom * (TMP4 * (-cI*(P2[0]) + cI*(P3[0])) + (V2[2]*(-cI*(TMP0) + cI*(TMP1)) + V3[2]*(cI*(TMP2) - cI*(TMP3))));
    V1[3]= denom * (TMP4 * (-cI*(P2[1]) + cI*(P3[1])) + (V2[3]*(-cI*(TMP0) + cI*(TMP1)) + V3[3]*(cI*(TMP2) - cI*(TMP3))));
    V1[4]= denom * (TMP4 * (-cI*(P2[2]) + cI*(P3[2])) + (V2[4]*(-cI*(TMP0) + cI*(TMP1)) + V3[4]*(cI*(TMP2) - cI*(TMP3))));
    V1[5]= denom * (TMP4 * (-cI*(P2[3]) + cI*(P3[3])) + (V2[5]*(-cI*(TMP0) + cI*(TMP1)) + V3[5]*(cI*(TMP2) - cI*(TMP3))));
}

__device__ void FFV1_0(COMPLEX_TYPE* F1, COMPLEX_TYPE* F2, COMPLEX_TYPE* V3, const COMPLEX_TYPE COUP, COMPLEX_TYPE& amp) {
    COMPLEX_TYPE cI(0, 1);
    
    COMPLEX_TYPE TMP5 = (F1[2] * (F2[4] * (V3[2]+V3[5]) + F2[5] * (V3[3] + cI * (V3[4]))) + 
                                 (F1[3] * (F2[4] * (V3[3] - cI * (V3[4])) + F2[5] * (V3[2] - V3[5])) + 
                                 (F1[4] * (F2[2] * (V3[2] - V3[5]) - F2[3] * (V3[3] + cI * (V3[4]))) + 
                                  F1[5] * (F2[2] * (-V3[3] + cI * (V3[4])) + F2[3] * (V3[2] + V3[5])))));
    amp = COUP * -cI * TMP5;
}

__device__ void FFV1_1(COMPLEX_TYPE* F2, COMPLEX_TYPE* V3, const COMPLEX_TYPE COUP, double M1_double, double W1_double, COMPLEX_TYPE* F1) {
    COMPLEX_TYPE cI(0, 1);
    COMPLEX_TYPE M1 = M1_double;
    COMPLEX_TYPE W1 = W1_double;
    //COMPLEX_TYPE COUP = COUP_comp;
    
    F1[0] = F2[0] + V3[0];
    F1[1] = F2[1] + V3[1];
    
    COMPLEX_TYPE P1[4];
    P1[0] = COMPLEX_TYPE(-F1[0].real(), 0.0);
    P1[1] = COMPLEX_TYPE(-F1[1].real(), 0.0);
    P1[2] = COMPLEX_TYPE(-F1[1].imag(), 0.0);
    P1[3] = COMPLEX_TYPE(-F1[0].imag(), 0.0);
    
    COMPLEX_TYPE denom = COUP/(P1[0]*P1[0] - P1[1]*P1[1] - P1[2]*P1[2] - P1[3]*P1[3] - M1 * (M1 -cI* W1));
    
    F1[2]= denom*cI*(F2[2]*(P1[0]*(-V3[2]+V3[5])+(P1[1]*(V3[3]-cI*(V3[4]))+(P1[2]*(cI*(V3[3])+V3[4])+P1[3]*(-V3[2]+V3[5]))))+(F2[3]*(P1[0]*(V3[3]+cI*(V3[4]))+(P1[1]*(-1./1.)*(V3[2]+V3[5])+(P1[2]*(-1./1.)*(cI*(V3[2]+V3[5]))+P1[3]*(V3[3]+cI*(V3[4])))))+M1*(F2[4]*(V3[2]+V3[5])+F2[5]*(V3[3]+cI*(V3[4])))));
    F1[3]= denom*(-cI)*(F2[2]*(P1[0]*(-V3[3]+cI*(V3[4]))+(P1[1]*(V3[2]-V3[5])+(P1[2]*(-cI*(V3[2])+cI*(V3[5]))+P1[3]*(V3[3]-cI*(V3[4])))))+(F2[3]*(P1[0]*(V3[2]+V3[5])+(P1[1]*(-1./1.)*(V3[3]+cI*(V3[4]))+(P1[2]*(cI*(V3[3])-V3[4])-P1[3]*(V3[2]+V3[5]))))+M1*(F2[4]*(-V3[3]+cI*(V3[4]))+F2[5]*(-V3[2]+V3[5]))));
    F1[4]= denom*(-cI)*(F2[4]*(P1[0]*(V3[2]+V3[5])+(P1[1]*(-V3[3]+cI*(V3[4]))+(P1[2]*(-1./1.)*(cI*(V3[3])+V3[4])-P1[3]*(V3[2]+V3[5]))))+(F2[5]*(P1[0]*(V3[3]+cI*(V3[4]))+(P1[1]*(-V3[2]+V3[5])+(P1[2]*(-cI*(V3[2])+cI*(V3[5]))-P1[3]*(V3[3]+cI*(V3[4])))))+M1*(F2[2]*(-V3[2]+V3[5])+F2[3]*(V3[3]+cI*(V3[4])))));
    F1[5]= denom*cI*(F2[4]*(P1[0]*(-V3[3]+cI*(V3[4]))+(P1[1]*(V3[2]+V3[5])+(P1[2]*(-1./1.)*(cI*(V3[2]+V3[5]))+P1[3]*(-V3[3]+cI*(V3[4])))))+(F2[5]*(P1[0]*(-V3[2]+V3[5])+(P1[1]*(V3[3]+cI*(V3[4]))+(P1[2]*(-cI*(V3[3])+V3[4])+P1[3]*(-V3[2]+V3[5]))))+M1*(F2[2]*(-V3[3]+cI*(V3[4]))+F2[3]*(V3[2]+V3[5]))));
    
    //return F1;
}

__device__ void FFV1_2(COMPLEX_TYPE* F1, COMPLEX_TYPE* V3, const COMPLEX_TYPE COUP, double M2_double, double W2_double, COMPLEX_TYPE* F2) {
    COMPLEX_TYPE cI(0, 1);
    COMPLEX_TYPE M2 = M2_double;
    COMPLEX_TYPE W2 = W2_double;
    //COMPLEX_TYPE COUP = COUP_comp;
    
    F2[0] = F1[0] + V3[0];
    F2[1] = F1[1] + V3[1];
    
    COMPLEX_TYPE P2[4];
    P2[0] = COMPLEX_TYPE(-F2[0].real(), 0.0);
    P2[1] = COMPLEX_TYPE(-F2[1].real(), 0.0);
    P2[2] = COMPLEX_TYPE(-F2[1].imag(), 0.0);
    P2[3] = COMPLEX_TYPE(-F2[0].imag(), 0.0);
    
    COMPLEX_TYPE denom = COUP/(P2[0]*P2[0] - P2[1]*P2[1] - P2[2]*P2[2] - P2[3]*P2[3] - M2 * (M2 -cI* W2));
    
    F2[2]= denom*cI*(F1[2]*(P2[0]*(V3[2]+V3[5])+(P2[1]*(-1./1.)*(V3[3]+cI*(V3[4]))+(P2[2]*(cI*(V3[3])-V3[4])-P2[3]*(V3[2]+V3[5]))))+(F1[3]*(P2[0]*(V3[3]-cI*(V3[4]))+(P2[1]*(-V3[2]+V3[5])+(P2[2]*(cI*(V3[2])-cI*(V3[5]))+P2[3]*(-V3[3]+cI*(V3[4])))))+M2*(F1[4]*(V3[2]-V3[5])+F1[5]*(-V3[3]+cI*(V3[4])))));
    F2[3]= denom*(-cI)*(F1[2]*(P2[0]*(-1./1.)*(V3[3]+cI*(V3[4]))+(P2[1]*(V3[2]+V3[5])+(P2[2]*(cI*(V3[2]+V3[5]))-P2[3]*(V3[3]+cI*(V3[4])))))+(F1[3]*(P2[0]*(-V3[2]+V3[5])+(P2[1]*(V3[3]-cI*(V3[4]))+(P2[2]*(cI*(V3[3])+V3[4])+P2[3]*(-V3[2]+V3[5]))))+M2*(F1[4]*(V3[3]+cI*(V3[4]))-F1[5]*(V3[2]+V3[5]))));
    F2[4]= denom*(-cI)*(F1[4]*(P2[0]*(-V3[2]+V3[5])+(P2[1]*(V3[3]+cI*(V3[4]))+(P2[2]*(-cI*(V3[3])+V3[4])+P2[3]*(-V3[2]+V3[5]))))+(F1[5]*(P2[0]*(V3[3]-cI*(V3[4]))+(P2[1]*(-1./1.)*(V3[2]+V3[5])+(P2[2]*(cI*(V3[2]+V3[5]))+P2[3]*(V3[3]-cI*(V3[4])))))+M2*(F1[2]*(-1./1.)*(V3[2]+V3[5])+F1[3]*(-V3[3]+cI*(V3[4])))));
    F2[5]= denom*cI*(F1[4]*(P2[0]*(-1./1.)*(V3[3]+cI*(V3[4]))+(P2[1]*(V3[2]-V3[5])+(P2[2]*(cI*(V3[2])-cI*(V3[5]))+P2[3]*(V3[3]+cI*(V3[4])))))+(F1[5]*(P2[0]*(V3[2]+V3[5])+(P2[1]*(-V3[3]+cI*(V3[4]))+(P2[2]*(-1./1.)*(cI*(V3[3])+V3[4])-P2[3]*(V3[2]+V3[5]))))+M2*(F1[2]*(V3[3]+cI*(V3[4]))+F1[3]*(V3[2]-V3[5]))));
}

#endif  // GOOGLE_CUDA

