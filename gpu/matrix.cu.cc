#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "matrix.h"
//#include "tensorflow/core/util/gpu_kernel_helper.h"
#include <iostream>
#include <math.h>
//#include <thrust/complex.h>

using namespace tensorflow;

using GPUDevice = Eigen::GpuDevice;

//#define COMPLEX_TYPE thrust::complex<double>
#define COMPLEX_CONJUGATE conj//thrust::conj

#define DEFAULT_BLOCK_SIZE 256//1024

__device__ double SQH = 0.70710676908493; // tf.math.sqrt(0.5) == 0.70710676908493;
__device__ COMPLEX_TYPE CZERO;// = COMPLEX_TYPE(0.0, 0.0);
template <typename T>
__device__ void vxxxxx(const double* p, double fmass, double nhel, double nsf, T*);
template <typename T>
__device__ void ixxxxx(const double* p, double fmass, double nhel, double nsf, T*);
template <typename T>
__device__ void oxxxxx(const double* p, double fmass, double nhel, double nsf, T*);
template <typename T>
__device__ void _ix_massive(const double* p, double fmass, double nsf, double nh, T* v);
template <typename T>
__device__ void _ix_massless(const double* p, double nhel, double nsf, double nh, T* v);
template <typename T>
__device__ void _ox_massless(const double* p, double nhel, double nsf, double nh, T* v);
template <typename T>
__device__ void _ox_massive(const double* p, double fmass, double nhel, double nsf, double nh, T* v);
template <typename T>
__device__ void _ix_massless_sqp0p3_zero(const double* p, double nhel, T& val);
template <typename T>
__device__ void _ix_massless_sqp0p3_nonzero(const double* p, double nh, double sqp0p3, T& val);
template <typename T>
__device__ void _ix_massive_pp_nonzero(const double* p, double fmass, double nsf, double nh, int ip, int im, double pp, T* v);
template <typename T>
__device__ void _ix_massless_nh_one(T* chi, T* v);
template <typename T>
__device__ void _ix_massless_nh_not_one(T* chi, T* v);
template <typename T>
__device__ void _ox_massive_pp_zero(double fmass, double nsf, int ip, int im, T* v);
template <typename T>
__device__ void _ox_massive_pp_nonzero(const double* p, double fmass, double nsf, double nh, double pp, T* v);
template <typename T>
__device__ void _vx_BRST_check(const double* p, double vmass, T* v);
template <typename T>
__device__ void _vx_no_BRST_check(const double *p, double vmass, double nhel, double nsv, double hel0, double nsvahl, double pp, double pt, T* v);
template <typename T>
__device__ void _vx_BRST_check_massless(const double* p, T* v);
template <typename T>
__device__ void _vx_BRST_check_massive(const double* p, double vmass, T* v);
template <typename T>
__device__ void _vx_no_BRST_check_massive(const double* p, double vmass, double nhel, double hel0, double nsvahl, double pp, double pt, T* v);
template <typename T>
__device__ void _vx_no_BRST_check_massive_pp_zero(double nhel, double nsvahl, T* v);
template <typename T>
__device__ void _vx_no_BRST_check_massive_pp_nonzero(const double* p, double vmass, double nhel, double hel0, double nsvahl, double pp, double pt, T* v);
template <typename T>
__device__ void _vx_no_BRST_check_massive_pp_nonzero_pt_nonzero(const double* p, double nhel, double hel0, double nsvahl, double pp, double pt, double emp, T* v);
template <typename T>
__device__ void _vx_no_BRST_check_massive_pp_nonzero_pt_zero(const double* p, double nhel, double nsvahl, T* v);
template <typename T>
__device__ void _vx_no_BRST_check_massless(const double* p, double nhel, double nsv, T* v);
template <typename T>
__device__ void _vx_no_BRST_check_massless_pt_nonzero(const double* p, double nhel, double nsv, double pp, double pt, T* v);
template <typename T>
__device__ void _vx_no_BRST_check_massless_pt_zero(const double* p, double nhel, double nsv, T* v);

template <typename T>
__device__ void VVV1P0_1(const T* V2, const T* V3, const T COUP, const double M1, const double W1, T*);
template <typename T>
__device__ void FFV1_0(const T* F1, const T* F2, const T* V3, const T COUP, T* amp);
template <typename T>
__device__ void FFV1_1(const T* F2, const T* V3, const T COUP, double M1, double W1, T*);
template <typename T>
__device__ void FFV1_2(const T* F1, const T* V3, const T COUP, double M1, double W1, T*);

__device__ double signn(double x, double y);
__device__ double signvecc(double x, double y);
/*
__device__ COMPLEX_TYPE csum(COMPLEX_TYPE a, COMPLEX_TYPE b) {
    return COMPLEX_TYPE(a.real() + b.real(), a.imag() + b.imag());
}

__device__ COMPLEX_TYPE cdiff(COMPLEX_TYPE a, COMPLEX_TYPE b) {
    return COMPLEX_TYPE(a.real() - b.real(), a.imag() - b.imag());
}

__device__ COMPLEX_TYPE cmult(COMPLEX_TYPE a, COMPLEX_TYPE b) {
    return COMPLEX_TYPE(a.real() * b.real() - a.imag() * b.imag(), a.imag() * b.real() + a.real() * b.imag());
}

__device__ COMPLEX_TYPE cmult(COMPLEX_TYPE a, double b) {
    return COMPLEX_TYPE(a.real() * b, a.imag() * b);
}

__device__ COMPLEX_TYPE cmult(double a, COMPLEX_TYPE b) {
    return cmult(b, a);
}

__device__ void assign(COMPLEX_TYPE& a, const COMPLEX_TYPE b) {
    a = b;
}

__device__ COMPLEX_TYPE cdiv(COMPLEX_TYPE a, COMPLEX_TYPE b) {
    double norm = b.real() * b.real() + b.imag() * b.imag();
    return COMPLEX_TYPE((a.real() * b.real() + a.imag() * b.imag())/norm, (a.imag() * b.real() - a.real() * b.imag())/norm);
}*/

__device__ COMPLEX_TYPE cconj(COMPLEX_TYPE a) {
    return COMPLEX_TYPE(a.real(), -a.imag());
}

__device__ COMPLEX_TYPE operator+(const COMPLEX_TYPE& a, const COMPLEX_TYPE& b) {
    return COMPLEX_TYPE(a.real() + b.real(), a.imag() + b.imag());
}

__device__ COMPLEX_TYPE operator-(const COMPLEX_TYPE& a, const COMPLEX_TYPE& b) {
    return COMPLEX_TYPE(a.real() - b.real(), a.imag() - b.imag());
}

__device__ COMPLEX_TYPE operator*(const COMPLEX_TYPE& a, const COMPLEX_TYPE& b) {
    return COMPLEX_TYPE(a.real() * b.real() - a.imag() * b.imag(), a.imag() * b.real() + a.real() * b.imag());
}

__device__ COMPLEX_TYPE operator/(const COMPLEX_TYPE& a, const COMPLEX_TYPE& b) {
    double norm = b.real() * b.real() + b.imag() * b.imag();
    return COMPLEX_TYPE((a.real() * b.real() + a.imag() * b.imag())/norm, (a.imag() * b.real() - a.real() * b.imag())/norm);
}

__device__ COMPLEX_TYPE operator-(const COMPLEX_TYPE& a) {
    return COMPLEX_TYPE(-a.real(), -a.imag());
}

__device__ COMPLEX_TYPE operator*(const COMPLEX_TYPE& a, const double& b) {
    return COMPLEX_TYPE(a.real() * b, a.imag() * b);
}

__device__ COMPLEX_TYPE operator*(const double& a, const COMPLEX_TYPE& b) {
    return b * a;
}

__device__ COMPLEX_TYPE operator/(const COMPLEX_TYPE& a, const double& b) {
    return COMPLEX_TYPE(a.real() / b, a.imag() / b);
}
 

// Define the CUDA kernel.
template <typename T>
__global__ void MatrixCudaKernel(const double* all_ps, const double* hel, const double* mdl_MT, const double* mdl_WT, const T* GC_10, const T* GC_11, 
            double* output_flat, const int nevents) {
    
    __shared__ double denom[2];
    denom[0] = 3.0;
    denom[1] = 3.0;
    /*denom[0] = T(3.0, 0.0);
    denom[1] = T(3.0, 0.0);*/
    
    //__shared__ double denom[2] = {3.0, 3.0};
    
    __shared__ double cf[4];
    cf[0] = 16;
    cf[1] = -2;
    cf[2] = -2;
    cf[3] = 16;
    //__shared__ double cf[4] = {16, -2, -2, 16};
    
    __shared__ double ZERO;
    ZERO = 0.;
    
    // Begin code
    //for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nevents; i += blockDim.x * gridDim.x) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    //printf("Thread id: %i\n", i);
    /*if (i == 99999) output_flat[i] = 1;
    else */if (i < nevents) {
        T w0[6], w1[6], w2[6], w3[6], w4[6];
        for (int j = 0; j < 6; j++) w4[j] = T(0,0);
        vxxxxx(all_ps+(16*i), ZERO, hel[0], -1, w0);
        vxxxxx(all_ps+(16*i+4), ZERO, hel[1], -1, w1);
        oxxxxx(all_ps+(16*i+8), mdl_MT[0], hel[2], +1, w2);
        ixxxxx(all_ps+(16*i+12), mdl_MT[0], hel[3], -1, w3);
        VVV1P0_1(w0, w1, GC_10[i], ZERO, ZERO, w4);
        
        // Amplitude(s) for diagram number 1
        
        T amp0(0,0);
        FFV1_0(w3, w2, w4, GC_11[i], &amp0);
        FFV1_1(w2, w0, GC_11[i], mdl_MT[0], mdl_WT[0], w4);
        
        // Amplitude(s) for diagram number 2
        
        T amp1(0,0);
        FFV1_0(w3, w4, w1, GC_11[i], &amp1);
        FFV1_2(w3, w0, GC_11[i], mdl_MT[0], mdl_WT[0], w4);
        
        // Amplitude(s) for diagram number 3
        
        T amp2(0,0);
        FFV1_0(w4, w2, w1, GC_11[i], &amp2);
        
        T jamp[2] = {T(0, 1) * amp0 - amp1, -T(0, 1) * amp0 - amp2};
        //T jamp[2] = {cdiff(cmult(T(0, 1), amp0), amp1), cdiff(cmult(cmult(-1, T(0, 1)), amp0), amp2)};
        
        //T ret(0, 0);
        double ret = 0;
        for (int a = 0; a < 2; a++) {
            for (int b = 0; b < 2; b++) {
                // ret = tf.einsum("ae, ab, be -> e", jamp, cf, tf.math.conj(jamp)/tf.reshape(denom, (ncolor, 1)))
                //ret += ((jamp[a] * cf[a * 2 + b] * COMPLEX_CONJUGATE(jamp[b])) / denom[b]).real();
                //ret += (cmult(cmult(jamp[a], cf[a * 2 + b]), cconj(jamp[b])) / denom[b]).real();
                ret += (jamp[a] * cf[a * 2 + b] * cconj(jamp[b]) / denom[b]).real();
                //ret += cdiv(cmult(cmult(jamp[a], cf[a * 2 + b]), cconj(jamp[b])), denom[b]).real();
            }
        }
        output_flat[i] = ret;//.real();
    }
}

template <typename T>
void MatrixFunctor<GPUDevice, T>::operator()(
    const GPUDevice& d, const double* all_ps, const double* hel, const double* mdl_MT, const double* mdl_WT, const T* GC_10, const T* GC_11, 
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
    
    
    MatrixCudaKernel<T><<<numBlocks, blockSize, 0, d.stream()>>>(all_ps, hel, mdl_MT, mdl_WT, GC_10, GC_11, output_flat, nevents);
    
}

// Explicitly instantiate functors for the types of OpKernels registered.
template struct MatrixFunctor<GPUDevice, COMPLEX_TYPE>;

template <typename T>
__device__ void vxxxxx(const double* p, double vmass, double nhel, double nsv, T* ret) {
    T v0 = T(p[0] * nsv, p[3] * nsv);
    T v1 = T(p[1] * nsv, p[2] * nsv);
    
    double pt2 = p[1] * p[1] + p[2] * p[2];
    double pp = min(p[0], sqrt(pt2 + p[3] * p[3]));
    double pt = min(pp, sqrt(pt2));
    
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
__device__ void ixxxxx(const double* p, double fmass, double nhel, double nsf, T* ret) {
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
__device__ void oxxxxx(const double* p, double fmass, double nhel, double nsf, T* ret) {
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

// _vx_*

template <typename T>
__device__ void _vx_BRST_check(const double* p, double vmass, T* v) {
    if (vmass == 0) {
        _vx_BRST_check_massless(p, v);
    }
    else {
        _vx_BRST_check_massive(p, vmass, v);
    }
}

template <typename T>
__device__ void _vx_no_BRST_check(const double* p, double vmass, double nhel, double nsv, double hel0, double nsvahl, double pp, double pt, T* v) {
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
__device__ void _vx_BRST_check_massless(const double* p, T* v) {
    for (int i = 0; i < 4; i++) {
        //v[i] = p[i]/p[0];
        v[i] = T(p[i]/p[0], 0);
        //assign(v[i], p[i]/p[0]);
    }
}

template <typename T>
__device__ void _vx_BRST_check_massive(const double* p, double vmass, T* v) {
    for (int i = 0; i < 4; i++) {
        //v[i] = p[i]/vmass;
        v[i] = T(p[i]/vmass, 0);
        //assign(v[i], p[i]/vmass);
    }
}

template <typename T>
__device__ void _vx_no_BRST_check_massive(const double* p, double vmass, double nhel, double hel0, double nsvahl, double pp, double pt, T* v) {
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
__device__ void _vx_no_BRST_check_massive_pp_zero(double nhel, double nsvahl, T* v) {
    double hel0 = 1.0 - abs(nhel);
    //SQH = sqrt(0.5); // !!!!
    v[0] = T(1, 0);
    v[1] = T(-nhel * SQH, 0.0);
    v[2] = T(0.0, nsvahl * SQH);
    v[3] = T(hel0, 0.0);
}

template <typename T>
__device__ void _vx_no_BRST_check_massive_pp_nonzero(const double* p, double vmass, double nhel, double hel0, double nsvahl, double pp, double pt, T* v) {
    double emp = p[0] / (vmass * pp);
    //SQH = sqrt(0.5); // !!!!
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
__device__ void _vx_no_BRST_check_massive_pp_nonzero_pt_nonzero(const double* p, double nhel, double hel0, double nsvahl, double pp, double pt, double emp, T* v) {
    //SQH = sqrt(0.5); // !!!!
    double pzpt = p[3] / (pp * pt) * SQH * nhel;
    v[0] = T(hel0 * p[1] * emp - p[1] * pzpt, -nsvahl * p[2] / pt * SQH);
    v[1] = T(hel0 * p[2] * emp - p[2] * pzpt, nsvahl * p[1] / pt * SQH);
}

template <typename T>
__device__ void _vx_no_BRST_check_massive_pp_nonzero_pt_zero(const double* p, double nhel, double nsvahl, T* v) {
    //SQH = sqrt(0.5); // !!!!
    v[0] = T(-nhel * SQH, 0);
    v[1] = T(0.0, nsvahl * signvecc(SQH, p[3]));
}

template <typename T>
__device__ void _vx_no_BRST_check_massless(const double* p, double nhel, double nsv, T* v) {
    double pp = p[0];
    double pt = sqrt(p[1] * p[1] + p[2] * p[2]);
    //SQH = sqrt(0.5); // !!!!
    
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
__device__ void _vx_no_BRST_check_massless_pt_nonzero(const double* p, double nhel, double nsv, double pp, double pt, T* v) {
    //SQH = sqrt(0.5); // !!!!
    double pzpt = p[3] / (pp * pt) * SQH * nhel;
    
    v[0] = T(-p[1] * pzpt, -nsv * p[2] / pt * SQH);
    v[1] = T(-p[2] * pzpt, nsv * p[1] / pt * SQH);
}

template <typename T>
__device__ void _vx_no_BRST_check_massless_pt_zero(const double* p, double nhel, double nsv, T* v) {
    //SQH = sqrt(0.5); // !!!!
    v[0] = T(-nhel * SQH, 0);
    v[1] = T(0, nsv * signvecc(SQH, p[3]));
}

// _ix* / _ox*


template <typename T>
__device__ void _ix_massive(const double* p, double fmass, double nsf, double nh, T* v) {
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

template <typename T>
__device__ void _ix_massless(const double* p, double nhel, double nsf, double nh, T* v) {
    double sqp0p3 = sqrt(max(p[0] + p[3], 0.0)) * nsf;
    
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
__device__ void _ox_massless(const double* p, double nhel, double nsf, double nh, T* v) {
    double sqp0p3 = sqrt(max(p[0] + p[3], 0.0)) * nsf;
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
__device__ void _ox_massive(const double* p, double fmass, double nhel, double nsf, double nh, T* v) {
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

template <typename T>
__device__ void _ox_massive_pp_zero(double fmass, double nsf, int ip, int im, T* v) {
    double sqm[2];
    sqm[0] = sqrt(fmass);
    sqm[1] = signn(sqm[0], fmass);
    
    v[0] = T((double)im * sqm[abs(im)], 0.0);
    v[1] = T((double)ip * nsf * sqm[abs(im)], 0.0);
    v[2] = T((double)im * nsf * sqm[abs(ip)], 0.0);
    v[3] = T((double)ip * sqm[abs(ip)], 0.0);
}

template <typename T>
__device__ void _ox_massive_pp_nonzero(const double* p, double fmass, double nsf, double nh, double pp, T* v) {
    double sf[] = {(1 + nsf + (1 - nsf) * nh) * 0.5, (1 + nsf - (1 - nsf) * nh) * 0.5};
    double omega[] = {sqrt(p[0] + pp), fmass / (sqrt(p[0] + pp))};
    
    int ip = (int) (1 + nh) / 2;
    int im = (int) (1 - nh) / 2;
    
    double sfomeg[] = {sf[0] * omega[ip], sf[1] * omega[im]};
    
    double pp3 = max(pp + p[3], 0.0);
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
    v[3] = T(sfomeg[0], 0.0) * chi[ip];/*
    v[0] = cmult(T(sfomeg[1], 0.0), chi[im]);
    v[1] = cmult(T(sfomeg[1], 0.0), chi[ip]);
    v[2] = cmult(T(sfomeg[0], 0.0), chi[im]);
    v[3] = cmult(T(sfomeg[0], 0.0), chi[ip]);*/
}

template <typename T>
__device__ void _ix_massless_sqp0p3_zero(const double* p, double nhel, T& val) {
    val = T(-nhel * sqrt(2.0 * p[0]), 0.0);
}

template <typename T>
__device__ void _ix_massless_sqp0p3_nonzero(const double* p, double nh, double sqp0p3, T& val) {
    val = T(nh * p[1] / sqp0p3, p[2] / sqp0p3);
}

template <typename T>
__device__ void _ix_massive_pp_nonzero(const double* p, double fmass, double nsf, double nh, int ip, int im, double pp, T* v) {
    double sf[] = {(1 + nsf + (1 - nsf) * nh) * 0.5, (1 + nsf - (1 - nsf) * nh) * 0.5};
    double omega[] = {sqrt(p[0] + pp), fmass / (sqrt(p[0] + pp))};
    
    double sfomeg[] = {sf[0] * omega[ip], sf[1] * omega[im]};
    
    double pp3 = max(pp + p[3], 0.0);
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
    v[3] = T(sfomeg[1], 0.0) * chi[ip];/*
    v[0] = cmult(T(sfomeg[0], 0.0), chi[im]);
    v[1] = cmult(T(sfomeg[0], 0.0), chi[ip]);
    v[2] = cmult(T(sfomeg[1], 0.0), chi[im]);
    v[3] = cmult(T(sfomeg[1], 0.0), chi[ip]);*/
}

template <typename T>
__device__ void _ix_massless_nh_one(T* chi, T* v) {
    CZERO = T(0.0, 0.0); // !!!!
    v[2] = chi[0];
    v[3] = chi[1];
    v[0] = CZERO;
    v[1] = CZERO;
}

template <typename T>
__device__ void _ix_massless_nh_not_one(T* chi, T* v) {
    CZERO = T(0.0, 0.0); // !!!!
    v[0] = chi[1];
    v[1] = chi[0];
    v[2] = CZERO;
    v[3] = CZERO;
}

// sign

__device__ double signn(double x, double y) {
    int sign = 0;
    if (y>=0) {
        return x;
    }
    else {
        return -x;
    }
    y >= 0 ? sign = 1 : sign = -1;
    return x * sign;
}

__device__ double signvecc(double x, double y) {
    return signn(x, y);
}

// V*

template <typename T>
__device__ void VVV1P0_1(const T* V2, const T* V3, const T COUP, const double M1_double, const double W1_double, T* V1) {
    
    // V2 -> 6-component vector
    // V3 -> 6-component vector
    
    T cI(0.0, 1.0);
    const T M1 = T(M1_double, 0.0);
    const T W1 = T(W1_double, 0.0);
    //COMPLEX_TYPE COUP = COUP_comp;
    
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
    /*
    T P20 = T(V2[0].real(), 0.0);
    T P21 = T(V2[1].real(), 0.0);
    T P22 = T(V2[1].imag(), 0.0);
    T P23 = T(V2[0].imag(), 0.0);
    
    T P30 = T(V3[0].real(), 0.0);
    T P31 = T(V3[1].real(), 0.0);
    T P32 = T(V3[1].imag(), 0.0);
    T P33 = T(V3[0].imag(), 0.0);
    */
    V1[0] = V2[0] + V3[0];
    V1[1] = V2[1] + V3[1];/*
    V1[0] = csum(V2[0], V3[0]);
    V1[1] = csum(V2[1], V3[1]);*/
    
    T P1[4];
    P1[0] = T(-V1[0].real(), 0.0);
    P1[1] = T(-V1[1].real(), 0.0);
    P1[2] = T(-V1[1].imag(), 0.0);
    P1[3] = T(-V1[0].imag(), 0.0);
    /*
    T P10 = T(-V1[0].real(), 0.0);
    T P11 = T(-V1[1].real(), 0.0);
    T P12 = T(-V1[1].imag(), 0.0);
    T P13 = T(-V1[0].imag(), 0.0);
    */
    T TMP0 = (V3[2]*P1[0] - V3[3]*P1[1] - V3[4]*P1[2] - V3[5]*P1[3]);
    T TMP1 = (V3[2]*P2[0] - V3[3]*P2[1] - V3[4]*P2[2] - V3[5]*P2[3]);
    T TMP2 = (P1[0]*V2[2] - P1[1]*V2[3] - P1[2]*V2[4] - P1[3]*V2[5]);
    T TMP3 = (V2[2]*P3[0] - V2[3]*P3[1] - V2[4]*P3[2] - V2[5]*P3[3]);
    T TMP4 = (V3[2]*V2[2] - V3[3]*V2[3] - V3[4]*V2[4] - V3[5]*V2[5]);
    /*
    T TMP0 = csum(csum(cmult(V3[2], P1[0]), cmult(-1, cmult(V3[3], P1[1]))), cmult(-1, csum(cmult(V3[4], P1[2]), cmult(-1, cmult(V3[5], P1[3])))));
    T TMP1 = csum(csum(cmult(V3[2], P2[0]), cmult(-1, cmult(V3[3], P2[1]))), cmult(-1, csum(cmult(V3[4], P2[2]), cmult(-1, cmult(V3[5], P2[3])))));
    T TMP2 = csum(csum(cmult(P1[0], V2[2]), cmult(-1, cmult(P1[1], V2[3]))), cmult(-1, csum(cmult(P1[2], V2[4]), cmult(-1, cmult(P1[3], V2[5])))));
    T TMP3 = csum(csum(cmult(V2[2], P3[0]), cmult(-1, cmult(V2[3], P3[1]))), cmult(-1, csum(cmult(V2[4], P3[2]), cmult(-1, cmult(V2[5], P3[3])))));
    T TMP4 = csum(csum(cmult(V3[2], V2[2]), cmult(-1, cmult(V3[3], V2[3]))), cmult(-1, csum(cmult(V3[4], V2[4]), cmult(-1, cmult(V3[5], V2[5])))));
    */
    T denom = COUP/(P1[0]*P1[0] - P1[1]*P1[1] - P1[2]*P1[2] - P1[3]*P1[3] - M1 * (M1 -cI* W1));
    //T denom = cdiv(COUP, (cdiff(cdiff(cdiff(cdiff(cmult(P1[0], P1[0]), cmult(P1[1], P1[1])), cmult(P1[2], P1[2])), cmult(P1[3], P1[3])), cmult(M1, cdiff(M1, cmult(cI, W1))))));
    
    
    V1[2]= denom * (TMP4 * (-cI*(P2[0]) + cI*(P3[0])) + (V2[2]*(-cI*(TMP0) + cI*(TMP1)) + V3[2]*(cI*(TMP2) - cI*(TMP3))));
    V1[3]= denom * (TMP4 * (-cI*(P2[1]) + cI*(P3[1])) + (V2[3]*(-cI*(TMP0) + cI*(TMP1)) + V3[3]*(cI*(TMP2) - cI*(TMP3))));
    V1[4]= denom * (TMP4 * (-cI*(P2[2]) + cI*(P3[2])) + (V2[4]*(-cI*(TMP0) + cI*(TMP1)) + V3[4]*(cI*(TMP2) - cI*(TMP3))));
    V1[5]= denom * (TMP4 * (-cI*(P2[3]) + cI*(P3[3])) + (V2[5]*(-cI*(TMP0) + cI*(TMP1)) + V3[5]*(cI*(TMP2) - cI*(TMP3))));/*
    V1[2]= cmult(denom, csum(csum(cmult(TMP4, cdiff(cmult(cI, P3[0]), cmult(cI, P2[0]))), cmult(V2[2], cdiff(cmult(cI, TMP1), cmult(cI, TMP0)))), cmult(V3[2], cdiff(cmult(cI, TMP2), cmult(cI, TMP3)))));
    V1[3]= cmult(denom, csum(csum(cmult(TMP4, cdiff(cmult(cI, P3[1]), cmult(cI, P2[1]))), cmult(V2[3], cdiff(cmult(cI, TMP1), cmult(cI, TMP0)))), cmult(V3[3], cdiff(cmult(cI, TMP2), cmult(cI, TMP3)))));
    V1[4]= cmult(denom, csum(csum(cmult(TMP4, cdiff(cmult(cI, P3[2]), cmult(cI, P2[2]))), cmult(V2[4], cdiff(cmult(cI, TMP1), cmult(cI, TMP0)))), cmult(V3[4], cdiff(cmult(cI, TMP2), cmult(cI, TMP3)))));
    V1[5]= cmult(denom, csum(csum(cmult(TMP4, cdiff(cmult(cI, P3[3]), cmult(cI, P2[3]))), cmult(V2[5], cdiff(cmult(cI, TMP1), cmult(cI, TMP0)))), cmult(V3[5], cdiff(cmult(cI, TMP2), cmult(cI, TMP3)))));/
    /*
    V1[2]= cmult(denom, (cmult(TMP4, csum(cmult(cmult(-1, cI), P2[0]), cmult(cI, P3[0]))) + cmult(V2[2], csum(cmult(cmult(-1, cI), TMP0), cmult(cI, TMP1)) + cmult(V3[2], cmult(cI, TMP2) - cmult(cI, TMP3)))));
    V1[3]= denom * (TMP4 * (-cI*(P2[1]) + cI*(P3[1])) + (V2[3]*(-cI*(TMP0) + cI*(TMP1)) + V3[3]*(cI*(TMP2) - cI*(TMP3))));
    V1[4]= denom * (TMP4 * (-cI*(P2[2]) + cI*(P3[2])) + (V2[4]*(-cI*(TMP0) + cI*(TMP1)) + V3[4]*(cI*(TMP2) - cI*(TMP3))));
    V1[5]= denom * (TMP4 * (-cI*(P2[3]) + cI*(P3[3])) + (V2[5]*(-cI*(TMP0) + cI*(TMP1)) + V3[5]*(cI*(TMP2) - cI*(TMP3))));*/
    /*
    T TMP0 = (V3[2]*P10   - V3[3]*P11   - V3[4]*P12   - V3[5]*P13);
    T TMP1 = (V3[2]*P20   - V3[3]*P21   - V3[4]*P22   - V3[5]*P23);
    T TMP2 = (P10*V2[2]   - P11*V2[3]   - P12*V2[4]   - P13*V2[5]);
    T TMP3 = (V2[2]*P30   - V2[3]*P31   - V2[4]*P32   - V2[5]*P33);
    T TMP4 = (V3[2]*V2[2] - V3[3]*V2[3] - V3[4]*V2[4] - V3[5]*V2[5]);
    
    T denom = COUP/(P10*P10 - P11*P11 - P12*P12 - P13*P13 - M1 * (M1 -cI* W1));
    
    V1[2]= V3[2] ;//+ V3[2]*(cI*(TMP2) - cI*(TMP3)));
    //V1[2]= denom * (TMP4 * (-cI*(P20) + cI*(P30)) + (V2[2]*(-cI*(TMP0) + cI*(TMP1)) + V3[2]*(cI*(TMP2) - cI*(TMP3))));
    V1[3]= denom * (TMP4 * (-cI*(P21) + cI*(P31)) + (V2[3]*(-cI*(TMP0) + cI*(TMP1)) + V3[3]*(cI*(TMP2) - cI*(TMP3))));
    V1[4]= denom * (TMP4 * (-cI*(P22) + cI*(P32)) + (V2[4]*(-cI*(TMP0) + cI*(TMP1)) + V3[4]*(cI*(TMP2) - cI*(TMP3))));
    V1[5]= denom * (TMP4 * (-cI*(P23) + cI*(P33)) + (V2[5]*(-cI*(TMP0) + cI*(TMP1)) + V3[5]*(cI*(TMP2) - cI*(TMP3))));*/
    /*
    T TMP0 = (cmult(V3[2], P1[0]) - cmult(V3[3], P1[1]) - cmult(V3[4], P1[2]) - cmult(V3[5], P1[3]));
    T TMP1 = (cmult(V3[2], P2[0]) - cmult(V3[3], P2[1]) - cmult(V3[4], P2[2]) - cmult(V3[5], P2[3]));
    T TMP2 = (cmult(P1[0], V2[2]) - cmult(P1[1], V2[3]) - cmult(P1[2], V2[4]) - cmult(P1[3], V2[5]));
    T TMP3 = (cmult(V2[2], P3[0]) - cmult(V2[3], P3[1]) - cmult(V2[4], P3[2]) - cmult(V2[5], P3[3]));
    T TMP4 = (cmult(V3[2], V2[2]) - cmult(V3[3], V2[3]) - cmult(V3[4], V2[4]) - cmult(V3[5], V2[5]));
    
    T TMP42 = -cI * P2[0] + cI * P3[0];
    T TMP4221 = V2[2]*(-cI * TMP0 + cI * TMP1);
    T TMP4222 = V3[2]*(cI * TMP2 - cI * TMP3);
    //printf("%f %f\n", TMP4221.real(), TMP4222.real());
    //printf("%f %f %f\n", TMP4221.real(), TMP4222.real(), TMP4221.real() + TMP4222.real());
    T denom = COUP/(P1[0]*P1[0] - P1[1]*P1[1] - P1[2]*P1[2] - P1[3]*P1[3] - M1 * (M1 -cI* W1));
    const T TMP422 = T(TMP4221.real() + TMP4222.real(), TMP4221.imag() + TMP4222.imag());
    V1[2]= TMP422;
    V1[3]= denom * (TMP4 * (cmult(-cI, P2[1]) + cmult(cI, P3[1])) + (V2[3]*(cmult(-cI, TMP0) + cmult(cI, TMP1)) + V3[3]*(cmult(cI, TMP2) - cmult(cI, TMP3))));
    V1[4]= denom * (TMP4 * (cmult(-cI, P2[2]) + cmult(cI, P3[2])) + (V2[4]*(cmult(-cI, TMP0) + cmult(cI, TMP1)) + V3[4]*(cmult(cI, TMP2) - cmult(cI, TMP3))));
    V1[5]= denom * (TMP4 * (cmult(-cI, P2[3]) + cmult(cI, P3[3])) + (V2[5]*(cmult(-cI, TMP0) + cmult(cI, TMP1)) + V3[5]*(cmult(cI, TMP2) - cmult(cI, TMP3))));*/
}

template <typename T>
__device__ void FFV1_0(const T* F1, const T* F2, const T* V3, const T COUP, T* amp) {
    T cI(0, 1);
    /*
    T v325p = V3[2]+V3[5];
    T v325m = V3[2]-V3[5];
    T v334p = V3[3]+cI*V3[4];
    T v334m = V3[3]-cI*V3[4];
    
    T TMP5 = F1[2] * ( F2[4] * v325p  + F2[5] * v334p)
           + F1[3] * ( F2[4] * v334m + F2[5] * v325m)
           + F1[4] * ( F2[2] * v325m - F2[3] * v334p)
           + F1[5] * (-F2[2] * v334m + F2[3] * v325p);
    
    */
    T TMP5 = (F1[2] * (F2[4] * (V3[2]+V3[5]) + F2[5] * (V3[3] + cI * (V3[4]))) + 
                                 (F1[3] * (F2[4] * (V3[3] - cI * (V3[4])) + F2[5] * (V3[2] - V3[5])) + 
                                 (F1[4] * (F2[2] * (V3[2] - V3[5]) - F2[3] * (V3[3] + cI * (V3[4]))) + 
                                  F1[5] * (F2[2] * (-V3[3] + cI * (V3[4])) + F2[3] * (V3[2] + V3[5])))));
    /*T TMP5 = csum(cmult(F1[2], csum(cmult(F2[4], csum(V3[2], V3[5])), cmult(F2[5], csum(V3[3], cmult(cI, V3[4]))))), 
                                 (csum(cmult(F1[3], csum(cmult(F2[4], cdiff(V3[3], cmult(cI, V3[4]))), cmult(F2[5], cdiff(V3[2], V3[5])))), 
                                 (csum(cmult(F1[4], cdiff(cmult(F2[2], cdiff(V3[2], V3[5]))          , cmult(F2[3], csum(V3[3], cmult(cI, V3[4]))))), 
                                  cmult(F1[5], csum(cmult(F2[2], cdiff(cmult(cI, V3[4]), V3[3])), cmult(F2[3], csum(V3[2], V3[5])))))))));*/
    *amp = COUP * -cI * TMP5;
    //*amp = cmult(COUP, cmult(cmult(-1, cI), TMP5));
}

template <typename T>
__device__ void FFV1_1(const T* F2, const T* V3, const T COUP, double M1_double, double W1_double, T* F1) {
    T cI(0, 1);
    T M1 = M1_double;
    T W1 = W1_double;
    //COMPLEX_TYPE COUP = COUP_comp;
    
    F1[0] = F2[0] + V3[0];
    F1[1] = F2[1] + V3[1];/*
    F1[0] = csum(F2[0], V3[0]);
    F1[1] = csum(F2[1], V3[1]);*/
    
    T P1[4];
    P1[0] = T(-F1[0].real(), 0.0);
    P1[1] = T(-F1[1].real(), 0.0);
    P1[2] = T(-F1[1].imag(), 0.0);
    P1[3] = T(-F1[0].imag(), 0.0);
    
    T denom = COUP/(P1[0]*P1[0] - P1[1]*P1[1] - P1[2]*P1[2] - P1[3]*P1[3] - M1 * (M1 -cI* W1));
    //T denom = cdiv(COUP, cdiff(cdiff(cdiff(cdiff(cmult(P1[0], P1[0]), cmult(P1[1], P1[1])), cmult(P1[2], P1[2])), cmult(P1[3], P1[3])), cmult(M1, cdiff(M1, cmult(cI, W1)))));
    
    F1[2]= denom*cI*(F2[2]*(P1[0]*(-V3[2]+V3[5])+(P1[1]*(V3[3]-cI*(V3[4]))+(P1[2]*(cI*(V3[3])+V3[4])+P1[3]*(-V3[2]+V3[5]))))+(F2[3]*(P1[0]*(V3[3]+cI*(V3[4]))+(P1[1]*(-1./1.)*(V3[2]+V3[5])+(P1[2]*(-1./1.)*(cI*(V3[2]+V3[5]))+P1[3]*(V3[3]+cI*(V3[4])))))+M1*(F2[4]*(V3[2]+V3[5])+F2[5]*(V3[3]+cI*(V3[4])))));
    F1[3]= denom*(-cI)*(F2[2]*(P1[0]*(-V3[3]+cI*(V3[4]))+(P1[1]*(V3[2]-V3[5])+(P1[2]*(-cI*(V3[2])+cI*(V3[5]))+P1[3]*(V3[3]-cI*(V3[4])))))+(F2[3]*(P1[0]*(V3[2]+V3[5])+(P1[1]*(-1./1.)*(V3[3]+cI*(V3[4]))+(P1[2]*(cI*(V3[3])-V3[4])-P1[3]*(V3[2]+V3[5]))))+M1*(F2[4]*(-V3[3]+cI*(V3[4]))+F2[5]*(-V3[2]+V3[5]))));
    F1[4]= denom*(-cI)*(F2[4]*(P1[0]*(V3[2]+V3[5])+(P1[1]*(-V3[3]+cI*(V3[4]))+(P1[2]*(-1./1.)*(cI*(V3[3])+V3[4])-P1[3]*(V3[2]+V3[5]))))+(F2[5]*(P1[0]*(V3[3]+cI*(V3[4]))+(P1[1]*(-V3[2]+V3[5])+(P1[2]*(-cI*(V3[2])+cI*(V3[5]))-P1[3]*(V3[3]+cI*(V3[4])))))+M1*(F2[2]*(-V3[2]+V3[5])+F2[3]*(V3[3]+cI*(V3[4])))));
    F1[5]= denom*cI*(F2[4]*(P1[0]*(-V3[3]+cI*(V3[4]))+(P1[1]*(V3[2]+V3[5])+(P1[2]*(-1./1.)*(cI*(V3[2]+V3[5]))+P1[3]*(-V3[3]+cI*(V3[4])))))+(F2[5]*(P1[0]*(-V3[2]+V3[5])+(P1[1]*(V3[3]+cI*(V3[4]))+(P1[2]*(-cI*(V3[3])+V3[4])+P1[3]*(-V3[2]+V3[5]))))+M1*(F2[2]*(-V3[3]+cI*(V3[4]))+F2[3]*(V3[2]+V3[5]))));
    
    //return F1;
}

template <typename T>
__device__ void FFV1_2(const T* F1, const T* V3, const T COUP, double M2_double, double W2_double, T* F2) {
    T cI(0, 1);
    T M2 = M2_double;
    T W2 = W2_double;
    //COMPLEX_TYPE COUP = COUP_comp;
    
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

#endif  // GOOGLE_CUDA

