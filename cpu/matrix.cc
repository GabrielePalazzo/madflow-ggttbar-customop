#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

#include <vector>

using namespace tensorflow;

// x = tf.random.uniform([10], dtype=tf.float64)
// x = float_me(np.random.rand(10))

REGISTER_OP("Matrix")
    .Input("to_zero: double")
    .Output("zeroed: double")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });

class Matrix {
 public:
  Matrix();
  ~Matrix() {};
  
  
  double nexternal;
  double ndiags;
  double ncomb;
  //initial_states = [[21, 21]]
  bool mirror_initial_states;
  Tensor helicities;
  double denominator;
  
  void smatrix(const Tensor* all_ps);
  void matrix(const Tensor* all_ps, Tensor hel);
};
    
class MatrixOp : public OpKernel {
 public:
  explicit MatrixOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<double>();

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    auto output_flat = output_tensor->flat<double>();

    // Do some stuff here...
    
    //std::cout << "Input: " << input << std::endl;
    
    Matrix mat;
    mat.smatrix(&input_tensor);

  }
};

Matrix::Matrix() {
    double hel[] = {-1,-1,-1,1,
        -1,-1,-1,-1,
        -1,-1,1,1,
        -1,-1,1,-1,
        -1,1,-1,1,
        -1,1,-1,-1,
        -1,1,1,1,
        -1,1,1,-1,
        1,-1,-1,1,
        1,-1,-1,-1,
        1,-1,1,1,
        1,-1,1,-1,
        1,1,-1,1,
        1,1,-1,-1,
        1,1,1,1,
        1,1,1,-1};
    helicities = Tensor(DT_DOUBLE, TensorShape({16, 4}));
    
    for (int i = 0; i < 4 * 16; i++)
        helicities.flat<double>()(i) = hel[i];
    
    nexternal = 4;
    ndiags = 3;
    ncomb = 16;
    //initial_states = [[21, 21]]
    mirror_initial_states = false;
    denominator = 256;
}

void Matrix::smatrix(const Tensor* all_ps) {
    //std::cout << "all_ps: " << all_ps->flat<double>() << std::endl;
    //std::cout << "helicities: " << helicities.flat<double>() << std::endl;
    
    TensorShape nevts = all_ps->shape();
    std::cout << "nevts: " << nevts << std::endl;
    Tensor ans = Tensor(DT_DOUBLE, nevts);
    
    for (int i = 0; i < helicities.dim_size(0); i++) {
        Tensor hel = helicities.Slice(i, i+1);
    }
    /*
        nevts = tf.shape(all_ps, out_type=DTYPEINT)[0]
        ans = tf.zeros(nevts, dtype=DTYPE)
        for hel in self.helicities:
            ans += self.matrix(all_ps,hel,mdl_MT,mdl_WT,GC_10,GC_11)
        print(DTYPE, nevts)
        return ans/self.denominator*/
}

void Matrix::matrix(const Tensor* all_ps, Tensor hel) {
    int ngraphs = 3;
    int nwavefuncs = 5;
    int ncolor = 2;
    double ZERO = 0.;
}

REGISTER_KERNEL_BUILDER(Name("Matrix").Device(DEVICE_CPU), MatrixOp);

