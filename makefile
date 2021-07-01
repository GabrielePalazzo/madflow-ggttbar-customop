CXX:=g++
NVCC:=/usr/local/cuda-11.3/bin/nvcc

TF_CFLAGS=$(shell python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS=$(shell python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')
#PATH_TO_INCLUDE=$(shell python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
CUDA_LFLAGS= -x cu -Xcompiler -fPIC
CUDA_PATH=/usr/local/cuda-11.3

CSRCS = $(wildcard cpu/*.cc)
GSRCS = $(wildcard gpu/*.cc)
CUDASRC = $(wildcard gpu/*.cu.cc)
SOURCES = $(filter-out $(CUDASRC), $(GSRCS))

TARGET_LIB=matrix.so
TARGET_LIB_CUDA=kernel_example.so

TARGETS=$(TARGET_LIB)

#TARGETS+=$(TARGET_LIB_CUDA)

#OBJECT_SRCS = $(SOURCES:.cc=.o)
#OBJECT_SRCS_CUDA = $(SRCS:.cc=.cudao)

OBJECT_SRCS = $(CSRCS:.cc=.o)
OBJECT_SRCS_CUDA = $(GSRCS:.cc=.cudao)

CFLAGS = ${TF_CFLAGS} -fPIC -O2 -std=c++11
CFLAGS_CUDA = $(CFLAGS) -D GOOGLE_CUDA=1 -I$(CUDA_PATH)/include
CFLAGS_NVCC = ${TF_CFLAGS} -O2 -std=c++11 -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -DNDEBUG --expt-relaxed-constexpr

LDFLAGS = -shared ${TF_LFLAGS}
LDFLAGS_CUDA = $(LDFLAGS) -L$(CUDA_PATH)/lib64 -lcudart

all: $(TARGETS)

cpu: $(TARGET_LIB)

gpu: $(TARGET_LIB_CUDA)

$(TARGET_LIB): $(OBJECT_SRCS)
	$(CXX) -o $@ $(CFLAGS) $^ $(LDFLAGS)

$(TARGET_LIB_CUDA): $(OBJECT_SRCS_CUDA)
	$(CXX) -o $@ $(CFLAGS_CUDA) $^ $(LDFLAGS_CUDA)

%.o: %.cc
	$(CXX) -c $(CFLAGS) $^ -o $@

%.cu.cudao: %.cu.cc
	$(NVCC) -c $(CFLAGS_NVCC) $^ -o $@

%.cudao: %.cc
	$(CXX) -c $(CFLAGS_CUDA) $^ -o $@

#cpu: zero_out.cc
#	$(CXX) -std=c++11 -I $(PATH_TO_INCLUDE) -shared zero_out.cc -o zero_out.so -fPIC $(TF_CFLAGS) $(TF_LFLAGS) -O2
	
#gpu: kernel_example.cc kernel_example.cu.cc kernel_example.h
#	$(NVCC) -std=c++11 -c kernel_example.cu.cc $(CUDA_LFLAGS) -o kernel_example.cu.o
#	$(CXX) -std=c++11 -I $(PATH_TO_INCLUDE) -shared kernel_example.cc kernel_example.cu.o kernel_example.h $(CUDA_LIB) -lcudart -o kernel_example.so -fPIC $(TF_CFLAGS) $(TF_LFLAGS) -O2

test: all zero_out_test.py kernel_example_test.py
	python3 zero_out_test.py
	python3 kernel_example_test.py

clean:
	rm -f $(TARGETS) $(OBJECT_SRCS) $(OBJECT_SRCS_CUDA)

