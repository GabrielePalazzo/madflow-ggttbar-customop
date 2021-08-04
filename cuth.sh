#!/bin/bash

ARGUMENT="--events_per_iteration 40000000 --events_per_device 10000"

make clean

rm -f cudatime.txt

list="32 50 64 100 128 160 192 200 224 256"

make
python cumadflow_exec.py

for i in $list
do
    sed -e "s&#define DEFAULT_BLOCK_SIZE 256//1024&#define DEFAULT_BLOCK_SIZE $i//1024&g" gpu/matrix.cu.cc > maketemp
    mv maketemp gpu/matrix.cu.cc
    export OMP_NUM_THREADS=$i
    echo "CUDA Threads per block: $i"
    make
    python cumadflow_exec.py ${ARGUMENT} 2>&1 | tee r.txt
    echo "custom op cuda $i threads per block" >> cudatime.txt
    grep "(took" r.txt | sed -e "s/.*(took //g" | sed -e "s/ s).*//g" >> cudatime.txt
    sed -e "s&#define DEFAULT_BLOCK_SIZE $i//1024&#define DEFAULT_BLOCK_SIZE 256//1024&g" gpu/matrix.cu.cc > maketemp
    mv maketemp gpu/matrix.cu.cc
done

rm r.txt
