#!/bin/bash

export CUDA_VISIBLE_DEVICES=""
ARGUMENT="--events_per_iteration 40000000 --events_per_device 10000"

python madflow_exec.py ${ARGUMENT} 2>&1 | tee r.txt
echo "python" > exectime.txt
grep "(took" r.txt | sed -e "s/.*(took //g" | sed -e "s/ s).*//g" >> exectime.txt

sed -e 's/OMP_CFLAGS = -fopenmp/OMP_CFLAGS = #-fopenmp/g' makefile > maketemp
mv maketemp makefile
make clean
make

python cmadflow_exec.py ${ARGUMENT} 2>&1 | tee r.txt
echo "custom op no omp" >> exectime.txt
grep "(took" r.txt | sed -e "s/.*(took //g" | sed -e "s/ s).*//g" >> exectime.txt

sed -e 's/OMP_CFLAGS = #-fopenmp/OMP_CFLAGS = -fopenmp/g' makefile > maketemp
mv maketemp makefile
make clean
make

for i in {1..12}
do
    export OMP_NUM_THREADS=$i
    echo "export OMP_NUM_THREADS=$i"
    python cmadflow_exec.py ${ARGUMENT} 2>&1 | tee r.txt
    echo "custom op omp $i threads" >> exectime.txt
    grep "(took" r.txt | sed -e "s/.*(took //g" | sed -e "s/ s).*//g" >> exectime.txt
done

rm r.txt
rm rr.txt
