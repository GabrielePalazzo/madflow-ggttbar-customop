#!/bin/bash

make clean
make

python cumadflow_exec.py 2>&1 | tee r.txt
echo "256" > thread_id.txt
grep "Thread id: " r.txt | sed -e "s/.*Thread id: //g" >> thread_id.txt

sed -e 's&#define DEFAULT_BLOCK_SIZE 256//1024&#define DEFAULT_BLOCK_SIZE 1024&g' gpu/matrix.cu.cc > maketemp
mv maketemp gpu/matrix.cu.cc
make clean
make

python cumadflow_exec.py 2>&1 | tee r.txt
echo "1024" >> thread_id.txt
grep "Thread id: " r.txt | sed -e "s/.*Thread id: //g" >> thread_id.txt

rm -f r.txt

sed -e 's&#define DEFAULT_BLOCK_SIZE 1024&#define DEFAULT_BLOCK_SIZE 256//1024&g' gpu/matrix.cu.cc > maketemp
mv maketemp gpu/matrix.cu.cc
