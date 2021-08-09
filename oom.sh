#!/bin/bash

#ARGUMENT="--events_per_iteration 40000000 --events_per_device 10000"

make clean

rm -f over_memory.txt

list="1 2 5 10 20"
base_value=1000000

make
python cumadflow_exec.py

for i in $list
do
    events=$(($i*${base_value}))
    ARGUMENT="--events_per_iteration $events"
    echo "events per iteration: $events"
    python cumadflow_exec.py ${ARGUMENT} 2>&1 | tee r.txt
    echo "cuda $events" >> over_memory.txt
    grep "(took" r.txt | sed -e "s/.*(took //g" | sed -e "s/ s).*//g" >> over_memory.txt
done

rm r.txt
