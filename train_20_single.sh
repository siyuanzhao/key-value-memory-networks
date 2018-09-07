#!/bin/bash
for ((i=1; i<=20; i++))
do
    echo $i
    python single.py --task_id $i --learning_rate 0.001
done
