#!/bin/bash

for ((a=0; a <= 15000 ; a+=10)); do
    echo $a
    cp output/weights_layer_1_${a}.pickle output/weights_layer_1.pickle
    python -u main.py params.json -s test --noprogress
done
