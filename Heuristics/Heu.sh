#!/bin/bash

for model in 1 ; do
    for init_assign in 2 ; do
        for k in 1 ; do
            for var in ../data/raw/* ;do
                julia -O3 MethodLinear.jl $(basename $var .jld2) $model $init_assign $k 1
            done
        done
    done 
done
