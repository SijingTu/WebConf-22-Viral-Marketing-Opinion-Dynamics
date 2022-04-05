#!/bin/bash

for init_assign in 1 ; do
    for k in 1 ; do
        for var in ../data/all/* ;do
            julia -O3 CompareFJ.jl $(basename $var .jld2) $init_assign $k
        done
    done
done 
