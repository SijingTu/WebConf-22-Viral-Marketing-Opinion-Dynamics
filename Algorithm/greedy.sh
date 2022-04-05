#!/bin/bash

# weighted cascade model:1; Trivalency Model:2
for model in 1 ; do
# Uniform distribution:1, Exponential distribution:2; Power-Law distribution:3
    for init_assign in 1 ; do  
# Choose dataset
        for var in Polbooks; do
# Choose the number of seed nodes
            for k in 1 3 5; do
# For the last parameter, 1 means marketing compaign, 2 means polarizing compaign
                julia -O3 NewLinear.jl $var $model $init_assign $k 1
            done
        done
    done 
done
