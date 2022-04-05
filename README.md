# Codes

## A Viral Marketing-Based Model For Opinion Dynamics in Online Social Networks

### Pre-calculation

Specify network parameters, edge probabilities, and the distributions of innate opinions:

```.\cal.sh```

The dataset resources are contained in the file `data/netsources.txt`.

### Examples

- For **greedy algorithm** on dataset *polbooks*, run  ```./greedy.sh``` inside folder `Algorithm`. The output is stored in `Algorithm/out`.

    -- Inside file `Algorithm/greedy.sh`, parameters are explained with annotations. 

- For **greedy huristics** on toy datasets, run ```./Heu.sh``` inside folder `Heuristics`.
The output is stored in `Heuristics/out`. 

- For **benchmark** on toy datasets, run ```./bench.sh``` inside folder `benchmark`. The output is stored in `benchmark/out`. 

### Environment

The codes are written in *Julia*, it can be downloaded on [https://julialang.org/](https://julialang.org/), the environment is *v1.6*.

- Additioanal external packages are `StatsBase`, `JSON`, `JLD2`, `SparseArrays`, `Laplacians`, `Printf`, `Statistics`, `Random`, `LinearAlgebra`, `DataStructures`, `SpecialFunctions`.  

**Acknowledgement**: Part of the codes are based on code of [Fast Evaluation for Relevant Quantities of Opinion Dynamics](https://github.com/Accelerator950113/OpinionQuantities)