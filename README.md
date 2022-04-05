# Codes

## A Viral Marketing-Based Model For Opinion Dynamics in Online Social Networks

### Pre-calculation

Pre-calculate network parameters, transmission probabilities, and $s$ distributions:

```.\cal.sh```

The dataset resources are contained in the file `data/netsources.txt`.

### Examples

- For **greedy algorithm** on dataset *polbooks*, run  ```./greedy.sh``` inside folder `Algorithm`. The output is stored in `Algorithm/out`.

    -- Inside file `Algorithm/greedy.sh`, parameters are explained with annotations. 

- For **greedy huristics** on toy datasets, run ```./Heu.sh``` inside folder `Heuristics`.
The output is stored in `Heuristics/out`. 

- For **benchmark** on toy datasets, run ```./bench.sh``` inside folder `benchmark`. The output is stored in `benchmark/out`. 

**Acknowledge**: Part of the codes are based on code of [Fast Evaluation for Relevant Quantities of Opinion Dynamics](https://github.com/Accelerator950113/OpinionQuantities)
