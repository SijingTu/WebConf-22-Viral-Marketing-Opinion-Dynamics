"""
	This File is used to calculate the benchmarks. 
    1, Influence Maximization
    2, Max Degree 
    3, Random Sampling
"""

include("../src/Graph.jl")
include("../src/Tools.jl")
include("../src/Algorithm.jl")
include("../src/Sampling.jl")
include("../src/Methods.jl")

using StatsBase
using JSON
using JLD2

"""
Set Seed Nodes & change innate opinions 
Set parameters.
k = 50
epsilon : s + epsilon
delta : edge probability parameter
beta: threshold
json struct js
"""

epsilon = 0.1
delta = 0.5
beta = 0.5
tags = []

jsdict = Dict{String, Any}()

# Read Graph
buf = split(ARGS[1], ',')
fileName = string("../data/all/", buf[1], ".jld2")

(name, n, m, V, E) = load(fileName, "name", "n", "m", "V", "E")
G = Graph(n, m, V, E)

merge!(jsdict, Dict("name"=>name, "n"=>n, "m"=>m)) 

if ARGS[2] == "1"
    #Weighted Cascade
    push!(tags, "WC")
    E_p = load(fileName, "E_WC")
elseif ARGS[2] == "2"
    #Trivalency
    push!(tags, "TM") # Trivalency Model
    E_p = load(fileName, "E_TM")
end
G2 = Graph(n, m, V, E_p)

# Before Making changes
if ARGS[3] == "1"
    # Uniform distribution
    push!(tags, "Uniform")
    s = load(fileName, "s_Uni")
elseif ARGS[3] == "2"
    # Exponential distribution
    push!(tags, "Exponential")
    s = load(fileName, "s_Exp")
elseif ARGS[3] == "3"
    # Power-law distribution
    push!(tags, "Power-law")
    s = load(fileName, "s_Pow")
end

# Different parameters to select k
if ARGS[4] == "1"
    push!(tags, "0.005")
    k = load(fileName, "k1")
elseif ARGS[4] == "2"
    push!(tags, "0.015")
    k = load(fileName, "k2")
elseif ARGS[4] == "3"
    push!(tags, "0.02")
    k = load(fileName, "k3")
end


merge!(jsdict, Dict("k"=>k))
merge!(jsdict, Dict("model"=>tags[1], "s_dist"=>tags[2], "k_par"=>tags[3]))

name = ["z_sum", "aci", "ad", "ap", "aidc"]

##################### Above are configs
#####################
# The only difference for marketing setting and backfire setting is that Δs is different 

if ARGS[5] == "1"
    Δs = delta_s_marketing(s, epsilon)
elseif ARGS[5] == "2"
    Δs = delta_s_backfire(s, epsilon, beta)
end


#######################
####################### Baselines

base_labels = ["max_influence_", "high_degree_", "random_"]
base_time = []
base_p = []

# rr-algorithm
(_, seeds, _, T) = ris_influence_max(G2, k, delta) ## RR to select seed nodes
(_, p_rr) = cascadeIC(G2, seeds, round(Int, 2*(10*10)*log(2*n)), delta) ## direct calculation, just to check 
push!(base_time, T)
push!(base_p, p_rr)

# high-degree-algorithm
(T, seeds_high_degree) = choose_high_degree(G2, k)
(_, p_degree) = cascadeIC(G2, seeds_high_degree, round(Int, 2*(10*10)*log(2*n)), delta) ## direct calculation, just to check 
push!(base_time, T)
push!(base_p, p_degree)

# random 
(T, p_random) = random_spread(G2, k, delta)
push!(base_time, T)
push!(base_p, p_random)


for i in 1:length(base_labels)
    nn = base_labels[i].*name
    base_t = base_labels[i]*"time"

    s2 = Δs .* base_p[i] .+ s 

    _, aci, ad, ap, aidc = Approx(G, s2)
    z_sum = sum(s2)

    #println("z_sum gain is ", z_sum - sum(s))

    result = [z_sum, aci, ad, ap, aidc]

    merge!(jsdict, Dict(nn[1]=>result[1], nn[2]=>result[2], nn[3]=>result[3], nn[4]=>result[4], nn[5]=>result[5]))
    merge!(jsdict, Dict(base_t=>base_time[i]))
end

js = JSON.json(jsdict)

println(jsdict)

if ARGS[5] == "1"
    open("out/log_benchmark_marketing.json", "a+") do ff
        write(ff, js)
        write(ff, "\n")
    end
elseif ARGS[5] == "2"
    open("out/log_benchmark_backfire.json", "a+") do ff
        write(ff, js)
        write(ff, "\n")
    end
end