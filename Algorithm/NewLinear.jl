include("../src/Graph.jl")
include("../src/Tools.jl")
include("../src/Algorithm.jl")
include("../src/Sampling.jl")
include("../src/Greedy.jl")

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

println(fileName)

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
    #push!(tags, "0.005")
    k = 1
elseif ARGS[4] == "3"
    #push!(tags, "0.015")
    k = 3
elseif ARGS[4] == "5"
    #push!(tags, "0.02")
    k = 5
end

println("n is ", n)
println("m is ", m)
println("k is ", k)

merge!(jsdict, Dict("k"=>k))
merge!(jsdict, Dict("model"=>tags[1], "s_dist"=>tags[2]))

name_greedy = ["aci", "ad", "ap", "aidc"]
name = ["aci", "ad", "ap", "aidc", "aidc"]

##################### Above are configs
#####################
# The only difference for marketing setting and backfire setting is that Δs is different 

if ARGS[5] == "1"
    Δs = delta_s_marketing(s, epsilon)
elseif ARGS[5] == "2"
    Δs = delta_s_backfire(s, epsilon, beta)
end



##################### Below are related to algorithms
######################


#println("s is ", Δs)
Δaci, Δad, Δap, Δaidc, Δaidc_upp = assign_node_weights(G, s, Δs) # linear part weight
lin_qua = [Δaci, Δad, Δap, Δaidc]
lin = [Δaci, Δad, Δap, Δaidc, Δaidc_upp]

Δci, Δd, Δp, Δidc = assign_pair_weights(G, Δs) # quadratic part weight
qua = [Δci, Δd, Δp, Δidc]

lin_labels = ["max_aci_", "max_ad_", "max_ap_", "max_aidc_", "max_aidc_upp_"] # max lin part
qua_labels = ["greedy_aci_", "greedy_ad_", "greedy_ap_", "greedy_aidc_"]

# The initial of different measures 
_, init_aci, init_ad, init_ap, init_aidc = Approx(G, s)
init_result_greedy = [init_aci, init_ad, init_ap, init_aidc, init_aidc]
#
init_labels = "init_".*name_greedy
for i in 1:length(name_greedy)
   merge!(jsdict, Dict(init_labels[i]=>init_result_greedy[i]))
end

#println("init aci is ", init_aci)

######################### For the pairs 

ris_me, T_me, seeds_me, R_me, R_sample = [], [], [], [], []

for i in 1:length(qua)
    (ris, seeds_qua, R_size, T_qua, R) = ris_measure_greedy(G2, k, lin_qua[i], qua[i], delta) ## RR to select seed nodes
    push!(ris_me, ris)
    push!(T_me, T_qua)
    push!(R_me, R_size)
    push!(R_sample, R)
    push!(seeds_me, seeds_qua)
end

for i in 1:length(qua)
    nn = qua_labels[i].*name_greedy
    lin_nn = qua_labels[i]*"rr" ## init rr_algorithm 
    lin_t = qua_labels[i]*"time"
    lin_size = qua_labels[i]*"r_size"


    merge!(jsdict, Dict(nn[i]=>(ris_me[i] + init_result_greedy[i])))
    merge!(jsdict, Dict(lin_t=>T_me[i]))
    merge!(jsdict, Dict(lin_size=>R_me[i]))
end

################### For maximize linear part


ris_me, seeds_me, T_me, R_me = [], [], [], []
# Calculation
for i = 1:length(lin)
    flag = 1
    (ris, seeds_lin, R_size, T_lin) = ris_measure_linear(G2, k, lin[i], flag, delta) ## RR to select seed nodes
    push!(ris_me, ris)
    push!(T_me, T_lin)
    push!(R_me, R_size)
    push!(seeds_me, seeds_lin)
end

for i = 1:length(lin)
    nn = lin_labels[i].*name
    lin_nn = lin_labels[i]*"lin" # only the linear part 
    lin_t = lin_labels[i]*"time"
    lin_size = lin_labels[i]*"r_size"


    if i < 5
        result = evaluation_ris(R_sample[i], n, lin_qua[i], qua[i], seeds_me[i])
    else 
        result = evaluation_ris(R_sample[4], n, lin_qua[4], qua[4], seeds_me[i])
    end

    merge!(jsdict, Dict(nn[i]=>(result + init_result_greedy[i])))
    merge!(jsdict, Dict(lin_t=>T_me[i]))
    merge!(jsdict, Dict(lin_size=>R_me[i]))
    merge!(jsdict, Dict(lin_nn=>ris_me[i] + init_result_greedy[i]))
end



####################### IO

js = JSON.json(jsdict)

println(jsdict)

if ARGS[5] == "1"
    open("out/greedy_marketing.json", "a+") do ff
        write(ff, js)
        write(ff, "\n")
    end
elseif ARGS[5] == "2"
    open("out/greedy_backfire.json", "a+") do ff
        write(ff, js)
        write(ff, "\n")
    end
end