"""
Compare our results with FJ model (and upper bound)

If we directly modify k nodes' innate opinions, each one modify with at most epsilon, 
how much will it influence the final measures?

Ref:
    Gaitonde, Jason, Jon Kleinberg, and Eva Tardos. 
    "Adversarial perturbations of opinion dynamics in networks." Proceedings of the 21st ACM Conference on Economics and Computation. 2020.

    Chen, Mayee F., and Miklos Z. Racz. 
    "Network disruption: maximizing disagreement and polarization in social networks." arXiv preprint arXiv:2003.08377 (2020).
"""

include("../src/Graph.jl")
include("../src/Tools.jl")
include("../src/Algorithm.jl")
include("../src/Sampling.jl")
include("../src/Methods.jl")

using LinearAlgebra
using StatsBase
using JSON
using JLD2

epsilon = 0.1

jsdict = Dict{String, Any}()
tags = []

# Read Graph
buf = split(ARGS[1], ',')
fileName = string("../data/all/", buf[1], ".jld2")

(name, n, m, V, E) = load(fileName, "name", "n", "m", "V", "E")
G = Graph(n, m, V, E)

merge!(jsdict, Dict("name"=>name, "n"=>n, "m"=>m)) 

# Before Making changes
if ARGS[2] == "1"
    # Uniform distribution
    push!(tags, "Uniform")
    s = load(fileName, "s_Uni")
elseif ARGS[2] == "2"
    # Exponential distribution
    push!(tags, "Exponential")
    s = load(fileName, "s_Exp")
elseif ARGS[2] == "3"
    # Power-law distribution
    push!(tags, "Power-law")
    s = load(fileName, "s_Pow")
end

# Different parameters to select k
if ARGS[3] == "1"
    push!(tags, "0.005")
    k = load(fileName, "k1")
elseif ARGS[3] == "2"
    push!(tags, "0.015")
    k = load(fileName, "k2")
elseif ARGS[3] == "3"
    push!(tags, "0.02")
    k = load(fileName, "k3")
end


merge!(jsdict, Dict("k"=>k))
merge!(jsdict, Dict("s_dist"=>tags[1], "k_par"=>tags[2]))

name = ["z_sum", "aci", "ad", "ap", "aidc"]

##################### Init 


# The initial of different measures 
_, init_aci, init_ad, init_ap, init_aidc = Approx(G, s)
init_result = [sum(s), init_aci, init_ad, init_ap, init_aidc, init_aidc] # the last one is a upper bound

init_labels = "init_".*name
for i in 1:length(name)
    merge!(jsdict, Dict(init_labels[i]=>init_result[i]))
end


#################### Changes 
# Precalculation

L = Symmetric(getL(G))

W = getW(G)
W = (W + W') / 2
#println(issymmetric(W))

Mp = W*W - ones(n, n)/n
Mp = (Mp' + Mp)/2

Md = W*L*W 
Md = (Md' + Md)/2

Mic = W*L*L*W 
Mic = (Mic' + Mic)/2

Midc = W


s_sqrt = sqrt(sum([i^2 for i in s]))


g = zeros(n)

for (ID, u, v, w) in G.E
    g[v] += 1
    g[u] += 1
end

d_max = maximum(g) / 2

########################################### Upper bound 

# Polarization
maxEigen = eigmax(Mp)
uppChange = maxEigen * epsilon^2 * k + min(2*k*epsilon, 2*sqrt(k)*epsilon*maxEigen*s_sqrt)
merge!(jsdict, Dict("FJ_upp_ap"=>uppChange))
#println("polarization ", uppChange)

# Disagreement 
maxEigen = eigmax(Md)
uppChange = maxEigen * epsilon^2 * k + min(2*k*epsilon*d_max, 2*sqrt(k)*epsilon*maxEigen*s_sqrt)
merge!(jsdict, Dict("FJ_upp_ad"=>uppChange))
#println("Disagreement ", uppChange)

# Internal conflict 
maxEigen = eigmax(Mic)
uppChange = maxEigen * epsilon^2 * k + 2*sqrt(k)*epsilon*maxEigen*s_sqrt
merge!(jsdict, Dict("FJ_upp_aci"=>uppChange))
#println("IC ", uppChange)

# Disagreement-Controversy
maxEigen = eigmax(Midc)
uppChange = maxEigen * epsilon^2 * k + 2*sqrt(k)*epsilon*maxEigen*s_sqrt
merge!(jsdict, Dict("FJ_upp_aidc"=>uppChange))
#println("IDC ", uppChange)


########################################### Algorithm
function greedyFJ(Mx, s, n, k, epsilon)
    s_new = deepcopy(s)

    Lvec = 2*Mx*s 
    cSet = Set{Int}()
    AllSet = BitSet(1:n)
    epVec = zeros(n)

    TotalGain = 0

    for i = 1:k
        MaxGain = 0
        MaxIdx = 0
        MaxFlag = 0
        for j in AllSet
            TmpGain1 = Lvec[j]*epsilon + sum([Mx[t, j]*epVec[t] + Mx[j, t]*epVec[t] for t in cSet])*(epsilon) + Mx[j, j]*epsilon^2
            TmpGain2 = -Lvec[j]*epsilon + sum([Mx[t, j]*epVec[t] + Mx[j, t]*epVec[t] for t in cSet])*(-epsilon) + Mx[j, j]*epsilon^2
            if  TmpGain1 > MaxGain
                MaxIdx = j
                MaxGain = TmpGain1
                MaxFlag = 1
            end
            if TmpGain2 > MaxGain
                MaxIdx = j
                MaxGain = TmpGain2
                MaxFlag = 2
            end 
        end
        if MaxIdx == 0
            break
        else 
            if MaxFlag == 1
                epVec[MaxIdx] = epsilon
                s_new[MaxIdx] = s[MaxIdx] + epsilon 
            elseif MaxFlag == 2
                epVec[MaxIdx] = epsilon
                s_new[MaxIdx] = s[MaxIdx] - epsilon 
            end
            push!(cSet, MaxIdx)
            delete!(AllSet, MaxIdx)
            TotalGain += MaxGain
        end
    end 

    #println("chosen nodes are ", cSet)
    #println("changes is: ", s_new'*Mx*s_new - s'*Mx*s, " or ", TotalGain, " or ", epVec'*Lvec + epVec'*Mx*epVec)
    return TotalGain 
end


# Polarization
Change = greedyFJ(Mp, s, n, k, epsilon)
merge!(jsdict, Dict("FJ_ap"=>Change))
#println("polarization ", uppChange)

# Disagreement 
Change = greedyFJ(Md, s, n, k, epsilon)
merge!(jsdict, Dict("FJ_ad"=>Change))
#println("Disagreement ", uppChange)

# Internal conflict 
Change = greedyFJ(Mic, s, n, k, epsilon)
merge!(jsdict, Dict("FJ_aci"=>Change))
#println("IC ", uppChange)

# Disagreement-Controversy
Change = greedyFJ(Midc, s, n, k, epsilon)
merge!(jsdict, Dict("FJ_aidc"=>Change))
#println("IDC ", uppChange)


# zsum 
merge!(jsdict, Dict("FJ_z_sum"=>k*epsilon))

#println("zsum ", k*epsilon)

js = JSON.json(jsdict)

println(jsdict)  

open("out/fj_compare.json", "a+") do ff
    write(ff, js)
    write(ff, "\n")
end
