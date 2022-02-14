"""
Compare our results with FJ model, how to choose k

Only choose n <= 5000 nodes 
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

if n > 3000
    println("large graph")
    exit()
end

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

mea = ["aci", "ad", "ap", "aidc"]
prefix = ["max_aci_", "max_ad_", "max_ap_", "max_aidc_"]
newpre = ["mark_", "back_"]

MxList = [Mic, Md, Mp, Midc]


####################

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

function choosek(Mx, s, n, epsilon, measure)
    left = 1
    right = n
    mid = floor((left + right)/ 2)


    while true 
        Tr = greedyFJ(Mx, s, n, mid, epsilon)
        if  Tr > measure 
            right = mid 
        elseif Tr <= measure 
            left = mid 
        end 
        mid = floor((left + right)/ 2)
        if (mid == left) || (mid == right) 
            break
        end
    end

    return left 
end

#println("test if out")

############ itrate all possible measures 
allLines = [readlines("comfolder/method_marketing.json"), readlines("comfolder/method_backfire.json")]
dic = [Dict{String, Any}(), Dict{String, Any}()]

for i = 1:2
    for j in allLines[i]
        global dic 
        dic[i] = JSON.parse(j)
        if dic[i]["name"] == name 
            break
        end
    end
end

#println(dic)

function ReturnK(MxList, s, n, epsilon, dic)
    newdic = Dict{String, Any}()
    for i = 1:2
        for j in 1:length(prefix)
            k = choosek(MxList[i], s, n, epsilon, dic[i][prefix[j]*mea[j]] - dic[i]["init_"*mea[j]])
            merge!(newdic, Dict(newpre[i]*mea[j]=>k)) 
        end
    end
    return newdic 
end 

newdic = ReturnK(MxList, s, n, epsilon, dic)

merge!(jsdict, newdic)

########## Write 

js = JSON.json(jsdict)
println(jsdict)  

open("out/fj_choose_k.json", "a+") do ff
    write(ff, js)
    write(ff, "\n")
end
