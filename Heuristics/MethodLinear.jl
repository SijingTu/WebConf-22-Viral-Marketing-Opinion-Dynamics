"""
	This File is used to calculate the Heuristics.  
    Also the init results
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

##################### Below are related to Heuristics
######################


#println("s is ", Δs)
Δaci, Δad, Δap, Δaidc, Δaidc_upp = assign_node_weights(G, s, Δs)
#println("aci is ", Δaci, "ad is ", Δad, "ap is ", Δap, "aidc is ", Δaidc)

lin = [Δs, Δaci, Δad, Δap, Δaidc, Δaidc_upp]
lin_labels = ["max_z_", "max_aci_", "max_ad_", "max_ap_", "max_aidc_", "max_aidc_upp_"]

ris_me, p_me, T_me, R_me = [], [], [], []

# The initial of different measures 
_, init_aci, init_ad, init_ap, init_aidc = Approx(G, s)
init_result = [sum(s), init_aci, init_ad, init_ap, init_aidc, init_aidc] # the last one is a upper bound

init_labels = "init_".*name
for i in 1:length(name)
    merge!(jsdict, Dict(init_labels[i]=>init_result[i]))
end

#println("init aci is ", init_aci)

# Calculation
for i in 1:length(lin)

    # if i <= 5
    #     println("now is ", name[i])
    # else
    #     println("now is aidc_upp")
    # end 
    
    flag = 1

    #if (i == 2) || (i == 3) || (i == 4)
    #    flag = 1
    #end

    (ris, seeds_lin, R_size, T_lin) = ris_measure_linear(G2, k, lin[i], flag, delta) ## RR to select seed nodes
    #if i == 2
    #    println("aci gain linear is ", ris)
    #end
    (_, p) = cascadeIC(G2, seeds_lin, round(Int, 2*(10*10)*log(2*n)), delta) ## direct calculation, just to check 
    push!(ris_me, ris)
    #println("ris_lower_bound is: ", ris)
    push!(p_me, p)
    push!(T_me, T_lin)
    push!(R_me, R_size)

end

# Only in order to check the z_sum 
#println("z_sum is, if we calculated by delta s ", sum(Δs .* p_me[1]))

# Output
for i in 1:length(lin)
    nn = lin_labels[i].*name
    lin_nn = lin_labels[i]*"lin" # only the linear part 
    lin_t = lin_labels[i]*"time"
    lin_size = lin_labels[i]*"r_size"

    s2 = Δs .* p_me[i] .+ s 

    _, aci, ad, ap, aidc = Approx(G, s2)
    z_sum = sum(s2)

    #if lin_labels[i] == "max_aci_"
    #    println("seed aci gain is, ", aci - init_aci)
    #end

    #println("z_sum gain is ", z_sum - sum(s))

    result = [z_sum, aci, ad, ap, aidc]
    merge!(jsdict, Dict(nn[1]=>result[1], nn[2]=>result[2], nn[3]=>result[3], nn[4]=>result[4], nn[5]=>result[5]))
    merge!(jsdict, Dict(lin_t=>T_me[i]))
    merge!(jsdict, Dict(lin_size=>R_me[i]))
    merge!(jsdict, Dict(lin_nn=>ris_me[i] + init_result[i]))
end




#println("sum of s is ", sum(s))

######################
####################### IO

js = JSON.json(jsdict)

println(jsdict)

if ARGS[5] == "1"
    open("out/method_marketing_small_par.json", "a+") do ff
        write(ff, js)
        write(ff, "\n")
    end
elseif ARGS[5] == "2"
    open("out/method_backfire_small_par.json", "a+") do ff
        write(ff, js)
        write(ff, "\n")
    end
end