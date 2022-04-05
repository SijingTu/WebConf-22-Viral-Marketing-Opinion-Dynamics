"""
	This File is used to pre-calculate some network parameters
	Transmission probabilities
	S distributions
"""

include("src/Graph.jl")
include("src/Tools.jl")
include("src/Algorithm.jl")
include("src/Sampling.jl")

using StatsBase
using JSON
using JLD2 # store data

# Read Graph
buf = split(ARGS[1], ',')
fileName = string("data/raw/", buf[1], ".txt")
networkType = "unweighted"
if size(buf, 1) > 1
	networkType = buf[2]
end
# Find LLC
G0 = readGraph(fileName, networkType)
G = getLLC(G0)
n = G.n


######################## Edge Weight

# Weighted Cascade Model
G21 = assignWeightedCascade(G) 
# Trivalency Model
G22 = assignTrivalency(G)


####################### Assign s

# S Uniform distribution
s1 = Uniform(G.n)
# S exponential distribution
s2 = Exponential(G.n)
# S Power-law distribution
s3 = powerLaw(G.n)


##################### Assign k

# Different parameters to select k
k1 = ceil(Int, 0.005 * n)
k2 = ceil(Int, 0.015 * n)
k3 = ceil(Int, 0.020 * n)


FileOut = "data/all/"*buf[1]*".jld2"
save(FileOut, Dict("name"=>buf[1], "n"=>n, "m"=>G.m, "V"=>G.V, "E"=>G.E, "E_WC"=>G21.E, "E_TM"=>G22.E, "s_Uni"=>s1, "s_Exp"=>s2, "s_Pow"=>s3, "k1"=>k1, "k2"=>k2, "k3"=>k3))