include("Graph_class.jl")
include("Tools.jl")

using Random
using SharedArrays
using Distributed

#addprocs(4)


# """
#     activated_nodes_of_this_round = getRRS_one_round(g::Array{Array{Tuple{Int32, Float64}, 1}, 1}, A, activated_nodes_of_last_round, round)

#     A: Activated nodes of all rounds
# """
# function getRRS_one_round(g :: Array{Array{Tuple{Int32, Float64}, 1}, 1}, A, activated_nodes_of_last_round, delta, round_seed)
#     activated_nodes_of_this_round = Set{Int32}()
#     tmp = 0
#     if round_seed == 1
#         for s in activated_nodes_of_last_round
#             for nb in g[s]
#                 if nb[1] in A
#                     continue
#                 end
#                 tmp = rand(1)[1]
#                 if tmp < nb[2]
#                     push!(activated_nodes_of_this_round, nb[1])
#                 end
#             end
#         end
#     else 
#         for s in activated_nodes_of_last_round
#             for nb in g[s]
#                 if nb[1] in A
#                     continue
#                 end
#                 tmp = rand(1)[1]
#                 if tmp < delta * nb[2]
#                     push!(activated_nodes_of_this_round, nb[1])
#                 end
#             end
#         end       
#     end

#     #activated_nodes_of_this_round = toarray(activated_nodes_of_this_round) # convert to vector
#     #A = vcat(A, activated_nodes_of_this_round) # add to original vector ``
#     return activated_nodes_of_this_round
# end

"""
    list_of_nodes = getRRS(G :: Graph, n, delta)
    A: Activated nodes of all rounds

    2021-07-18, also return the nodes get picked
    
    2021-10-17, combine with getRRS_one_round

"""
function getRRS(g:: Array{Array{Tuple{Int32, Float64}, 1}, 1}, n::Number, delta::Number)
    rr_seeds = rand(1:n, 1)
    tmp = rand(sum([length(g[i]) for i in 1:n])) #m

    A = Set{Int32}(rr_seeds[1])
    round_seed = 1
    activated_nodes_of_last_round = A
    activated_nodes_of_this_round = Set{Int32}()
    count = 1 # random number 

    while length(activated_nodes_of_last_round) != 0
        for s in activated_nodes_of_last_round
            for nb in g[s]
                if nb[1] in A
                    continue
                end

                if round_seed == 1
                    if tmp[count] < nb[2]
                        push!(activated_nodes_of_this_round, nb[1])
                    end
                else
                    if tmp[count] < delta * nb[2]
                        push!(activated_nodes_of_this_round, nb[1])
                    end
                end                        
                count += 1
            end      
        end

        union!(A, activated_nodes_of_this_round)
        activated_nodes_of_last_round = activated_nodes_of_this_round
        activated_nodes_of_this_round = Set{Int32}()
        round_seed += 1
    end

    return toarray(A), rr_seeds[1]
end

"""
    node_selection_influence_max(R, k)

    Node selection procedure

    Return F???(SEED), and SEED
"""
function node_selection_influence_max(R, k)
    copy_k = k
    copy_R = deepcopy(R)

    SEED = zeros(Int32, copy_k)
    size = length(copy_R) # initial size 
    tmp_size,  cm, fc, ris_spread = 0, 0, 0, 0
    # The process of node-selection 
    for i in 1:copy_k
        if length(copy_R) == 0
            copy_k = i - 1 
            break
        end
        flat_list = vcat(copy_R...) # Flat the list 
        (cm, fc) = most_common(flat_list)
        tmp_size += fc
        SEED[i] = cm
        filter!(e -> !(cm in e), copy_R)
    end 

    ris_spread = tmp_size / size 

    return ris_spread, SEED[1:copy_k]
end


"""
    ris_influence_max(G::Graph, k, delta)

    delta: network parameter (ack, spread, reject)
    ???: accuracy parameter

    ris only for inluence maximization algorithm.
    Return: influence_spread, seed nodes, sample_size 

        Tang, Youze, Yanchen Shi, and Xiaokui Xiao. 
        "Influence maximization in near-linear time: A martingale approach." 
        Proceedings of the 2015 ACM SIGMOD International Conference on Management of Data. 2015.
"""
function ris_influence_max(G::Graph, k, delta = 0.5, ?? = 0.3, ??? = 1)
    # Create the "reversed" Link Table
    g = Array{Array{Tuple{Int32, Float64}, 1}, 1}(undef, G.n)
    for i = 1 : G.n
        g[i] = []
    end
   
    for (_, u, v, w) in G.E
        push!(g[v], (u, w))
    end
    
    T = time()

    n = G.n
    ????? = ???2 * ??
    ????? = (2 + 2/3*?????)*(logchoose(n, k) + ??? * log(n) + log(log2(n)))*n / (?????)^2
    ????? = 4*n*(??/3 + 2)*(??? * log(n) + log(2) + logchoose(n, k)) / (??)^2
    LB = 1
    R, SEED = [], []
    ris_spread, x, ????? = 0, 0, 0

    for i in 1 : round(log2(n) - 1)
        x = n / (2^i)
        ????? = ????? / x 
        if length(R) <= ?????
            append!(R, [getRRS(g, n, delta)[1] for _ in 1 : (????? - length(R) + 1)])
        end

        ris_spread, _ = node_selection_influence_max(R, k)

        if n*ris_spread >= (1 + ?????)*x
            LB = n * ris_spread / (1 + ?????)
            break 
        end

    end

    ?? = ????? / LB 
    if length(R) <= ??
        append!(R, [getRRS(g, n, delta)[1] for _ in 1 : (?? - length(R) + 1)])
    end 

    ris_spread, SEED = node_selection_influence_max(R, k)


    T = time() - T
    return ris_spread * n, SEED, length(R), T
end


function choose_high_degree(G::Graph, k)
    T = time()
    g = zeros(G.n)
    for (_, u, _, _) in G.E
        g[u] += 1
    end

    seednodes = topk(g, k)
    T = time() - T

    return T, seednodes
end 

function random_spread(G::Graph, k, delta, samplesize = 10)
    n = G.n 
    tmp = 0
 
    T = time()

    for i = 1 : ceil(Int, log(n) * samplesize)
        rand_k_nodes = rand(1:n, k)
        (_, p4) = cascadeIC(G, rand_k_nodes, round(Int, 2*(10*10)*log(2*n)), delta)
        tmp = tmp .+ p4
    end

    T = time() - T

    return T / ceil(Int, log(n) * samplesize), tmp / ceil(Int, log(n) * samplesize)
end