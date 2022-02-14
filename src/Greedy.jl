include("Graph_class.jl")
include("Tools.jl")
include("Sampling.jl")
include("Algorithm.jl")

using LinearAlgebra


"""
    node_selection_linear_sum(R, k)

    (1) R: a list of tuples, each tuple contains a rr-set, and its corresponding sampled vertex
    (2) n: number of vertices in the graph
    (3) node_weight: a list of node weights
    (4) k: budget for seednodes.  

    % Inside the function
    (1) reverse_R_list Vector{Set{Int}}: for each vertex u, stores all the IDs of rr sets that u covers. The values will be updated in the function
    (2) value_list: PriorityQueue{Int, Float64}(Base.Order.Reverse) for each vertex u, stores the sum of weights of the rr sets that u covers. The values will be updated in the function
    (3) weight_list: each rr_set contains a weight, the weight equals to the weight of corresponding vertex if covered. 
    (4) size: The length of the rr set
"""
function node_selection_linear_sum(R, n, node_weight, k)
    T = time()

    rrl = [r[1] for r in R]
    nodel = [r[2] for r in R] # the sampled vertex 

    #println(R)

    len_rrs = length(rrl)

    rR = [Set{Int}() for _ in 1:n]
    vl = PriorityQueue{Int32, Float64}(Base.Order.Reverse) #init as priorityQueue
    wl = [node_weight[i] for i in nodel]
    tmp_rr = []
    tmp_rr_another = []

    rr3D = falses(len_rrs, n) # Init a bit matrix for quick check 
    for j in 1:length(rrl) 
        for i in rrl[j]
            rr3D[j, i] = true
        end 
    end 

    #println("Time prepare", time() - T)

    for i in 1:n
        #tmp_rr = [j for j in 1:len_rrs if i in rrl[j]] # All the rr IDs covered by node i
        tmp_rr = [j for j in 1:len_rrs if rr3D[j, i] == true]
        #println("equal or not", isequal(tmp_rr, tmp_rr_another))
        union!(rR[i], tmp_rr)
        #rR[i] = Set{Int}(tmp_rr) 
        #println([weight_list[k] for k in tmp_rr])
        vl[i] = sum([wl[k] for k in rR[i]])
    end

    #println("Time first loop", time() - T)

    SEED = zeros(Int32, k) # In oder to store the seeds
    cR = Set{Int}() # The rr-sets going to be covered
    tmp_value, ris_value = 0, 0
    k_copy = k 
    #tmp_size,  cm, fc, ris_value = 0, 0, 0, 0
    # The process of node-selection 
    for j in 1:k
        SEED[j] = peek(vl)[1] # Choose the node with the highest priority
        if peek(vl)[2] < 0 #if can not find a seed with positive addition
            k_copy = j - 1
            break
        end
        tmp_value += peek(vl)[2]
        #println("peek value is ", peek(vl)[2])
        #println("seed is ", SEED[j])
        cR = rR[SEED[j]] 
        #println("cR are", cR)
        for idx in 1 : n
            #println("wl is ")
            #println(wl)
            #println("intersection is")
            #println("vl[idx] is ", vl[idx])
            vl[idx] -= sum(wl[[v for v in intersect(rR[idx], cR)]]) # Update value_list
            rR[idx] = setdiff(rR[idx], cR) # update reverse_R_list
        end
    end 

    #println("Time second loop", time() - T)

    ris_value = tmp_value / len_rrs 

    return ris_value, SEED[1:k_copy]
end


"""
    ris_measure_linear(G::Graph, k, node_weight, delta)

    delta: network parameter (ack, spread, reject)
    ℓ: accuracy parameter

    ris for maximize the measures of their linear part. 
    Return: ris_value, seed nodes, sample_size 

    flag: For some indices that are so small, we set a upper bound on sampling size (aci, ad, ap)

        Tang, Youze, Yanchen Shi, and Xiaokui Xiao. 
        "Influence maximization in near-linear time: A martingale approach." 
        Proceedings of the 2015 ACM SIGMOD International Conference on Management of Data. 2015.
""" 
function ris_measure_linear(G::Graph, k, node_weight, flag = 0, delta = 0.5, ϵ = 0.4, ℓ = 1)
    # Create the "reversed" Link Table
    g = Array{Array{Tuple{Int32, Float64}, 1}, 1}(undef, G.n)
    for i = 1 : G.n
        g[i] = []
    end
   
    for (_, u, v, w) in G.E
        push!(g[v], (u, w))
    end
    
    T = time()

    χ = maximum(abs.(node_weight))

    n = G.n
    ϵ′ = 2 * ϵ # let ϵ' large enough
    λ′ = (2 + 4/3*ϵ′)*(logchoose(n, k) + ℓ * log(n) + log(log2(n)))*n / (ϵ′)^2

    λ⁺ = 8*n*χ*(ϵ/3 + 1)*(ℓ * log(n) + log(2) + logchoose(n, k)) / (ϵ)^2
    #LB = χ
    LB = 1 # when LB < 1, then we do not need much accuracy
    R, SEED = [], []
    ris_value, x, θᵢ = 0, 0, 0

    #println("before loop time is ", time() - T)

    for i in 1 : round(log2(n) - 1)
        y = n / (2^i)
        θᵢ = λ′ / y 
        if length(R) <= θᵢ
            append!(R, [getRRS(g, n, delta) for _ in 1 : (θᵢ - length(R) + 1)])
        end

        if (flag == 1) && (length(R) >= n*50) # add condition to avoid too much sampling
            break
        end

        #println("loop is ", i)
        #println("Time Before selection is ", time() - T)

        ris_value, _ = node_selection_linear_sum(R, n, node_weight, k)

        #println("Time After selection is ", time() - T)

        if ris_value * n < 1 # avoid too many sampling
            #println("small value")
            break 
        else 
            if n*ris_value >= (1 + ϵ′)*y*χ
                LB = n * ris_value / (1 + ϵ′)
                break 
            end
        end
    end

    θ = λ⁺ / LB 
    if (length(R) <= θ)
        if (θ <= n*50) || (flag == 0)
            append!(R, [getRRS(g, n, delta) for _ in 1 : (θ - length(R) + 1)])
        else
            #println("avoid much sampling")
            append!(R, [getRRS(g, n, delta) for _ in 1 : (n*50 - length(R) + 1)])
        end
    end

    #println("Time Before last calculation is ", time() - T)

    ris_value, SEED = node_selection_linear_sum(R, n, node_weight, k)

    #println("Time After last calculation is ", time() - T)

    T = time() - T
    return ris_value * n, SEED, length(R), T
end

## Below for greedy algorithm

function evaluation_ris(R, n, node_weight, pair_weight, seeds)
    rrl = [r[1] for r in R]
    nodel = [r[2] for r in R] # node pair in this case

    len_rrs = length(R)
    nR = div(length(R), 2)

    rR = Set{Int}()
    wl = [n*pair_weight[nodel[i], nodel[i + nR]] for i in 1:nR]
    sl = [node_weight[nodel[i]] for i in 1:nR]

    for i in seeds
        union!(rR, [j for j in 1:len_rrs if i in rrl[j]]) # All the rr IDs covered by node i
    end

    return sum([wl[k] for k in rR if (k + nR) in rR])* n / nR + sum([sl[k] for k in rR if k <= nR])* n / nR # as init, i should cover the pair of sampled rr-id
end 



# For general greedy algorithm
function node_selection_pair(R, n, node_weight, pair_weight, k)
    rrl = [r[1] for r in R]
    nodel = [r[2] for r in R] # node pair in this case

    len_rrs = length(R)
    tmp_rr = []
    nR = div(length(R), 2)

    rR = [Set{Int}() for _ in 1:n]
    vl = PriorityQueue{Int32, Float64}(Base.Order.Reverse) #init as priorityQueue
    wl = [n*pair_weight[nodel[i], nodel[i + nR]] for i in 1:nR] # weight for the pairs 
    sl = [node_weight[nodel[i]] for i in 1:nR] # weight for single 
    cover = [false for _ in 1 : len_rrs] # mark if a node is covered in general 
    valid_cover = [false for _ in 1 : nR] # only when the gain is achived

    rr3D = falses(len_rrs, n) # Init a bit matrix for quick check 
    for j in 1:length(rrl) 
        for i in rrl[j]
            rr3D[j, i] = true
        end 
    end 

    for i in 1:n
        tmp_rr = [j for j in 1:len_rrs if rr3D[j, i] == true] # All the rr IDs covered by node i
        rR[i] = Set{Int}(tmp_rr)
        vl[i] = sum([wl[k] for k in rR[i] if (k + nR) in rR[i]]) # as init, i should cover the pair of sampled rr-id, only for pairs 
        vl[i] += sum([sl[k] for k in rR[i] if k <= nR]) # the single weight 
    end
    

    SEED = zeros(Int32, k)
    cR = Set{Int}() # The rr-sets going to be covered
    tmp_value, ris_value = 0, 0
    k_copy = k 
    # The process of node-selection 
    for j in 1:k
        SEED[j] = peek(vl)[1] # Choose the node with the highest priorit

        NewValid, NewValid_pair, TmpInter, TmprDiff = Set{Int}(), Set{Int}(), Set{Int}(), Set{Int}() # newly real-covered node, to update valid_cover, only stores the id < n; TmpInter 
        if peek(vl)[2] < 0 #if can not find a seed with positive addition
            k_copy = j - 1
            break
        end
        tmp_value += peek(vl)[2]
        cR = rR[SEED[j]] # covered rr-ids by seed[j]

        # check all the covers, can they make pairs?
        for c in cR
            #println("c is ", c)
            if c > nR 
                if (cover[c - nR] == true) 
                    push!(NewValid, c)
                    valid_cover[c - nR] = true
                end
            else
                if (cover[c + nR] == true)
                    push!(NewValid, c)
                    valid_cover[c] = true
                end
            end
        end

        # check all the pairs in cR, in order to make pairs
        for c in cR
            if c < nR
                if (c + nR) in cR
                    push!(NewValid_pair, c)
                    valid_cover[c] = true
                end
            end
        end 
    
        # update cover
        for c in cR
            cover[c] = true # update cover
        end

        #println("NewValid is ", NewValid)

        for idx in 1 : n
            TmprDiff = setdiff(rR[idx], cR) # update reverse_R_list 
            TmpInter = intersect(rR[idx], cR)

            vl[idx] -= sum(sl[[v for v in TmpInter if v <= nR]]) # for single nodes

            for v in TmpInter 
                 # Update value_list, subtraction
                if v in NewValid 
                    vl[idx] -= wl[(v > nR) ? v - nR : v]
                end
                if (v in NewValid_pair) && ((v + nR) in TmpInter)
                    vl[idx] -= wl[v]
                end
            end
            
            rR[idx] = TmprDiff # update set 

            for v in TmprDiff 
                # Update value_list, avoid pair
                if (((v + nR) in cR) && ((v + nR) ∉ TmpInter))  || (((v - nR) in cR) && ((v - nR) ∉ TmpInter))
                    vl[idx] += wl[(v > nR) ? v - nR : v] # Update value_list
                end 
            end
        end

    end 

    ris_value = tmp_value / nR

    return ris_value, SEED[1:k_copy]
end


function ris_measure_greedy(G::Graph, k, node_weight, pair_weight, delta = 0.5, ϵ = 0.3, ℓ = 1)
    # Create the "reversed" Link Table
    g = Array{Array{Tuple{Int32, Float64}, 1}, 1}(undef, G.n)
    for i = 1 : G.n
        g[i] = []
    end
   
    for (_, u, v, w) in G.E
        push!(g[v], (u, w))
    end
    
    T = time()

    χ = maximum(abs.(node_weight)) + maximum(abs.(pair_weight)) * G.n

    n = G.n
    ϵ′ = √2 * ϵ # let ϵ' large enough
    λ′ = (2 + 4/3*ϵ′)*(logchoose(n, k) + ℓ * log(n) + log(log2(n)))*n / (ϵ′)^2

    λ⁺ = 8*n*χ*(ϵ/3 + 1)*(ℓ * log(n) + log(2) + logchoose(n, k)) / (ϵ)^2
    #LB = χ
    LB = 1 # when LB < 1, then we do not need much accuracy
    R, SEED = [], []
    ris_value, θᵢ = 0, 0
    UppSample = n * n * 20  # set upper bound on sampling size

    for i in 1 : round(log2(n))
        y = n^2 / (2^i)
        θᵢ = div(λ′, y) 
        if length(R) <= θᵢ
            append!(R, [getRRS(g, n, delta) for _ in 1 :  2*(θᵢ - length(R) + 1)])
        end

        ris_value, _ = node_selection_pair(R, n, node_weight, pair_weight, k)

        if (length(R) >= UppSample) # add condition to avoid too much sampling
            break
        end

        if ris_value * n < 1 # avoid too many sampling
           break 
        else 
            if n * ris_value >= (1 + ϵ′)*y*χ
                LB = n * ris_value / (1 + ϵ′)
                break 
            end
        end
    end

    θ = div(λ⁺, LB) 
    if (length(R) <= θ)
        if (θ <= UppSample)
            append!(R, [getRRS(g, n, delta) for _ in 1 : 2*(θ - length(R) + 1)])
        else
            append!(R, [getRRS(g, n, delta) for _ in 1 : 2*(UppSample - length(R) + 1)])
        end
    end
    #append!(R, [getRRSTwo(g, n, delta) for _ in 1 : (θ - length(R) + 1)])

    ris_value, SEED = node_selection_pair(R, n, node_weight, pair_weight, k)

    T = time() - T
    return ris_value * n , SEED, length(R), T, R
end

function assign_pair_weights(G::Graph, Δs)
	sL = getSparseL(G)
    Inv = getW(G)
    sT = Matrix{Float64}(I, G.n, G.n) - ones(G.n,G.n)/G.n 

    p = Δs .* (Δs .* (Inv*sT*Inv))' # polarization
    d = Δs .* (Δs .* (Inv*sL*Inv))'
    ci = Δs .* (Δs .* (Inv*sL*sL*Inv))'
    idc = Δs .* (Δs .* (Inv))'

	return ci, d, p, idc
end

"""
    assign_node_weights(G::Graph, s, Δs)

    G: The input graph for computing different measures. 
    s: The vector of innate opinions
    Δs: The change of the innate opinion of a vertex once it gets influenced. 

    This function is supposed to create different node weights depending on different measures.
    The measures include Internal Conflict Index, Disagreement Index, Polarization Index, and Controversy index + Polarization Index. 
"""
function assign_node_weights(G::Graph, s, Δs; eps = 1e-6)
    IpL = getSparseIpL(G)
	sL = getSparseL(G)
	f = approxchol_sddm(IpL, tol=0.1*eps)

	z = f(s)

	# calculate aC_I(G)
	aci = 2 * Δs .* f(sL * (sL * z))

	# calculate aD(G)
	ad = 2 * Δs .* f(sL * z)

	# calculate aP(G)
	ap = 2 * Δs .* (f(z) .- (sum(s) / G.n))

	# calculate aI_dc(G)
	aidc = 2 * Δs .* z

    aidc_upper = 2 * Δs .* z +  Δs .* Δs

	return aci, ad, ap, aidc, aidc_upper 
end

"""
    delta_s_marketing(s, epsilon)

    How Δs is calculated under marketing setting
"""
function delta_s_marketing(s, epsilon)
    return [1 > (s[i] + epsilon) ? epsilon : 1 - s[i] for i in 1 : length(s)]
end 

"""
    delta_s_backfire(s, epsilon, beta)

    How Δs is calculated under back fire setting
"""
function delta_s_backfire(s, epsilon, beta)
    delta_s = []
    for v in s
        if v >= beta
            if v + epsilon > 1
                push!(delta_s, 1 - v)
            else
                push!(delta_s, epsilon)
            end
        else
            if v - epsilon < 0
                push!(delta_s, -v)
            else 
                push!(delta_s, -epsilon)
            end
        end
    end
    return delta_s
end 

