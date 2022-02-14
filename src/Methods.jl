include("Graph_class.jl")
include("Tools.jl")
include("Sampling.jl")


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
    ris_value_pre = 0

    for i in 1 : round(log2(n) - 1)
        y = n / (2^i)
        θᵢ = λ′ / y 
        if length(R) <= θᵢ
            append!(R, [getRRS(g, n, delta) for _ in 1 : (θᵢ - length(R) + 1)])
        end

        if (flag == 1) && (length(R) >= n*200) # add condition to avoid too much sampling
            break
        end

        #println("loop is ", i)
        #println("Time Before selection is ", time() - T)

        ris_value, _ = node_selection_linear_sum(R, n, node_weight, k)

        #println("Time After selection is ", time() - T)

        if i != 1
            if abs(ris_value - ris_value_pre) * n < 1  # avoid too small changes, which is then meaningless 
                break 
            else
                ris_value_pre = ris_value
            end
        end 

        if n*ris_value >= (1 + ϵ′)*y*χ
            LB = n * ris_value / (1 + ϵ′)
            break 
        end
    end

    θ = λ⁺ / LB 
    if (length(R) <= θ)
        if (θ <= n*200) || (flag == 0)
            append!(R, [getRRS(g, n, delta) for _ in 1 : (θ - length(R) + 1)])
        else
            #println("avoid much sampling")
            append!(R, [getRRS(g, n, delta) for _ in 1 : (n*200 - length(R) + 1)])
        end
    end

    #println("Time Before last calculation is ", time() - T)

    ris_value, SEED = node_selection_linear_sum(R, n, node_weight, k)

    #println("Time After last calculation is ", time() - T)

    T = time() - T
    return ris_value * n, SEED, length(R), T
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

