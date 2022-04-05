```
MIT License

Copyright (c) 2021 Qi Bao

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```



include("Graph.jl")
include("Tools.jl")


using SparseArrays
using Laplacians
using Printf
using Statistics

function Exact(G, s) # Returns ci, d, p, idc
	T = time()
	L = getSparseL(G)
	W = getW(G)
	avgs = sum(s)/G.n
	s2 = zeros(G.n)
	for i = 1 : G.n
		s2[i] = s[i] - avgs
	end
	z = W * s
	z2 = W * s2
	# calculate C_I(G)
	lz = L * z
	ci = lz' * lz
	# calculate D(G)
	d = z2' * L * z2
	# calculate P(G)
	p = z2' * z2
	# calculate C(G)
	c = z' * z
	# calculate I_dc(G)
	idc = d + c
	# END CALCULATION
	T = time() - T
	return T, ci, d, p, idc
end

function Approx(G, s; eps = 1e-6) # Returns aci, ad, ap, aidc
	T = time()
	IpL = getSparseIpL(G)
	sL = getSparseL(G)
	f = approxchol_sddm(IpL, tol=0.1*eps)
	avgs = sum(s)/G.n
	s2 = zeros(G.n)
	for i = 1 : G.n
		s2[i] = s[i] - avgs
	end
	z = f(s)
	z2 = f(s2)

	# calculate aC_I(G)
	alz = sL * z
	aci = alz' * alz
	# calculate aD(G)
	ad = z2' * sL * z2
	# calculate aP(G)
	ap = z2' * z2
	# calculate aC(G)
	ac = z' * z
	# calculate aI_dc(G)
	aidc = ad + ac
	# END CALCULATION
	T = time() - T
	return T, aci, ad, ap, aidc
end


# experiment for smaller graphs
function doExp(G, s, lg)
	T, ci, d, p, idc = Exact(G, s)
	T2, aci, ad, ap, aidc = Approx(G, s)
	println(lg, "Exact  Time : ", T)
	println(lg, "Approx Time : ", T2)

	println(lg, "ci : ", ci)
	println(lg, "d : ", d)
	println(lg, "p : ", p)
	println(lg, "idc : ", idc)

	println(lg, "ERROR of ci : ", abs(aci-ci)/ci)
	println(lg, "ERROR of d : ", abs(ad-d)/d)
	println(lg, "ERROR of p  : ", abs(ap-p)/p)
	println(lg, "ERROR of idc : ", abs(aidc-idc)/idc)
	println(lg)
end

# experiment for the large graphs
function doLarge(G, s, lg)
	T2, aci, ad, ap, aidc = Approx(G, s)
	println(lg, "Approx Time : ", T2)

	println(lg, "aci : ", aci)
	println(lg, "ad : ", ad)
	println(lg, "ap : ", ap)
	println(lg, "aidc : ", aidc)

	println(lg)
end


# New Methods
function Approx_marketing(G, s_flag, p, epsilon, t=10)
	z_list, aci_list, ad_list, ap_list, aidc_list = [], [], [], [], [], []
	aci, ad, ap, aidc = 0, 0, 0, 0

	if s_flag == 0
		s = [0 for _ in 1:G.n]
		s = epsilon * p .+ s
		s[s .> 1] .= 1 # set elements of s1 > 1 back to 1

		_, aci, ad, ap, aidc = Approx(G, s)
		push!(z_list, sum(s))
		push!(aci_list, aci)
		push!(ad_list, ad)
		push!(ap_list, ap)
		push!(aidc_list, aidc)
	else
		for _ = 1:t
			if s_flag == 1
				s = Uniform(G.n)
			elseif s_flag == 2
				s = Exponential(G.n)
			elseif s_flag == 3
				s = powerLaw(G.n)
			end

			z_init = sum(s)
			_, aci_init, ad_init, ap_init, aidc_init = Approx(G, s)

			s = epsilon * p .+ s
			s[s .> 1] .= 1 # set elements of s1 > 1 back to 1

			_, aci, ad, ap, aidc = Approx(G, s)
			push!(z_list, (sum(s) - z_init) / z_init)
			push!(aci_list, (aci - aci_init) / aci_init)
			push!(ad_list, (ad - ad_init) / ad_init)
			push!(ap_list, (ap - ap_init) / ap_init)
			push!(aidc_list, (aidc - aidc_init) / aidc_init)
		end
	end
	z_sum = mean(z_list)
	z_std = std(z_list)
	aci = mean(aci_list)
	aci_std = std(aci_list)
	ad = mean(ad_list)
	ad_std = std(ad_list)
	ap = mean(ap_list)
	ap_std = std(ap_list)
	aidc = mean(aidc_list)
	aidc_std = std(aidc_list)

	return z_sum, aci, ad, ap, aidc, z_std, aci_std, ad_std, ap_std, aidc_std
end

function Approx_backfire(G, s_flag, p, epsilon, beta, t=10)
	z_list, aci_list, ad_list, ap_list, aidc_list = [], [], [], [], [], []
	aci, ad, ap, aidc = 0, 0, 0, 0	

	for _ = 1:t
		if s_flag == 0
			s = [0.5 for _ in 1:G.n]
		elseif s_flag == 1
			s = Uniform(G.n)
		elseif s_flag == 2
			s = Exponential(G.n)
		elseif s_flag == 3
			s = powerLaw(G.n)
		end

		z_init = sum(s)
		_, aci_init, ad_init, ap_init, aidc_init = Approx(G, s)


		s2 = deepcopy(s)
		for sn in 1:G.n
			if s[sn] >= beta
				s2[sn] += epsilon * p[sn] 
			else 
				s2[sn] -= epsilon * p[sn]
			end
		end
		
		s2[s2 .> 1] .= 1 # set elements of s1 > 1 back to 1
		s2[s2 .< 0] .= 0

		_, aci, ad, ap, aidc = Approx(G, s2)
		push!(z_list, (sum(s2) - z_init) / z_init)
		push!(aci_list, (aci - aci_init) / aci_init)
		push!(ad_list, (ad - ad_init) / ad_init)
		push!(ap_list, (ap - ap_init) / ap_init)
		push!(aidc_list, (aidc - aidc_init) / aidc_init)

		if s_flag == 0
			break
		end
	end
	
	z_sum = mean(z_list)
	z_std = std(z_list)
	aci = mean(aci_list)
	aci_std = std(aci_list)
	ad = mean(ad_list)
	ad_std = std(ad_list)
	ap = mean(ap_list)
	ap_std = std(ap_list)
	aidc = mean(aidc_list)
	aidc_std = std(aidc_list)

	return z_sum, aci, ad, ap, aidc, z_std, aci_std, ad_std, ap_std, aidc_std
end
