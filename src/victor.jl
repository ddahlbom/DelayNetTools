"""
	dspike(s1::Array{T,1}, s2::Array{T,1}, q) where T <: AbstractFloat

Victor-Purpura metric based on spike location. (Could optimize memory
usage -- only need two rows at a given time.)
"""
function dspike(s1::Array{T,1}, s2::Array{T,1}, q) where T <: AbstractFloat
	l1 = length(s1)
	l2 = length(s2)
	G = zeros(l1,l2)
	G[1,1:l2] .= collect(0:l2-1)
	G[1:l1,1] .= collect(0:l1-1)
	for i ∈ 2:l1
		for j ∈ 2:l2
			G[i,j] = min(G[i-1,j]+1, G[i,j-1]+1, G[i-1,j-1]+q*abs(s1[i]-s2[j]))
		end
	end
	G[l1,l2]
end


"""
	dinterval(s1::Array{T,1}, s2::Array{T,1}, q) where T <: AbstractFloat

Victor-Purpura metric based on spike interval
"""
function dinterval(s1::Array{T,1}, s2::Array{T,1}, q) where T <: AbstractFloat
	sort!(s1); sort!(s2)
	t0 = min(minimum(s1), minimum(s2)) - 0.001
	t1 = max(maximum(s1), maximum(s2)) + 0.001
	intervals1 = zeros(length(s1)+1)
	intervals2 = zeros(length(s2)+1)
	for i ∈ 2:length(s1)
		intervals1[i] = s1[i] - s1[i-1]
	end
	for i ∈ 2:length(s2)
		intervals2[i] = s2[i] - s2[i-1]
	end
	intervals1[1], intervals2[1] = s1[1]-t0, s2[1]-t0
	intervals1[end], intervals2[end] = t1-s1[end], t1-s2[end]
	l1 = length(intervals1)
	l2 = length(intervals2)
	G = zeros(l1,l2)
	G[1,1:l2] .= collect(0:l2-1)
	G[1:l1,1] .= collect(0:l1-1)
	for i ∈ 2:l1
		for j ∈ 2:l2
			G[i,j] = min(G[i-1,j]+1,
						 G[i,j-1]+1,
						 G[i-1,j-1]+q*abs(intervals1[i]-intervals2[j]) )
		end
	end
	G[l1,l2]
end


"""
	Family distances
"""

