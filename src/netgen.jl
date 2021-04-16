"""

units (x,y,z) are in millimeters; z assumed vertical (column length)
density in thousands per cubic millimeter
λ -- connection probability decay rate 
v -- conduction velocity (m/s (or mm/ms))
"""


function analyzeblock(delmat::Array{Int64,2}, fs)
	n = size(delmat)[1]
	conmat = map(x -> x != 0 ? 1 : 0, delmat)
	numdels = sum(conmat)
	println("Number of nodes:\t$(n)")
	println("Probability of contact:\t$(numdels/(n*n))")
	println("Average delay:\t$(1000.0*sum(delmat)/(fs*numdels)) ms")
end

function analyzeblock(delmat::Array{Int64,2}, fs, ps,
					  bounds1::Tuple{Float64,Float64,Float64},
					  bounds2::Tuple{Float64,Float64,Float64})
	n = size(delmat)[1]
	idcs = findall( x -> bounds1[1] <= x[1] <= bounds2[1] &&
				   		 bounds1[2] <= x[2] <= bounds2[2] &&
				   		 bounds1[3] <= x[3] <= bounds2[3],
						 ps)
	conmat = map(x -> x != 0 ? 1 : 0, delmat)
	numdels = sum(conmat[idcs,:])
	p_contact = numdels/(length(idcs)*length(idcs))
	avgdelay = 1000.0*sum(delmat[idcs,:])/(fs*numdels)
	println("Number of nodes:\t$(length(idcs))")
	println("Probability of contact:\t$p_contact")
	println("Average delay:\t$avgdelay ms")
end

function delayhistvals(delmat::Array{Int64,2}, fs)
	n = size(delmat)[1]
	numdels = sum( map( x -> x != 0 ? 1 : 0, delmat ) )
	vals = zeros(Float64, numdels)
	idx = 1
	for i ∈ 1:size(delmat)[1]
		for j ∈ 1:size(delmat)[2]
			if delmat[i,j] != 0
				vals[idx] = 1000.0 * delmat[i,j] / fs
				idx += 1
			end
		end
	end
	vals
end

function lenstobounds(lens::Array{Int64,1}) 
	bounds = Array{Int64,1}(undef,length(lens)+1)
	bounds[1] = 1
	for i ∈ 1:length(lens)
		bounds[i+1] = bounds[i] + lens[i]
	end
	bounds
end

dist(p1, p2) = sqrt(sum((p1 .- p2) .^ 2))
delaysamples(d,v,fs) = Int(round((d/v)*fs))


function sampletargets(center, positions, λ; factor=1.0) # λ = scale length
    n = length(positions)
    targets = zeros(Int64,0)
    for i ∈ 1:n
        d = dist(center, positions[i])
	    p_contact = factor*λ*exp( -d/λ )
        if rand() < p_contact
            push!(targets, i)
        end
    end
    targets
end


function sampletargets(center, positions, λ,
                       neurontypes, type::SimpleNeuronType;
                       factor=1.0)
    n = length(positions)
    targets = zeros(Int64,0)
    for i ∈ 1:n
        d = dist(center, positions[i])
	    p_contact = factor*λ*exp( -d/λ )
        if (rand() < p_contact) && (neurontypes[i] == type)
            push!(targets, i)
        end
    end
    targets
end



function genpatch(dims::Tuple{T,T,T},
				  types::Array{SimpleNeuronType,1},
				  ρs::Array{Float64,1},
				  λs::Array{Float64,1},
				  vs::Array{Float64,1},
				  fs;
				  probfactor = [1.0 for _ ∈ 1:length(types)],
				  numslices = 10,
				  maxlen=1.0,
				  verbose=false) where T <: Number
	@assert length(types) == length(ρs) == length(λs) == length(vs)
	slicedepth = dims[3]/numslices;
	positions = Array{Array{Tuple{T,T,T},1},1}(undef,0)
	neurons = Array{Array{SimpleNeuronType,1},1}(undef,0)
	lens = Array{Int64,1}(undef,length(types))
	for (i,ρ) ∈ enumerate(ρs)
		pointsinslice = dims[1]*dims[2]*slicedepth*ρs[i]
		ps = vcat([[(rand()*dims[1], rand()*dims[2], slicedepth*(rand()+k))
					for _ ∈ 1:pointsinslice] for k ∈ 0:numslices-1]...)
		push!(positions, ps)
		ns = [types[i] for _ ∈ 1:length(ps)]
		push!(neurons,ns)
		lens[i] = length(ns)
	end
	positions = vcat(positions...)
	neurontypes = vcat(neurons...)
	n = length(positions)
	delgraph = zeros(Int64,n,n)
	idx = 1
	bounds = lenstobounds(lens)
	for i ∈ 1:n
		if i == bounds[idx+1]
			idx += 1
		end
		for j ∈ 1:n
			d = dist(positions[i], positions[j])
			p_contact = probfactor[idx]*exp( -d/λs[idx] )
			if d < maxlen && rand() < p_contact
				if neurontypes[i] == SimpleNeuronType(:fs) ||
					neurontypes[i] == SimpleNeuronType(:inh)
					if neurontypes[i] == neurontypes[j]
						delgraph[i,j] = 0 #no inh->inh connections
					else
						#delgraph[i,j] = 1
						delgraph[i,j] = max(delaysamples(d,vs[idx],fs), 1)
					end
				else
					delgraph[i,j] = max(delaysamples(d,vs[idx],fs), 1)
					delgraph[i,j] = delgraph[i,j] == 1 ? 1 + Int(round(rand()*4.0)) : delgraph[i,j] 
				end
			end
		end
	end

	verbose && (analyzeblock(delgraph,fs))
	verbose && (println("Average distance (overall): ", avgdistance(positions)))
	verbose && (println("Average distance (connected): ", 
						avgdistance(positions, delgraph)))
	
	return positions, neurontypes, delgraph
end


function gensyn(delgraph, neurontypes, weights::Dict{SimpleNeuronType,Float64})
	@assert length(neurontypes) == size(delgraph)[1]
	syngraph = zeros(Float64, size(delgraph))
	for i ∈ 1:size(delgraph)[1]
		w = weights[neurontypes[i]]
		for j ∈ 1:size(delgraph)[2]
			syngraph[i,j] = delgraph[i,j] != 0 ? w : 0.0	
		end
	end
	syngraph
end



function avgdistance(ps)
	n = length(ps)
	cumulativedist = 0.0
	for i ∈ 1:n
		for j ∈ 1:n
			cumulativedist += sqrt( sum( (ps[i] .- ps[j]) .^ 2 ) )
		end
	end
	cumulativedist / (n * n)
end


function avgdistance(ps, graph)
	n = length(ps)
	cumulativedist = 0.0
	count = 0
	for i ∈ 1:n
		for j ∈ 1:n
			if graph[i,j] != 0
				cumulativedist += sqrt( sum( (ps[i] .- ps[j]) .^ 2 ) )
				count += 1
			end
		end
	end
	cumulativedist / count
end

function distances(ps, graph)
	n = length(ps)
	distances = Array{Float64,1}(undef,n*n)
	count = 1
	for i ∈ 1:n
		for j ∈ 1:n
			if graph[i,j] != 0
				distances[count] = sqrt( sum( (ps[i] .- ps[j]) .^ 2 ) )
				count += 1
			end
		end
	end
	resize!(distances, count-1)	
end


