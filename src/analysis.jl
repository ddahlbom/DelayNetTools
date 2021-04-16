
function populationrate(spikes::Array{Spike,1}, windowdur, hopsize)
	ts = [s.t for s ∈ spikes]
	t_min = minimum(ts)
	t_max = maximum(ts)
	ts_bases = t_min:hopsize:t_max-windowdur
	rates = zeros(length(ts_bases))
	for (k, t_base) ∈ enumerate(ts_bases)
		# i1 = searchsortedfirst(ts, t_base)
		# i2 = searchsortedlast(ts, t_base+windowdur)
		# rates[k] = length(ts[i1:i2])/windowdur
		spikes = filter(t -> t_base <= t < t_base + windowdur, ts)	
		rates[k] = length(spikes)/windowdur
	end
	(ts_bases, rates)
end


gaussian(x, x0, σ) = 1.0/(σ*√(2*π)) * exp(-0.5 * (x - x0)^2 / σ^2 )


function imagefromspikes(spikes::Array{Spike,1}, t1, t2, n1, n2, fs=1000.0, σ=0.002)
	spikes = filter( s -> t1 <= s.t < t2, spikes )
	image = zeros(n2-n1+1, Int( round((t2-t1)*fs )) )
	if length(spikes) == 0
		return image
	end
	@assert minimum(s.n+1 for s ∈ spikes) >= n1 && maximum(s.n for s ∈ spikes) <= n2
	sigmalen = Int(round(3σ*fs))
	ts = range(-3σ, 3σ, length=2*sigmalen+1)
	bump = gaussian.(ts, 0.0, σ)
	bumplen = length(bump)
	halfbumplen = Int(floor(bumplen/2))
	centeridx = Int(ceil(bumplen/2))
	for s ∈ spikes
		n_idx = s.n - n1 + 1 # + 2 to also account for 0 indexing in C
		t_idx = Int(round((s.t - t1)*fs)) + 1
		t_lo = max(1, t_idx - halfbumplen)
		t_hi = min(size(image, 2), t_idx + halfbumplen)
		offset_lo = t_idx - t_lo
		offset_hi = t_hi - t_idx
		image[n_idx, t_lo:t_hi] .+= bump[centeridx-offset_lo:centeridx+offset_hi]
	end
	image
end


function populationpst(spikes::Array{Spike,1}, inputtimes, dur, n1, n2,
							fs, σ=0.002)
	cumimage = imagefromspikes(spikes, inputtimes[1], inputtimes[1]+dur,
							   n1, n2, fs, σ)
	count = 0
	for time ∈ inputtimes[2:end]
		image = imagefromspikes(spikes, time, time+dur, n1, n2, fs, σ)
		if size(image) == size(cumimage) # in case at end of data
			cumimage .+= image
			count += 1
		end
	end
	cumimage ./ maximum(cumimage)#count
end


function channelsort(spikes::Array{Spike,1}, n::Int)
	times = [Array{Float64,1}(undef,0) for _ ∈ 1:n]
	for s ∈ spikes
		push!(times[s.n], s.t)
	end
	times
end


function sparseac(spiketimes::Array{Float64,1})
	actimes = zeros(length(spiketimes)^2)
	c = 1
	for τ ∈ spiketimes
		for t ∈ spiketimes
			actimes[c] = t - τ		
			c += 1
		end
	end
	return actimes |> sort
end


function sparseac(spikes::Array{Spike,1})
	spiketimes = [s.t for s ∈ spikes]
	sparseac(spiketimes)
end


function densehalfac(ts::Array{Float64,1}, fs; norm=true)
	maxdelay = abs(maximum(ts) - minimum(ts))
	maxsamples = Int(round(maxdelay * fs))
	n = maxsamples
	ac = zeros(Float64, n+1)
	for i ∈ 1:length(ts)
		for j ∈ i:length(ts)
			ac[Int(round((ts[j]-ts[i])*fs))+1] += 1.0
		end
	end
	norm && ac ./= ac[1] 
	times = (0:n) .* (1/fs)
	return times, ac
end


function densehalfac(spikes::Array{Spike,1}, fs; norm=true)
	ts = [s.t for s ∈ spikes]
	densehalfac(ts, fs; norm=norm)
end


function denseac(spiketimes::Array{Float64,1}, fs; norm=true)
	maxdelay = abs(maximum(spiketimes) - minimum(spiketimes))
	maxsamples = Int(round(maxdelay * fs))
	ac = zeros(2*maxsamples+1)
	mid = maxsamples+1
	for τ ∈ spiketimes
		for t ∈ spiketimes
			ac[Int(round((t-τ)*fs)) + mid] += 1.0
		end
	end
	norm && ac ./= ac[mid]
	times = (-maxsamples:maxsamples) .* (1/fs)
	return times, ac
end


function denseac(spikes::Array{Spike,1}, fs; norm=true)
	spiketimes = [s.t for s ∈ spikes]
	denseac(spiketimes, fs)
end

function spiketodense(spikes::Array{Spike,1}, t1, t2, fs)
	ts = t1:(1.0/fs):t2
	densetrain = zeros(length(ts))
	for spike ∈ spikes
		densetrain[ Int(round((spike.t -t1)*fs)) + 1 ] = 1.0
	end
	return ts, densetrain
end


function corranalysis(spikes::Array{Spike,1}, input::Array{Spike,1},
					  intimes, windur, steps, fs)
	ts, in_ac = densehalfac(input, fs)
	corrvalues = [Array{Float64,1}(undef, length(steps)) for _ ∈ 1:length(intimes)]
	for (i, intime) ∈ enumerate(intimes)
		for (j, step) ∈ enumerate(steps)
			ss = filter(s -> intime+step <= s.t < intime+step+windur, spikes)
			ts, out_ac = densehalfac(ss, fs)	
			len = min(length(out_ac), length(in_ac)) # calculate outside loop
			# corrvalues[i][j] = sum(in_ac[1:len] .* out_ac[1:len])
			corrvalues[i][j] = cor([in_ac[1:len]'; out_ac[1:len]']')[1,2]
			println(corrvalues)
		end
	end
	return corrvalues
end


function pstspikes(spikes::Array{Spike,1}, intimes, windur; offset=0.0)
	spikeblocks = [ Array{Spike,1}(undef,0) for _ ∈ 1:length(intimes) ] 
	for (k, time) ∈ enumerate(intimes)
		spikeblocks[k] = filter(s -> time+offset <= s.t < time+offset+windur,
							    spikes)
	end
	spikeblocks
end


function averagedelaylength(synapses)
	delays = unique([s.delay for s ∈ synapses]) |> sort
	delay_counts = Dict{UInt16,Float64}()
	delay_vals = Dict{UInt16,Float64}()
	for d ∈ delays
		delay_counts[d] = 0.0
		delay_vals[d] = 0.0
	end

	for s ∈ synapses
		delay_counts[s.delay] += 1.0	
		delay_vals[s.delay] += s.strength
	end

	avg_vals = zeros(Float64, length(delays))
	for (k,d) ∈ enumerate(delays)
		avg_vals[k] = delay_vals[d] / delay_counts[d]
	end
	return Int64.(delays), avg_vals
end


## PG search prototyping -- now in C/MPI
# struct PolychronousGroup
# 	nodes::Array{Int64,1}
# 	target::Int64
# 	times::Array{Float64,1}
# 	weights::Array{Float64,1}
# end
# 
# 
# """
# 	Finds all polychronous groups associated with a set of "base" neurons.
# 	Loops should be fused/optimized (annoying but doable and necessary)
# """
# function testbasegroup(group, graph::DenseGraph{Int64}, weights::DenseGraph{Float64}; threshold=20.0)
# 	s = length(group)
# 	pgs = Array{PolychronousGroup,1}(undef,0)
# 	contacts = Array{Array{Int64,1},1}(undef,s)
# 	for i ∈ 1:s
# 		contacts[i] = graph.contacts[graph.startidcs[group[i]]:graph.startidcs[group[i]]+graph.counts[group[i]]-1]
# 	end
# 	commoncontacts = copy(contacts[1])
# 	for i ∈ 2:s
# 		commoncontacts = filter!(x -> x ∈ contacts[i], commoncontacts)
# 	end
# 	idcs = zeros(Int64,s)
# 	for commoncontact ∈ commoncontacts
# 		for i ∈ 1:s
# 			idcs[i] = findfirst(x -> x == commoncontact, contacts[i])
# 		end
# 		ws = [weights.vals[weights.startidcs[group[k]]+idcs[k]-1] for k ∈ 1:s]
# 		if sum(ws) >= threshold 
# 			delays  = [graph.vals[graph.startidcs[group[k]]+idcs[k]-1] for k ∈ 1:s]
# 			delays = maximum(delays) .- delays
# 			push!(pgs, PolychronousGroup(copy(group), commoncontact, delays, ws))
# 		end
# 	end
# 	return pgs
# end
# 
# 
# 
# """
# 
# Generate appropriate loop structure with a macro...
# """
# function findpgs(graph, synapses, s=3; threshold=18.0, maxpgs = 10000)
# 	n = size(graph)[1]
# 	group = zeros(Int64, s)
# 	graph = densegraph(graph) 	# remove after resolving graph in C code
# 	#weights = densegraph(weights)
# 	weights = densifysynapses(synapses, n)
# 	pgs = Array{PolychronousGroup,1}(undef,maxpgs)
# 	numpgs = 0
# 	if s == 1
# 		for i ∈ 1:n
# 			group[1] = i
# 			pgnew = testbasegroup(group, graph, weights; threshold=threshold)
# 			if length(pgnew) > 0
# 				push!(pgs, pgnew...)
# 			end
# 		end
# 	elseif s == 2
# 		for i ∈ 1:n
# 			for j ∈ i+1:n
# 				group[1], group[2] = i, j
# 				pgnew = pgingroup(group, graph, weights; threshold=threshold)
# 				if length(pgnew) > 0
# 					push!(pgs, pgnew...)
# 				end
# 			end
# 		end
# 	elseif s == 3
# 		for i ∈ 1:n
# 			println("Finished $i")
# 			for j ∈ i+1:n
# 				for k ∈ j+1:n
# 					group[1],group[2],group[3] = i,j,k
# 					pgnew = pgingroup(group, graph, weights; threshold=threshold)
# 					if length(pgnew) > 0
# 						for a ∈ 1:length(pgnew)
# 							numpgs += 1
# 							pgs[numpgs] = pgnew[a]
# 							if numpgs == maxpgs
# 								return pgs
# 							end
# 						end
# 					end
# 				end
# 			end
# 		end
# 	elseif s == 4
# 		for i ∈ 1:n
# 			for j ∈ i+1:n
# 				for k ∈ j+1:n
# 					for l ∈ k+1:n
# 						group[1],group[2],group[3],group[4] = i,j,k,l
# 						pgnew = pgingroup(group, graph, weights; threshold=threshold)
# 						if length(pgnew) > 0
# 							push!(pgs, pgnew...)
# 						end
# 					end
# 				end
# 			end
# 		end
# 	end
# 	resize!(pgs,numpgs)
# end
# 
# 
# """
# 	pginspikes(pg::Array{Spike,1},spikes::Array{Spike,1})
# 
# Spikes must be time-sorted
# """
# # export pginspikes
# function pginspikes(pg::Array{Spike,1},spikes::Array{Spike,1};
# 					timetol=0.001,hitthreshold=1.0)
# 	baseidcs = findall(x -> x.n == pg[1].n, spikes)
# 	groupinstances = Array{Array{Spike,1},1}(undef,0)
# 	for baseidx ∈ baseidcs
# 		numfound = 0
# 		t1 = spikes[baseidx].t
# 		t2 = t1 + pg[end].t + timetol
# 		k = baseidx
# 		while k < length(spikes) 
# 			k += 1
# 			if spikes[k].t > t2
# 				break
# 			end
# 		end
# 		endidx = k
# 		subspikes = @view spikes[baseidx:endidx]
# 		# subspikes = @view spikes[:]
# 		# println(length(subspikes))
# 		localinstance = Array{Spike,1}(undef,length(pg))
# 		localinstance[1] = spikes[baseidx]
# 		numhits = 1
# 		for pgspike ∈ pg[2:end]
# 			for spike ∈ subspikes	
# 				if pgspike.n == spike.n
# 					if abs((pgspike.t+t1)-spike.t) <= timetol 
# 						numhits += 1
# 						localinstance[numhits] = spike
# 					end
# 				end
# 			end
# 		end
# 		if numhits/length(pg) >= hitthreshold
# 			localinstance = resize!(localinstance,numhits)
# 			push!(groupinstances,localinstance)
# 		end
# 	end
# 
# 	return groupinstances
# end
# 
# """
# 	resultsintochunks(results::ExperimentOutput, dur=0.35)
# 
# Optimize this later.
# """
# function resultsintochunks(results::ExperimentOutput, dur=0.35)
# 		
# end
# 
# 
# """
# """
# # export showfoundgroup
# function showfoundgroup(foundgroup, spikes)
# 	t1 = foundgroup[1].t-0.1
# 	t2 = foundgroup[end].t+0.1
# 	p = spikeraster(filter(s->t1<=s.t<=t2, spikes); reuse=false)
# 	spikeraster(p, foundgroup; markersize=7, markeralpha=0.3,markercolor=:red)
# end
# 
# 
