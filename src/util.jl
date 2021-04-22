function runexperiment(trialtype::String, args...; kwargs...)
	runexperiment(ExperimentType(trialtype), args...; kwargs...)	
end


function runexperiment(::ExperimentType{:new}, name::AbstractString,
					   mp::ModelParams, delgraph::Array{Int64, 2},
					   tp::TrialParams, input;
					   numprocs = 8, execloc="./")
	writemparams(mp, name)
	writetparams(tp, name)
	writeinput(input, name)
	writedelgraph(delgraph, name)
	run(`mpirun -np $(numprocs) $(execloc)runtrial-mpi 0 $(name) $(name)`)
	spikes = loadspikes(name)
	inputtimes, inputids = loadinputtimes(name)
	synapses = loadsynapses(name)
	ExperimentOutput(name, name, tp, input, spikes, inputtimes, inputids, synapses)
end

function runexperiment(::ExperimentType{:new}, name::AbstractString,
					   m::Model,
					   tp::TrialParams, input; weights=:none,
					   numprocs = 8, execloc="./")
    println("dispatched here.")
	writemparams(m.params, name)
	writedelgraph(m.delgraph, name)
	writesyngraph(m.syngraph, name)
	writeneurons(m.neurons, name)
	writetparams(tp, name)
    if weights == :none
	    writeinput(input, name)
    elseif typeof(weights) <: Number
        println("and, again, here")
        writeinput(input, name; weight=weights)
    else
        writeinput(input, weights, name)
    end
	run(`mpirun -np $(numprocs) $(execloc)runtrial-mpi 0 $(name) $(name)`)
	spikes = loadspikes(name)
	inputtimes, inputids = loadinputtimes(name)
	synapses = loadsynapses(name)
	ExperimentOutput(name, name, tp, input, spikes, inputtimes, inputids, synapses)
end

function runexperiment(::ExperimentType{:resume}, modelname::AbstractString,
					   trialname::AbstractString, tp::TrialParams, input;
					   numprocs = 8, execloc="./")
	writetparams(tp, trialname)
	writeinput(input, trialname)
	run(`mpirun -np $(numprocs) $(execloc)runtrial-mpi 1 $(modelname) $(trialname)`)
	spikes = loadspikes(trialname)
	inputtimes, inputids = loadinputtimes(trialname)
	synapses = loadsynapses(trialname)
	ExperimentOutput(modelname, trialname, tp, input, spikes, inputtimes, inputids, synapses)
end

function stageexperiment(name::AbstractString, m::Model, tp::TrialParams, input;
						 dest="./")
	writemparams(m.params, dest*name)
	writedelgraph(m.delgraph, dest*name)
	writesyngraph(m.syngraph, dest*name)
	writeneurons(m.neurons, dest*name)
	writetparams(tp, dest*name)
	writeinput(input, dest*name)
end

function pgsearch(name::AbstractString;
				  basesize=3, threshold=18.0, duration=0.100,
				  maxgroups=50000, mingrouplen=3,
				  numprocs=8, execloc="/home/dahlbom/research/delnet/")
	run(`mpirun -np $(numprocs) $(execloc)pgsearch-mpi $(name) $(basesize) $(threshold) $(duration) $(maxgroups)`)
	spikes = loadspikes(name*"pg")
	groups = []
	t1 = 0.0
	t2 = t1 + duration
	idx1 = 1
	idx2 = 0
	for k ∈ 1:length(spikes)
		if spikes[k].t > t2 
			idx2 = k-1
			if idx2-idx1+1 >= mingrouplen
				push!(groups, [Spike(s.n,s.t-spikes[idx1].t)
							   for s ∈ spikes[idx1:idx2]])
			end
			t1 += duration
			t2 += duration
			idx1 = idx2
		end
	end
	return groups
end


struct DenseGraph{T}
	contacts::Array{Int64,1}
	vals::Array{T,1}
	startidcs::Array{Int64,1}
	counts::Array{Int64,1}
end

function densegraph(g; offset=1)
	n = size(g)[1]
	numcontacts = zeros(Int64,n)
	startidcs = ones(Int64,n)

	# Set up memory
	for c ∈ 1:n
		for r ∈ 1:n
			if g[r,c] != 0
				numcontacts[r] += 1
			end
		end
	end
	for i ∈ 2:n
		startidcs[i] = startidcs[i-1] + numcontacts[i-1]
	end

	# Copy over contents into dense structure
	numgraphentries = sum(numcontacts)
	densegraph = zeros(Int64, numgraphentries) 
	graphvals = zeros(typeof(g[1,1]), numgraphentries)
	counts = zeros(Int64,n)
	for r ∈ 1:n
		for c ∈ 1:n
			if g[r,c] != 0
				densegraph[startidcs[r]+counts[r]] = c
				graphvals[startidcs[r]+counts[r]] = g[r,c]		
				counts[r] += 1
			end
		end
	end
	DenseGraph(densegraph, graphvals, startidcs, numcontacts)
end

function densifysynapses(ss, n; offset=1)
	numentries = length(ss)
	counts = zeros(Int64, n)
	startidcs = ones(Int64, n)
	contacts = zeros(Int64, numentries)
	vals = zeros(Float64, numentries)
	runningcounts = zeros(Int64, n)
	for s ∈ ss
		counts[s.source+1] += 1
	end
	for i ∈ 2:n
		startidcs[i] = startidcs[i-1] + counts[i-1]
	end

	for s ∈ ss
		contacts[startidcs[s.source+1] + runningcounts[s.source+1]] = s.dest+offset
		vals[startidcs[s.source+1] + runningcounts[s.source+1]] = s.strength
		runningcounts[s.source+1] += 1	
	end
	DenseGraph(contacts, vals, startidcs, counts)	
end
