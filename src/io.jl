

# --------------- Parameter Writing ---------------
function writemparams(m::ModelParams, trialname)
	open(trialname * "_mparams.txt", "w") do f write(f, "fs\t\t\t$(m.fs)\n")
		write(f, "num_neurons\t\t$(Float64(m.num_neurons))\n")
		write(f, "p_contact\t\t$(m.p_contact)\n")
		write(f, "p_exc\t\t\t$(m.p_exc)\n")
		write(f, "maxdelay\t\t$(m.maxdelay)\n")
		write(f, "synmax\t\t\t$(m.synmax)\n")
		write(f, "tau_pre\t\t\t$(m.tau_pre)\n")
		write(f, "tau_post\t\t$(m.tau_post)\n")
		write(f, "a_pre\t\t\t$(m.a_pre)\n")
		write(f, "a_post\t\t\t$(m.a_post)\n")
	end
end


function writetparams(t::TrialParams, trialname)
	open(trialname * "_tparams.txt", "w") do f
		write(f, "dur\t\t\t$(t.dur)\n")
		write(f, "lambda\t\t\t$(t.lambda)\n")
		write(f, "randspikesize\t\t$(t.randspikesize)\n")
		write(f, "randinput\t\t$(Float64(t.randinput))\n")
		write(f, "inhibition\t\t$(Float64(t.inhibition))\n")
		write(f, "inputmode\t\t$(t.inputmode)\n")
		write(f, "multiinputmode\t\t$(t.multiinputmode)\n")
		write(f, "inputweight\t\t$(t.inputweight)\n")
		write(f, "recordstart\t\t$(t.recordstart)\n")
		write(f, "recordstop\t\t$(t.recordstop)\n")
		write(f, "lambdainput\t\t$(t.lambdainput)\n")
		write(f, "inputrefractorytime\t$(t.inputrefractorytime)\n")
	end
end


# --------------- IO for delnet Communication ---------------
function read(s::IO, ::Type{Spike})
    n = read(s,Int64)
    t = read(s,Float64)
	Spike(n,t)
end

function readneurons(trialname)
    neurons = Nothing
	open(trialname*"_neurons.bin", "r") do f
		n = Base.read(f, Int64)
        neurons = Array{SimpleNeuron,1}(undef,n)
		for k ∈ 1:n
            a = Base.read(f, Float64) 
            b = Base.read(f, Float64) 
            c = Base.read(f, Float64) 
            d = Base.read(f, Float64) 
            neurons[k] = SimpleNeuron(a,b,c,d)
		end
	end
    neurons
end


function writeneurons(neurons::Array{SimpleNeuron,1}, trialname)
	open(trialname*"_neurons.bin", "w") do f
		write(f, length(neurons))
		for neuron ∈ neurons
			write(f, neuron.a)
			write(f, neuron.b)
			write(f, neuron.c)
			write(f, neuron.d)
		end
	end
end


function writeinput(input::Array{Spike,1}, trialname; weight=18.0)
	open(trialname * "_input.bin", "w") do f
		write(f,Int64(1)) 			# number of inputs (only 1 here)
		write(f,length(input)) 		# write an Int64 (long int)
        for spike ∈ input
			write(f,spike.n-1)
			write(f,spike.t)
		end
        for _ ∈ 1:length(input)
            write(f,weight)
        end
	end
end

function writeinput(input::Array{Spike,1}, weights::Array{Float64,1}, trialname)
	open(trialname * "_input.bin", "w") do f
		write(f,Int64(1)) 			# number of inputs (only 1 here)
		write(f,length(input)) 		# write an Int64 (long int)
        @assert length(weights) == length(input)
        for spike ∈ input
			write(f,spike.n-1)
			write(f,spike.t)
		end
        for weight ∈ weight
            write(f,weight)
        end
	end
end

function writeinput(inputs::Array{Array{Spike,1},1},
                    weights::Array{Array{Float64,1},1}, trialname)
	open(trialname * "_input.bin", "w") do f
		write(f,length(inputs)) 			# number of inputs
        for (i, input) ∈ enumerate(inputs)
			write(f,length(input)) 		# write an Int64 (long int)
            for spike ∈ input
				write(f,spike.n-1)
				write(f,spike.t)
			end
            for weight ∈ weights[i]
                write(f,weight)
            end
		end
	end
end

function writeinput(inputs::Array{Array{Spike,1},1}, trialname; weight=18.0)
	open(trialname * "_input.bin", "w") do f
		write(f,length(inputs)) 			# number of inputs
        for input ∈ inputs
			write(f,length(input)) 		# write an Int64 (long int)
            for spike ∈ input
				write(f,spike.n-1)
				write(f,spike.t)
			end
            for _ ∈ 1:length(input)
                write(f,weight)
            end
		end
	end
end


function loadinput(trialname)
	open(trialname * "_input.bin", "r") do f
		numinputs = Base.read(f,Int64)
		inputs = Array{Array{Spike,1},1}(undef,0)
		for _ ∈ 1:numinputs
			len = Base.read(f,Int64)
			input = Array{Spike, 1}(undef,len)
			#input = read(f, input)
			for k ∈ 1:len
				input[k] = Spike(Base.read(f, Int64) + 1, Base.read(f, Float64))
			end
			push!(inputs, input)
		end
		return length(inputs) == 1 ? inputs[1] : inputs 	# type stability here...
	end
end


function writedelgraph(graph::Array{Int64,2}, trialname)
	open(trialname * "_graph.bin", "w") do f
		n = size(graph)[1]
		graphraw = reshape(graph', (n*n,))
		write(f, Int64(n)) 	# write an Int64 (long int)
		write(f, graphraw) 			
	end
end

function loaddelgraph(trialname)
	open(trialname * "_graph.bin", "r") do f
		n = Base.read(f, Int64)
		graphraw = Array{UInt64, 1}(undef, n*n)
		read!(f, graphraw);
		graph = reshape(graphraw, (n, n))'
		return graph
	end
end


function writesyngraph(graph::Array{Float64,2}, trialname)
	open(trialname * "_syngraph.bin", "w") do f
		n = size(graph)[1]
		graphraw = reshape(graph', (n*n,))
		write(f, Int64(n)) 	# write an Int64 (long int)
		write(f, graphraw) 			
	end
end

function loadbinvector(filename; datatype=Float64)
	open(filename, "r") do f
		len = Base.read(f, UInt64)
		input = Array{datatype, 1}(undef, Int64(len))
		read!(f, input)
		return input
	end
end


function loadsynapses(trialname)
	filename = trialname*"_synapses.bin"
	open(filename, "r") do f
		len = Int64(Base.read(f, UInt64))
		strengths = Array{Float64,1}(undef, len)
		sources = Array{UInt64,1}(undef, len)
		dests = Array{UInt64,1}(undef, len)
		delays = Array{UInt16,1}(undef, len)
		read!(f, strengths)
		read!(f, sources)
		read!(f, dests)
		read!(f, delays)
		return [Synapse(strengths[i], sources[i], dests[i], delays[i]) for i ∈ 1:len]
	end

end


function loadspikesmat(trialname)
	open(trialname * "_spikes.txt", "r") do f
		lines = readlines(f)
		ts = [parse(Float64, split(line, "  ")[1]) for line ∈ lines]
		ns = [parse(Int64, split(line, "  ")[2]) for line ∈ lines]
		return [ts ns]
	end
end

function loadinputtimes(trialname)
	open(trialname * "_instarttimes.txt", "r") do f
		lines = readlines(f)
		ts = [parse(Float64, split(line, "  ")[1]) for line ∈ lines]
		ids = [parse(Int64, split(line, "  ")[2])+1 for line ∈ lines]
		return ts, ids
	end
end


function loadspikes(trialname::String)
	open(trialname * "_spikes.txt", "r") do f
		lines = readlines(f)
		n = length(lines)
		spikes = Array{Spike,1}(undef, n)
		for k ∈ 1:n 
			# Note "+1" for 1-based indexing in Julia
			spikes[k] = Spike( parse(Int64, split(lines[k], "  ")[2])+1,
							   parse(Float64, split(lines[k], "  ")[1]))
		end
		return spikes 
	end
end
