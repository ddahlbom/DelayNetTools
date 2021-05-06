
################################################################################
# Fundamental "neural" data types
################################################################################
struct Spike
	n::UInt64
	t::Float64
end
Base.show(io::IO, s::Spike) = print(io, "(Neuron $(s.n): $(s.t) s)")

struct Synapse
	strength::Float64
	source::UInt64
	dest::UInt64
	delay::UInt16 	# in number of samples
end
Base.show(io::IO, syn::Synapse) = print(io, "(Source: $(syn.source), Destination: $(syn.dest), Delay: $(syn.delay), Strength: $(syn.strength))")

abstract type Neuron end


################################################################################
# Izhikevich neuron
################################################################################
struct SimpleNeuron <: Neuron
	a::Float64
	b::Float64
	c::Float64
	d::Float64
end

struct SimpleNeuronType{T} end
SimpleNeuronType(s::Symbol) = SimpleNeuronType{s}()
SimpleNeuronType(s::AbstractString) = SimpleNeuronType{Symbol(s)}()

SimpleNeuron(s::Symbol) = SimpleNeuron(SimpleNeuronType{Symbol(s)}())
SimpleNeuron(s::AbstractString) = SimpleNeuron(SimpleNeuronType(s))

# Regular Spiking (Exc)
SimpleNeuron(::SimpleNeuronType{:rs}) = SimpleNeuron(0.02, 0.2, -65.0, 8.0)
SimpleNeuron(::SimpleNeuronType{:RS}) = SimpleNeuron(0.02, 0.2, -65.0, 8.0)
# Intrinsically Bursting (Exc)
SimpleNeuron(::SimpleNeuronType{:ib}) = SimpleNeuron(0.02, 0.2, -55.0, 4.0)
SimpleNeuron(::SimpleNeuronType{:IB}) = SimpleNeuron(0.02, 0.2, -55.0, 4.0)
# Chattering (Exc)
SimpleNeuron(::SimpleNeuronType{:ch}) = SimpleNeuron(0.02, 0.2, -50.0, 2.0)
SimpleNeuron(::SimpleNeuronType{:CH}) = SimpleNeuron(0.02, 0.2, -50.0, 2.0)
# Fast Spiking (Inh)
SimpleNeuron(::SimpleNeuronType{:fs}) = SimpleNeuron(0.1, 0.2, -65.0, 2.0)
SimpleNeuron(::SimpleNeuronType{:FS}) = SimpleNeuron(0.1, 0.2, -65.0, 2.0)
# Low-Threshold Spiking (Inh)
SimpleNeuron(::SimpleNeuronType{:lts}) = SimpleNeuron(0.02, 0.25, -65.0, 2.0)
SimpleNeuron(::SimpleNeuronType{:LTS}) = SimpleNeuron(0.02, 0.25, -65.0, 2.0)
# Thalamo-Cortical 
SimpleNeuron(::SimpleNeuronType{:tc}) = SimpleNeuron(0.02, 0.25, -65.0, 0.05)
SimpleNeuron(::SimpleNeuronType{:TC}) = SimpleNeuron(0.02, 0.25, -65.0, 0.05)
# Resonator 
SimpleNeuron(::SimpleNeuronType{:rz}) = SimpleNeuron(0.1, 0.26, -65.0, 2.0)
SimpleNeuron(::SimpleNeuronType{:RZ}) = SimpleNeuron(0.1, 0.26, -65.0, 2.0)

# Random Excitatory
SimpleNeuron(::SimpleNeuronType{:exc}) = begin
	c = rand()^2.0
	SimpleNeuron(0.02, 0.2, -65.0+15.0c, 8.0-6.0c)
end
SimpleNeuron(::SimpleNeuronType{:EXC}) = SimpleNeuron(SimpleNeuronType(:exc))
# Random Inhibitory
SimpleNeuron(::SimpleNeuronType{:inh}) = begin
	r = rand()
	SimpleNeuron(0.02+0.08r, 0.25-0.05r, -65.0, 2.0)
end
SimpleNeuron(::SimpleNeuronType{:INH}) = SimpleNeuron(SimpleNeuronType(:inh))


################################################################################
# Gerstner neuron
################################################################################
mutable struct SRMNeuron <: Neuron
	u::Float64
	u_max::Float64
	u_rest::Float64
	Θ::Float64
	τ_refr::Float64
	τ_memb::Float64
	t_lastevent::Float64
	t_lastspike::Float64
	ϵ_last::Float64
end

SRMNeuron() = SRMNeuron(-65.0, 8.0, -65.0, -50.0, 7.0/1000.0, 3.0/1000.0,
						0.0, 0.0, 0.0)

SRMNeuron(τ_membrane) = SRMNeuron(-65.0, 8.0, -65.0, -50.0, 7.0/1000.0, τ_membrane, 
								  0.0, 0.0, 0.0)


######################################################################
# Delay lines 
######################################################################

mutable struct DelayLine
	numstored::Int64
	delaylen::Int64
	counts::Vector{Int64}
end

DelayLine(delaylen::Int64, maxevents::Int64) =
	DelayLine(0, delaylen, zeros(Int64, maxevents))

DelayLine(delaylen::Int64) = DelayLine(0, delaylen, zeros(Int64, 64))

DelayLine(numstored::Int64, delaylen::Int64, maxevents::Int64) = 
	DelayLine(numstored, delaylen, zeros(Int64, maxevents))


################################################################################
# Types for managing experiments
################################################################################
struct ModelParams
	fs::Float64

	# Graph (revise later) -- now specified directly, but still used here
	num_neurons::UInt64
	p_contact::Float64
	p_exc::Float64
	maxdelay::Float64 # ms

	# STDP
	synmax::Float64
	tau_pre::Float64
	tau_post::Float64
	a_pre::Float64
	a_post::Float64
end
ModelParams() = ModelParams(2000.0, 1000, 0.1, 0.8, 20.0, 10.0,
							0.02, 0.02, 1.20, 1.0)

struct TrialParams
	dur::Float64
	lambda::Float64
	randspikesize::Float64
	randinput::Bool
	inhibition::Bool
    stdp::Bool
	inputmode::UInt64
	multiinputmode::UInt64
	inputweight::Float64
	recordstart::Float64
	recordstop::Float64
	lambdainput::Float64
	inputrefractorytime::Float64
end

TrialParams() = TrialParams(1.0, 3.0, 20.0, 1, 1, 1, 1, 1, 20.0, 0.0, 1.0, 0.5, 0.0)

function Base.show(io::IO, tp::TrialParams)
	println(io, "\tInput mode: $(tp.inputmode == 1 ? "periodic" : "poissonian")")
	println(io, "\tInput timing density (if random): $(tp.lambdainput) (1/s)")
	println(io, "\tInput size: $(tp.inputweight)")
	println(io, "\tDuration: $(tp.dur) (s)")	
	println(io, "\tRandom input: $(tp.randinput)")
	println(io, "\tRand spike density (λ): $(tp.lambda) (1/s)")
	println(io, "\tRandom spike size: $(tp.randspikesize)")
	println(io, "\tInhibition: $(tp.inhibition)")
	println(io, "\tSTDP: $(tp.stdp)")
	println(io, "\tRecord start: $(tp.recordstart)")
	println(io, "\tRecord stop: $(tp.recordstop)")
end

struct Experiment
	name::String
	mp::ModelParams
	tp::TrialParams
	input::Array{Spike,1}
end

struct Model
	params::ModelParams
	delgraph::Array{Int64,2}
	syngraph::Array{Float64,2}
	neurons::Array{SimpleNeuron,1}
end

struct ExperimentOutput
	modelname::AbstractString
	trialname::AbstractString
	tp::TrialParams
	input::Array{Array{Spike,1},1}
	output::Array{Spike,1}
	inputtimes::Array{Float64,1}
	inputids::Array{Int64,1}
	synapses::Array{Synapse,1}
end


function Base.show(io::IO, exp::ExperimentOutput)
	println(IO, "Model Name: $(exp.modelname)")
	println(IO, "Trial Name: $(exp.trialname)")
	println(IO, "Trial Parameters")
	Base.show(io, exp.tp)
end

struct ExperimentType{T} end


ExperimentType(s::AbstractString) = ExperimentType{Symbol(s)}()

################################################################################
# Delay learning types
################################################################################
struct CategoricalSpiker
	dls::Array{DelayLine,1}
	neuron::SRMNeuron
	weight::Float64
end

CategoricalSpiker(inputs::T, possibledelays::T, weight;
				  τ_membrane=3.0/1000.0,
				  maxevents=64) where T <: Union{Array{Int64,1}, UnitRange{Int64}} = begin
	dls = [DelayLine(rand(possibledelays), maxevents) for _ ∈ inputs]
	neuron = SRMNeuron(τ_membrane)
	CategoricalSpiker(dls, neuron, weight)
end

