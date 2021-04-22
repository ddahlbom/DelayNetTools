module DelayNetTools

using Random, GLMakie, AbstractPlotting

include("types.jl")
export Spike, Synapse, Neuron, SimpleNeuronType, SimpleNeuron, SRMNeuron, DelayLine
export ModelParams, TrialParams, Experiment, Model, ExperimentOutput
export CategoricalSpiker

include("signalgen.jl")
export denseperiodic, sparseperiodic
export densepoisson, sparsepoisson, sparserefractorypoisson 
export channeldup, channelscatter, stochasticblock, genrandinput

include("victor.jl")

# include("auditory.jl") # needs auditory filters

include("neuron.jl")
export izhupdate_euler, izhupdate_rk4
export resetsrmneuron, srmneuronupdate
export clearbuffer, pushevent, advance  # revise so have !

include("netgen.jl")
export analyzeblock, delayhistvals, sampletargets, genpatch, gensyn 

include("io.jl")

include("util.jl")
export runexperiment, stageexperiment, pgsearch

include("analysis.jl")
export populationrate, populationpst, imagefromspikes
export sparseac, denseac, densehalfac, corranalysis
export pstspikes#, psthistograms

include("plotting.jl")
export spikeraster, spatialplot, spatialanim 

include("delayml.jl")
export trainclassifier, test, trainmulticlassifier, testmulti


end
