# using AuditoryFilters     # seems to break dependencies, find alternative

# include("/home/dahlbom/.julia/dev/DelNetExperiment/src/ExternalPeripheralModels.jl")
# include("/home/dahlbom/research/carfac/JCARFAC/src/JCARFAC.jl")
# EPM = ExternalPeripheralModels


################################################################################
# Interface to Zilany and Meddis Models
################################################################################

# function zilanymodel(sig::Array{Float64,1}, fs, db, f_lo, f_hi, numchannels;
#                      anf_num = (0, 10, 0), species="human", seed=1)
#     params = EPM.ZilanyParams(fs, db, anf_num, (f_lo, f_hi, numchannels),
#                               species, seed)
#     cfs_an, spikes = EPM.sigtospikedata(sig, params) |> EPM.spikedatatochannels
# end




################################################################################
# Tools for simple sub-cortical model
################################################################################

function inhomogenouspoisson(λs, fs; rt = 0.0, densityfactor = 5000.0)
	probs = (λs ./ fs) * densityfactor 
	ts = collect(0.0:1/fs:(length(probs)-1)/fs)
	(maximum(probs) > 1.0) && println("WARNING: Maximum ($(maximum(probs))) greater than 1.0")
	(minimum(probs) < 0.0) && println("WARNING: Minimum prob. less than 0.0")
	spiketimes = Array{Float64,1}(undef,0)
	lastfired = -1.0
	for (k,t) ∈ enumerate(ts)
		if rand() < probs[k]
			if t > lastfired + rt
				push!(spiketimes,t)
				lastfired = t
			end
		end
	end
	spiketimes
end


# function auditoryspikes(signal::Vector{Float64}, numchannels, nervesperchannel, fs;
# 						rt=0.0, densityfactor=1000.0, lowf=100.0)
# 	periout = simpleperiphery(signal, numchannels, fs; lowf=lowf)
# 	numsamples, numchannels = size(periout)
# 	n = 1
# 	out = Array{Spike,1}(undef,0)
# 	for chan ∈ numchannels:-1:1
# 		#k = numchannels-chan+1
# 		k = chan
# 		for nerve ∈ 1:nervesperchannel
# 			spiketimes = inhomogenouspoisson(periout[:,k], fs;
# 										 rt=rt, densityfactor=densityfactor)
# 			for time ∈ spiketimes 
# 				push!(out, Spike(n, time))
# 			end
# 			n += 1
# 		end
# 	end
# 	out
# end
# 
# function auditoryspikes(signal::Vector{Float64}, numchannels, nervesperchannel, 
# 						outputchannels, fs;
# 						rt=0.0, densityfactor=100.0, lowf=100.0)
# 	spikes = auditoryspikes(signal, numchannels, nervesperchannel, fs;
# 							rt=rt, densityfactor=densityfactor, lowf=lowf)
# 	@assert length(outputchannels) >= numchannels * nervesperchannel
# 	[Spike(outputchannels[s.n], s.t) for s ∈ spikes]
# end
# 
# function simpleperiphery(signal::Vector{Float64}, numchannels, fs; lowf=100.0)
# 	fb = make_erb_filterbank(fs, numchannels, lowf)
# 	fbout = filt(fb, signal)
# 	map(x -> x < 0.0 ? 0.0 : x, fbout)
# end

function plotperiphery(periout::Array{T,2}, fs) where T <: Number
	p = plot(; legend=:none)
	numsamples, numchannels = size(periout)
	ts = 0:1/fs:(numsamples-1)/fs
	for chan ∈ numchannels:-1:1
		plot!(p, ts, periout[:,chan] .+ 0.25*(numchannels - chan + 1))
	end
	p
end

function hc(f0, nh, dur, fs;
			norm=true,
			weights=[1.0 for k ∈ 1:nh],
			ϕs=[0.0 for k ∈ 1:nh])
	ts = 0.0:1/fs:dur-1.0/fs
	sig = zeros(length(ts))
	for (k,w) ∈ enumerate(weights)
		sig .+= w .* sin.(2π*f0*k .* ts .+ ϕs[k])
	end
	if norm
		sig ./= maximum(sig)
	end
	sig
end

