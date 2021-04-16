
exponentialsample(λ) = -log(rand()) / λ


function denseperiodic(f, dur, fs; magnitude=20.0)
	N_pat = Int(round(dur * fs))
	dn_pat = Int(round(fs/f))
	train = zeros(N_pat)
	idx = 1
	while idx <= N_pat
		train[idx] = magnitude 
		idx += dn_pat
	end
	train
end


function sparseperiodic(f, dur)
	numspikes = Int(floor(dur*f))
	dt = 1.0/f
	[dt*(k-1) for k ∈ 1:numspikes]
end


function sparsepoisson(λ, dur)
	ts = Array{Float64, 1}(undef, 0)
	t = 0.0
	while t < dur
		push!(ts, t)
		t += exponentialsample(λ)
	end
	ts
end


function sparserefractorypoisson(λ, dur, deaddur)
	ts = Array{Float64, 1}(undef, 0)
	t = 0.0
	while t < dur
		push!(ts, t)
		sample = exponentialsample(λ)
		while sample < deaddur
			sample = exponentialsample(λ)
		end
		t += sample
	end
	ts
end


function densepoisson(λ, dur, fs; magnitude=20.0)
	ts = sparsepoisson(λ, dur)
	output = zeros(Int64(round(dur*fs))+1)
	for t ∈ ts
		output[Int64(round(t*fs))+1] = magnitude 
	end
	output
end


function channeldup(times::Array{T,1}, channels) where T <: Number
	numspikes = length(times)
	numchan = length(channels)
	spikes = Array{Spike,1}(undef,numspikes*numchan)
	count = 1
	for i ∈ 1:numspikes
		for channel ∈ channels 
			spikes[count] = Spike(channel, times[i])
			count += 1
		end
	end
	spikes
end


function channelscatter(times::Array{T,1}, available_channels) where T <: Number
	n = length(times)
	spikes = Array{Spike, 1}(undef, n)
	if length(available_channels) > n
		channels = shuffle(available_channels)[1:n]	
		for i ∈ 1:n
			spikes[i] = Spike(channels[i], times[i])
		end
	else
		for i ∈ 1:n
			spikes[i] = Spike(rand(available_channels), times[i])
		end
	end
	spikes
end


function stochasticblock(times::Array{T,1}, p, channels) where T <: Number
	n = length(times)
	spikes = Array{Spike,1}(undef,0)
	for time ∈ times
		for channel ∈ channels
			if rand() < p
				push!(spikes, Spike(channel, time))
			end
		end
	end
	spikes
end


function genrandinput(λ, dur, nchannels)
	times = sparsepoisson(λ, dur)
	channelscatter(times, 1:nchannels)
end


function gendupinput(λ, dur, channels)
	times = sparsepoisson(λ, dur)
	channeldup(times, channels)
end
