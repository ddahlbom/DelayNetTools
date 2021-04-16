
################################################################################
# Spike Rasters
################################################################################

function spikeraster(spikes::Array{Spike,1}; kwargs...)
    points = [Point2f0(s.t, s.n) for s ∈ spikes]
    scatter(points; kwargs...)
end


# spikeraster(spikes::Array{Spike,1}; kwargs...) = spikeraster(Scene(), spikes; kwargs...)


################################################################################
# Spatial Plots and Animation
################################################################################

function spatialplot(neurontypes, positions, colors::Dict{SimpleNeuronType,Any}; 
                     size = 5.0, scale=1.0,
                     kwargs...)
    positions = [p .* scale for p ∈ positions]
    scene = Scene()
    colors = [colors[neurontypes[i]] for i ∈ 1:length(neurontypes)]
    scatter!(scene, positions;
             markersize=size, color=colors, kwargs...)
    scene
end


function spatialplot(neurontypes, positions, colors::Dict{SimpleNeuronType,Any},
                     firedidcs::Array{Int,1}; 
                     firedcolor = :red,
                     firedalpha = 0.5,
                     firedsize = 7.5,
                     kwargs...)
    scene = spatialplot(neurontypes, positions, colors; kwargs...)
    if haskey(kwargs, :scale)
        positions = [p .* kwargs[:scale] for p ∈ positions]
    end
    scatter!(scene, [positions[i] for i ∈ firedidcs];
             color=(firedcolor, firedalpha), markersize=firedsize)
    scene
end

function spatialplot(neurontypes, positions, color::Dict{SimpleNeuronType,Any},
                     graph::Array{Int64,2}; kwargs...)
    scene = spatialplot(neurontypes, positions, color; kwargs...)
	n = size(graph)[1]
    for i ∈ 1:n 
        for j ∈ 1:n
			if graph[i,j] != 0
				lines!(scene,
                       [positions[i][1], positions[j][1]],
  					   [positions[i][2], positions[j][2]],
                       [positions[i][3], positions[j][3]];
                       color=(:black, 0.2))
			end
		end
	end
    return scene 
end


function spatialanim(neurontypes, positions, colors::Dict{SimpleNeuronType,Any},
                     times::Tuple{Float64,Float64},
                     dt::Float64,
                     spikes::Array{Spike,1};
                     name="spikes",
                     size=0.0100,
                     framerate=20,
                     firedsize=0.0150,
                     numfiringsteps=5,
                     alphas=(0.4,0.8),
                     kwargs...)
    # Set up base plot
    #scene = spatialplot(neurontypes, positions, colors; kwargs...)
    scene = Scene()
    colors = [colors[neurontypes[i]] for i ∈ 1:length(neurontypes)]
    scatter!(scene, positions;
             markersize=size, color=colors, kwargs...)
    # Add overlay of red dots
    firedcolors = [(:red, 0.0) for i ∈ 1:length(neurontypes)]
    firedsizes = [firedsize for _ ∈ 1:length(neurontypes)]
    scatter!(scene, positions;
             markersize=firedsizes, color=firedcolors, strokewidth=0)
    # Bookeeping for animation
    ts = times[1]:dt:times[2]-dt
    prioralphas = LinRange(alphas[2], alphas[1], numfiringsteps)
    priorsizes = LinRange(firedsize, size, numfiringsteps)
    #firedidcs = Array{Array{Int64,1},1}(undef, numfiringsteps)
    firedidcs = [zeros(Int64,0) for _ ∈ 1:numfiringsteps]
    # Start animation
    record(scene, name*".mp4", ts; framerate=framerate) do t
        for i ∈ numfiringsteps:-1:2
            firedidcs[i] = firedidcs[i-1]
        end
        firedcolors = [(:red, 0.0) for i ∈ 1:length(neurontypes)]
        firedsizes = [firedsize for _ ∈ 1:length(neurontypes)]
        firedidcs[1] = [Int64(s.n) for s ∈ filter(s -> t <= s.t < t+dt, spikes)]
        for (i,idcs) ∈ enumerate(firedidcs)
            for idx ∈ idcs
                fcolor = neurontypes[idx] == SimpleNeuronType(:exc) ? (:red) : (:blue)
                firedcolors[idx] = (fcolor,prioralphas[i])
                firedsizes[idx] = priorsizes[i]
            end
        end
        scene[end].color[] = firedcolors
        scene[end].markersize[] = firedsizes
        update!(scene)
    end
end


## Move to plotting functions
# function psthistograms(spikes::Array{Spike,1}, intimes, windur; offset=0.0)
# 	spikeblocks = pstspikes(spikes, intimes, windur; offset=offset)
# 	acs = map(sparseac, spikeblocks)
# 	# acs = [filter(x-> x >= 0, ac) for ac ∈ acs]
# 	hists = [histogram(ac) for ac ∈ acs]
# 	p = plot(hists..., reuse=false)
# end

