function test(classifier::Array{CategoricalSpiker,1}, inputs, fs;
			  mindelay = 0.001,
			  maxdelay = 0.020,
			  weight=0.5,
			  maxduration=0.150,
			  verbose=false)
	dt = 1.0/fs
	spikers = classifier
	numcategories = length(spikers)
	numchannels = length(spikers[1].dls)

	results = zeros(Int64, length(inputs)) 

	for (i,input) ∈ enumerate(inputs)
		unspikedidcs = collect(1:numcategories)
		spiketimes = maxdelay .* ones(Float64, numcategories) .+ 0.1
		numcontributors = zeros(Int64, numcategories)
		contributoridcs = [zeros(Int64, numchannels) for _ ∈ 1:numcategories]
		for t ∈ 0.0:dt:maxduration
			# advance buffers and update neurons
			numcontributors[unspikedidcs] .*= 0
			for spikeridx ∈ unspikedidcs 
				spiker = spikers[spikeridx]
				# Advance buffers
				for (k,dl) ∈ enumerate(spiker.dls)
					inputspike = advance(dl)
					if inputspike != 0
						numcontributors[spikeridx] += 1
						contributoridcs[spikeridx][numcontributors[spikeridx]] = k
					end
				end
				# Update neuron (if any inputs)
				spiked = false
				if numcontributors[spikeridx] > 0
					spiked = srmneuronupdate(spiker.neuron, t,
										spiker.weight*numcontributors[spikeridx])
				end
				# Remove neuron from processing if fired
				if spiked
					spiketimes[spikeridx] = t
					filter!(x -> x != spikeridx, unspikedidcs)
				end
			end
			# push new input into buffers 
			inputspikes = filter(s -> t <= s.t < t+dt, input)
			for spikeridx ∈ unspikedidcs
				spiker = spikers[spikeridx]
				for spike ∈ inputspikes
					pushevent(spiker.dls[spike.n])
				end
			end
		end
		firingorder = sortperm(spiketimes)
		results[i] = firingorder[1]
		
		# Reset classifier state
		for spiker ∈ spikers
			resetsrmneuron(spiker.neuron)
			for dl ∈ spiker.dls
				clearbuffer(dl)
			end
		end
	end

	return results
end

function trainclassifier(inputs, labels, numcategories, numchannels, fs;
						 weight=0.5,
						 μ = 0.005,
						 mindelay = 0.001,
						 maxdelay = 0.020,
						 maxduration=0.150,
						 numtrainingcycles=1,
						 τ_membrane=3.0/1000.0,
						 verbose=false)
	dt = 1.0/fs
	channels = 1:numchannels
	categories = 1:numcategories
	d_min = Int(round(mindelay*fs))
	d_max = Int(round(maxdelay*fs))
	possibledelays = d_min:d_max
	spikers = [CategoricalSpiker(1:numchannels, possibledelays, weight; 
								   	maxevents = Int(round(d_max)),
									τ_membrane = τ_membrane)
			   for _ ∈ 1:numcategories]


	for i ∈ 1:numtrainingcycles
	println("Training cycle $i of $numtrainingcycles")
	numcorrect = 0
	numties = 0
	numwrong = 0
	numtested = 0
	for (label,input) ∈ zip(labels,inputs)
		unspikedidcs = collect(1:numcategories)
		spiketimes = maxdelay .* ones(Float64, numcategories) .+ 0.1
		numcontributors = zeros(Int64, numcategories)
		contributoridcs = [zeros(Int64, numchannels) for _ ∈ 1:numcategories]
		for t ∈ 0.0:dt:maxduration
			# advance buffers and update neurons
			numcontributors[unspikedidcs] .*= 0
			for spikeridx ∈ unspikedidcs 
				spiker = spikers[spikeridx]
				for (k,dl) ∈ enumerate(spiker.dls)
					inputspike = advance(dl)
					if inputspike != 0
						numcontributors[spikeridx] += 1
						contributoridcs[spikeridx][numcontributors[spikeridx]] = k
					end
				end
				spiked = false
				if numcontributors[spikeridx] > 0
					spiked = srmneuronupdate(spiker.neuron, t,
										spiker.weight*numcontributors[spikeridx])
				end
				if spiked
					verbose && println("$spikeridx spiked at $t (correct: $label)")
					spiketimes[spikeridx] = t
					filter!(x -> x != spikeridx, unspikedidcs)
				end
			end
			
			# push events
			inputspikes = filter(s -> t <= s.t < t+dt, input)
			for spikeridx ∈ unspikedidcs
				spiker = spikers[spikeridx]
				for spike ∈ inputspikes
					pushevent(spiker.dls[spike.n])
				end
			end
		end
		verbose && println("----------")

		# Stochastic update of delay times
		firingorder = sortperm(spiketimes)
		if spiketimes[firingorder[1]] < spiketimes[firingorder[2]] && firingorder[1] == label
			if !(spiketimes[firingorder[2]] -
				 spiketimes[label] > μ)
				# Shorten random delay on correct-label spiker
				if numcontributors[label] > 0
					#println("Yay, contributors!")
					didx = rand(contributoridcs[label][1:numcontributors[label]])
				else
					didx = rand(1:numchannels)
				end
				if spikers[label].dls[didx].delaylen > d_min
					spikers[label].dls[didx].delaylen -= 1
				end
				# Lengthen random delay on random incorrect-label spiker
				while (sidx = rand(categories)) == label end
				if numcontributors[sidx] > 0
					#println("Yay, contributors!")
					didx = rand(contributoridcs[sidx][1:numcontributors[sidx]])
				else
					didx = rand(1:numchannels)
				end
				if spikers[sidx].dls[didx].delaylen < d_max
					spikers[sidx].dls[didx].delaylen += 1
				end
			end
			numcorrect += 1
		else
			# Shorten random delay on correct-label spiker
			if numcontributors[label] > 0
				#println("Yay, contributors!")
				didx = rand(contributoridcs[label][1:numcontributors[label]])
			else
				didx = rand(channels)
			end
			if spikers[label].dls[didx].delaylen > d_min
				spikers[label].dls[didx].delaylen -= 1
			end
			# Lengthen delay on random incorrect-label spiker that spikes before
			# desired category
			#prespikeridces = filter(x -> x <= label, firingorder)
			#prespikeridces = firingorder[1:label-1] 
			prespikeridcs = findall(x -> x <= spiketimes[label],
								   spiketimes)
			filter!(x -> x != label, prespikeridcs)
			sidx = rand(prespikeridcs)
			if numcontributors[sidx] > 0
				#println("Yay, contributors!")
				didx = rand(contributoridcs[sidx][1:numcontributors[sidx]])
			else
				didx = rand(1:numchannels)
			end
			if spikers[sidx].dls[didx].delaylen < d_max
				spikers[sidx].dls[didx].delaylen += 1
			end
			#spikers[idx].dls[rand(channels)].delaylen += 1
			if spiketimes[firingorder[1]] == spiketimes[firingorder[2]]
				numties += 1
			else
				numwrong += 1
			end
		end
		numtested += 1
		# Reset neurons between inputs
		for spiker ∈ spikers
			resetsrmneuron(spiker.neuron)
			for dl ∈ spiker.dls
				clearbuffer(dl)
			end
		end
	end
	println("Correct: $(numcorrect/numtested)")
	println("Wrong: $(numwrong/numtested)")
	println("Ties: $(numties/numtested)")
	println()
	end
	
	return spikers
end



function trainmulticlassifier(inputs, labels, numcategories,
                              numchannels, numneurons, fs;
                              weight=0.5,
                              μ = 0.005,
                              mindelay = 0.001,
                              maxdelay = 0.020,
                              maxduration=0.150,
                              numtrainingcycles=1,
                              τ_membrane=3.0/1000.0,
                              verbose=false)
	dt = 1.0/fs
	channels = 1:numchannels
	categories = 1:numcategories
	d_min = Int(round(mindelay*fs))
	d_max = Int(round(maxdelay*fs))
	possibledelays = d_min:d_max
	# spikers = [[CategoricalSpiker(1:numchannels, possibledelays, weight; 
	# 							   	maxevents = Int(round(d_max)),
    #                                 τ_membrane = τ_membrane) for _ ∈ 1:numneurons]
    #             for _ ∈ 1:numcategories]
	# spikers = [[CategoricalSpiker(1:numchannels, possibledelays, weight; 
	# 							   	maxevents = Int(round(d_max)),
    #                                 τ_membrane = τ_membrane) for _ ∈ 1:numneurons]
    #             for _ ∈ 1:numcategories]
    spikers = Array{CategoricalSpiker,2}(undef,(numcategories,numneurons)) 
    for r ∈ 1:numcategories
        for c ∈ 1:numneurons
            spikers[r,c] = CategoricalSpiker(1:numchannels, possibledelays, weight; 
                                             maxevents = Int(round(d_max)),
                                             τ_membrane = τ_membrane)
        end
    end


	for i ∈ 1:numtrainingcycles
        println("Training cycle $i of $numtrainingcycles")
        numcorrect = 0
        numties = 0
        numwrong = 0
        numtested = 0
        for (label,input) ∈ zip(labels,inputs)
            unspikedcats = [collect(1:numcategories) for _ ∈ 1:numneurons]
            spiketimes = ones(Float64, numcategories, numneurons)
            numcontributors = zeros(Int64, numcategories, numneurons)
		    # contributoridcs = [zeros(Int64, numchannels) for _ ∈ 1:numcategories]
            contributoridcs = zeros(Int64, numchannels, numcategories, numneurons)

            # Run input through classifier neurons
            for t ∈ 0.0:dt:maxduration
                for n ∈ 1:numneurons
                    # advance buffers and update neurons
                    numcontributors[unspikedcats[n],n] .= 0     # <--- here's the problem
                    for spikercat ∈ unspikedcats[n]
                        spiker = spikers[spikercat,n]
                        for (k,dl) ∈ enumerate(spiker.dls)
                            inputspike = advance(dl)
                            if inputspike != 0
                                numcontributors[spikercat,n] += 1
						        contributoridcs[numcontributors[spikercat,n], spikercat, n] = k
                            end
                        end
                        spiked = false
                        if numcontributors[spikercat,n] > 0
                            spiked = srmneuronupdate(spiker.neuron, t,
                                                     spiker.weight*numcontributors[spikercat,n])
                        end
                        if spiked
                            verbose && println("$spikercat ($(n)) spiked at $t (correct: $label)")
                            spiketimes[spikercat,n] = t
                            filter!(x -> x != spikercat, unspikedcats[n])
                        end
                    end
                    # push events
                    inputspikes = filter(s -> t <= s.t < t+dt, input)
                    for spikercat ∈ unspikedcats[n]
                        spiker = spikers[spikercat,n]
                        for spike ∈ inputspikes
                            pushevent(spiker.dls[spike.n])
                        end
                    end
                end
            end
            verbose && println("----------")
                
		    # Stochastic update of delay times
            incatorders = zeros(Int64, size(spiketimes))
            for k ∈ 1:numcategories
                incatorders[k,:] = sortperm(spiketimes[k,:])
            end
            besttimes = [spiketimes[c,incatorders[c,1]] for c ∈ 1:numcategories]
            crosscatorders = sortperm(besttimes)
            firsttospike = findfirst(i->i==1, crosscatorders)
            secondtospike = findfirst(i->i==2, crosscatorders)

            if firsttospike != label
                # If wrong category spiked first, shorten the correct-category 
                # spiker that spiked with the shortest latency with the shortest
                # latency & lengthen the wrong-category spiker with the shortest
                # latency
                numwrong += 1

                # shorten shortest latency right-category spiker
                if numcontributors[label,incatorders[label,1]] > 0
                    didx = rand(contributoridcs[1:numcontributors[label,incatorders[label,1]],label,incatorders[label,1]])
 				else
 					didx = rand(1:numchannels)
 				end
                if spikers[label,incatorders[label,1]].dls[didx].delaylen > d_min
                    spikers[label,incatorders[label,1]].dls[didx].delaylen -= 1
 				end

                # lengthen shortest latency wrong-category spiker
                if numcontributors[firsttospike,incatorders[firsttospike,1]] > 0
                    didx = rand(contributoridcs[1:numcontributors[firsttospike,incatorders[firsttospike,1]],firsttospike,incatorders[firsttospike,1]])
                else
                    didx = rand(1:numchannels)
                end
                if spikers[firsttospike,incatorders[firsttospike,1]].dls[didx].delaylen < d_max
                    spikers[firsttospike,incatorders[firsttospike,1]].dls[didx].delaylen += 1
                end
            else
                # Otherwise, check if margin is large enough. If not, do same as
                # above, otherwise do nothing.
                if besttimes[1] == besttimes[2]
                    numties += 1
                else
                    numcorrect += 1
                end
                if besttimes[2] - besttimes[1] < μ
                    if numcontributors[label,incatorders[label,1]] > 0
                        didx = rand(contributoridcs[1:numcontributors[label,incatorders[label,1]],label,incatorders[label,1]])
                    else
                        didx = rand(1:numchannels)
                    end
                    if spikers[label,incatorders[label,1]].dls[didx].delaylen > d_min
                        spikers[label,incatorders[label,1]].dls[didx].delaylen -= 1
                    end

                    # lengthen shortest latency wrong-category spiker
                    if numcontributors[firsttospike,incatorders[firsttospike,1]] > 0
                        didx = rand(contributoridcs[1:numcontributors[firsttospike,incatorders[firsttospike,1]],firsttospike,incatorders[firsttospike,1]])
                    else
                        didx = rand(1:numchannels)
                    end
                    if spikers[firsttospike,incatorders[firsttospike,1]].dls[didx].delaylen < d_max
                        spikers[firsttospike,incatorders[firsttospike,1]].dls[didx].delaylen += 1
                    end
                end
            end

     		# Reset neurons between inputs
            for spiker ∈ spikers
                resetsrmneuron(spiker.neuron)
                for dl ∈ spiker.dls
                    clearbuffer(dl)
                end
            end
     		numtested += 1
        end
        correctrate = numcorrect/numtested
        println("Correct: $(correctrate)")
        println("Wrong: $(numwrong/numtested)")
        println("Ties: $(numties/numtested)")
        println()
        if correctrate == 1.0
            break
        end
    end

    return spikers
end


function testmulti(spikers::Array{CategoricalSpiker,2}, inputs, fs;
                              weight=0.5,
                              maxduration=0.150,
                              verbose=false)
	dt = 1.0/fs
    numcategories, numneurons = size(spikers)
	numchannels = length(spikers[1,1].dls)

	results = zeros(Int64, length(inputs)) 

    for (i,input) ∈ enumerate(inputs)
        unspikedcats = [collect(1:numcategories) for _ ∈ 1:numneurons]
        spiketimes = ones(Float64, numcategories, numneurons)
        numcontributors = zeros(Int64, numcategories, numneurons)
        # contributoridcs = [zeros(Int64, numchannels) for _ ∈ 1:numcategories]
        contributoridcs = zeros(Int64, numchannels, numcategories, numneurons)

        # Run input through classifier neurons
        for t ∈ 0.0:dt:maxduration
            for n ∈ 1:numneurons
                # advance buffers and update neurons
                numcontributors[unspikedcats[n],n] .= 0
                for spikercat ∈ unspikedcats[n]
                    spiker = spikers[spikercat,n]
                    for (k,dl) ∈ enumerate(spiker.dls)
                        inputspike = advance(dl)
                        if inputspike != 0
                            numcontributors[spikercat,n] += 1
                            contributoridcs[numcontributors[spikercat,n], spikercat, n] = k
                        end
                    end
                    spiked = false
                    if numcontributors[spikercat,n] > 0
                        spiked = srmneuronupdate(spiker.neuron, t,
                                                 spiker.weight*numcontributors[spikercat,n])
                    end
                    if spiked
                        verbose && println("$spikercat ($(n)) spiked at $t (correct: $label)")
                        spiketimes[spikercat,n] = t
                        filter!(x -> x != spikercat, unspikedcats[n])
                    end
                end
                # push events
                inputspikes = filter(s -> t <= s.t < t+dt, input)
                for spikercat ∈ unspikedcats[n]
                    spiker = spikers[spikercat,n]
                    for spike ∈ inputspikes
                        pushevent(spiker.dls[spike.n])
                    end
                end
            end
        end
            
        # Find first firerer
        incatorders = zeros(Int64, size(spiketimes))
        for k ∈ 1:numcategories
            incatorders[k,:] = sortperm(spiketimes[k,:])
        end
        besttimes = [spiketimes[c,incatorders[c,1]] for c ∈ 1:numcategories]
        crosscatorders = sortperm(besttimes)
        results[i] = findfirst(x->x==1, crosscatorders)

        # Reset classifiers
        for spiker ∈ spikers
            resetsrmneuron(spiker.neuron)
            for dl ∈ spiker.dls
                clearbuffer(dl)
            end
        end
    end

    return results
end
