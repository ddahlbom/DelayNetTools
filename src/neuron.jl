################################################################################
# Izhikevich 
################################################################################
@inline f1(v, u, input) = (0.04*v + 5.0)*v + 140.0 - u + input
@inline f2(v, u, a, b)  = a*(b*v - u)

@inline function izhupdate_rk4(v, u, input, h, p::SimpleNeuron)
    half_h = 0.5h
    sixth_h = 0.16666666666666666h

    K1 = f1(v, u, 0.0)
    L1 = f2(v, u, p.a, p.b)
    K2 = f1(v + half_h*K1, u + half_h*L1, 0.0)
    L2 = f2(v + half_h*K1, u + half_h*L1, p.a, p.b)
    K3 = f1(v + half_h*K2, u + half_h*L2, 0.0)
    L3 = f2(v + half_h*K2, u + half_h*L2, p.a, p.b)
    K4 = f1(v + h*K3, u + h*L3, 0.0)
    L4 = f2(v + h*K3, u + h*L3, p.a, p.b)

    (v + sixth_h * (K1 + 2K2 + 2K3 + K4) + h*input,
     u + sixth_h * (L1 + 2L2 + 2L3 + L4))
end

function izhupdate_euler(v, u, input, h, p::SimpleNeuron)
    (v + h*((0.04v + 5.0)*v + 140.0 - u + input),
     u + h*(p.a*(p.b*v - u)))
end


################################################################################
# Gerstner
################################################################################
function resetsrmneuron(n::SRMNeuron)
	n.u = -65.0
	n.u_max = 8.0
	n.u_rest = -65.0
	n.Θ = -50.0
	n.τ_refr = 7.0/1000.0
	n.τ_memb = 3.0/1000.0
	n.t_lastevent = 0.0
	n.t_lastspike = 0.0
	n.ϵ_last = 0.0
	return n
end

function srmneuronupdate(n::SRMNeuron, t, bumpsize)
	spiked = false
	ϵ_new = n.ϵ_last*exp((n.t_lastevent - t)/n.τ_memb) + bumpsize*n.u_max
	n.t_lastevent = t
	n.u = n.u_rest + ϵ_new 
	if n.u > n.Θ && t - n.t_lastspike > n.τ_refr
		spiked = true
		n.t_lastspike = t
		n.u = n.u_rest
		n.ϵ_last = 0.0
	else
		n.ϵ_last = ϵ_new
	end
	return spiked
end


################################################################################
# Delay Lines
################################################################################
function clearbuffer(dl::DelayLine)
	dl.numstored = 0
	dl.counts .*= 0
end


function pushevent(dl::DelayLine)
	if dl.numstored < length(dl.counts)
		dl.numstored += 1
		dl.counts[dl.numstored] = dl.delaylen;
	else
		println("buffer overflow")
	end
end

function advance(dl::DelayLine)
	out = 0
	if dl.counts[1] == 1 
		dl.counts[1] = 0
		out = 1
		for i ∈ 2:dl.numstored
			dl.counts[i-1] = dl.counts[i]
		end
		dl.numstored -= 1
	end
	for i ∈ 1:dl.numstored
		dl.counts[i] -= 1
	end
	return out
end


