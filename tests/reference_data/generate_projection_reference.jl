# Run with an existing Effort.jl environment:
# julia --project=/absolute/path/to/Effort.jl generate_projection_reference.jl
# Existing fixtures are never overwritten.
using Effort, Printf, SHA

k = collect(range(0.01, 1.0; length=100))
ko = collect(range(0.03, 0.8; length=31))
signals = (sin.(5k), 0.5cos.(7k), 0.15sin.(5k).*cos.(7k))
matrices = ntuple(i -> hcat(signals[i], cos.((i+1).*k) .+ 0.3signals[i], i.*exp.(-k)), 3)
qpar = [1.02, 0.97, 1.04]
qperp = [0.98, 1.03, 0.96]
source = joinpath(dirname(pathof(Effort)), "projection.jl")
for (name, method) in (("cubic", Effort.Cubic()), ("akima", Effort.Akima()))
    destination = joinpath(@__DIR__, "projection_$(name).txt")
    isfile(destination) && error("Refusing to overwrite $destination")
    # Independent scalar calls also define the per-column AP reference.
    outputs = [Effort.apply_AP(k, ko, (x[:, c] for x in matrices)...,
                              qpar[c], qperp[c]; method=method) for c in 1:3]
    open(destination, "w") do io
        println(io, "# Effort.jl ", pkgversion(Effort), "; Julia ", VERSION)
        println(io, "# projection.jl sha256: ", bytes2hex(sha256(read(source))))
        println(io, "# k=range(0.01,1.0,length=100), ko=range(0.03,0.8,length=31)")
        println(io, "# qpar=[1.02,0.97,1.04]; qperp=[0.98,1.03,0.96]; n_GL_points=8")
        println(io, "# signals and independent columns: see generate_projection_reference.jl")
        println(io, "# columns: ko, c1_P0,c1_P2,c1_P4,c2_P0,c2_P2,c2_P4,c3_P0,c3_P2,c3_P4")
        for i in eachindex(ko)
            @printf(io, "%.17e", ko[i])
            for c in 1:3, ell in 1:3
                @printf(io, " %.17e", outputs[c][ell][i])
            end
            println(io)
        end
    end
    println("Saved ", destination)
end
