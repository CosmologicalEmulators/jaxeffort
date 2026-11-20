# Usage: julia --project=/absolute/path/to/Effort.jl generate_canonical_reference.jl \
#            /path/to/Effort-reactant_dev/src/builtins.jl /path/to/extracted/artifacts
# The artifacts must be those pinned by reactant_dev ea985a7 (see Artifacts.toml).
using Effort, Printf, SHA
length(ARGS) == 2 || error("Expected builtins.jl and extracted artifact directory")
include(abspath(ARGS[1]))
const source_hash = bytes2hex(sha256(read(ARGS[1])))

function save(name, data, description)
    path = joinpath(@__DIR__, name * ".txt")
    isfile(path) && error("Refusing to overwrite $path")
    open(path, "w") do io
        println(io, "# Julia ", VERSION, "; Effort ", pkgversion(Effort))
        println(io, "# Canonical source: Effort.jl reactant_dev ea985a7; builtins.jl sha256=", source_hash)
        println(io, "# ", description)
        for row in eachrow(data)
            println(io, join((@sprintf("%.17e", x) for x in row), " "))
        end
    end
    println("Saved ", path)
end

biases = [2.0, -0.7, 0.3, -0.2, 0.05, 0.02, -0.01, 0.8, 0.9, -0.4, 0.2]
for name in ("pybird_eftoflss", "velocileptors")
    isfile(joinpath(@__DIR__, "canonical_" * name * ".txt")) && continue
    data = hcat(BUILTIN_BIAS_COMBINATIONS[name](biases), BUILTIN_JAC_BIAS_COMBINATIONS[name](biases))
    save("canonical_" * name, data, "biases=$(biases); columns: coefficient, 11 Jacobian columns")
end
k = [0.01, 0.07, 0.15, 0.29]
names = ["$(model)_ell$(ell)" for model in ("pybird", "velocileptors") for ell in (0, 2, 4)]
if !isfile(joinpath(@__DIR__, "canonical_stochastic.txt"))
    save("canonical_stochastic", hcat(k, (BUILTIN_STOCHMODELS[n](k) for n in names)...),
         "columns: k; 3 columns each for " * join(names, ", "))
end

cosmo = [1.0, 3.044, 0.9649, 67.36, 0.02237, 0.12, 0.06, -1.0, 0.0]
for (model, folder, biasname) in (("pybird", "PyBirdmnuw0wacdm", "pybird_eftoflss"),
                                 ("lpt", "trained_effort_velocileptors_lpt_mnuw0wacdm", "velocileptors"),
                                 ("rept", "trained_effort_velocileptors_rept_mnuw0wacdm", "velocileptors"))
    isfile(joinpath(@__DIR__, "canonical_spectra_" * model * ".txt")) && continue
    outputs = []
    for ell in (0, 2, 4)
        path = joinpath(ARGS[2], folder, string(ell))
        # The old generic loader reads the corrected Julia files. Replace only
        # function fields with the upstream canonical functions for exact parity.
        old = Effort.load_multipole_emulator(path * "/")
        emu = Effort.PℓEmulator(P11=old.P11, Ploop=old.Ploop, Pct=old.Pct,
            BiasCombination=BUILTIN_BIAS_COMBINATIONS[biasname],
            JacobianBiasCombination=BUILTIN_JAC_BIAS_COMBINATIONS[biasname],
            StochModel=BUILTIN_STOCHMODELS["$(model == "pybird" ? "pybird" : "velocileptors")_ell$(ell)"])
        # The legacy loader evals Julia postprocessing files inside the loop.
        value, jac = Base.invokelatest(Effort.get_Pℓ_jacobian, cosmo, 0.8, biases, emu)
        push!(outputs, hcat(vec(emu.P11.kgrid), value, jac))
    end
    save("canonical_spectra_" * model, hcat(outputs...),
         "cosmology=$(cosmo); D=0.8; biases=$(biases); 13 columns per ell=0,2,4: k,P,J[1:11]")
end
