module JuliaLSODA

using LinearAlgebra

using Reexport: @reexport
@reexport using DiffEqBase

const G_NYH = Ref(0)
const G_LENYH = Ref(0)

# newly added static variables
const ML = Ref(0)
const MU = Ref(0)
const IMXER = Ref(0)
const MORD = [0, 12, 5]
const YP1 = Ref{Vector{Float64}}()
const YP2 = Ref{Vector{Float64}}()
const SM1 = [0., 0.5, 0.575, 0.55, 0.45, 0.35, 0.25, 0.2, 0.15, 0.1, 0.075, 0.05, 0.025]

# static variables for lsoda()
const CCMAX = Ref(0.0)
const EL0 = Ref(0.0)

const YH = Ref{Vector{Vector{Float64}}}()

end # module
