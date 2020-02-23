module JuliaLSODA

using LinearAlgebra

using Reexport: @reexport
@reexport using DiffEqBase

export LSODA

macro defconsts(expr, var)
    expr.head === :vect || error("Use it like `@defconsts [ccmax, el0, h, hmin, hmxi, hu, rc, tn] Ref(0.0)`")
    blk = quote end
    for name in expr.args
        push!(blk.args, :(const $name = $var))
    end
    return esc(blk)
end

# newly added static variables
const MORD = [0, 12, 5]
const SM1 = [0., 0.5, 0.575, 0.55, 0.45, 0.35, 0.25, 0.2, 0.15, 0.1, 0.075, 0.05, 0.025]

@defconsts [CCMAX, EL0, H, HMIN, HMXI, HU, RC, TN,
            TSW, PDNORM,
            CONIT, CRATE, HOLD, RMAX] Ref(0.0)

@defconsts [G_NYH, G_LENYH,
            ML, MU, IMXER,
            ILLIN, INIT, MXSTEP, MXHNIL, NHNIL, NTREP, NSLAST, NYH, IERPJ, IERSL,
            JCUR, JSTART, KFLAG, L, METH, MITER, MAXORD, MAXCOR, MSBP, MXNCF, N, NQ, NST,
            NFE, NJE, NQU,
            IXPR, JTYP, MUSED, MXORDN, MXORDS,
            IALTH, IPUP, LMAX, NSLP,
            PDEST, PDLAST, RATIO,
            ICOUNT, IRFLAG] Ref(0)

const EL = zeros(14)
const ELCO = zeros(13, 14)
const TESCO = zeros(13, 4)
const CM1 = zeros(13)
const CM2 = zeros(6)

@defconsts [YH, WM] Ref{Matrix{Float64}}()
@defconsts [YP1, YP2, EWT, SAVF, ACOR] Ref{Vector{Float64}}()

struct LSODA <: DiffEqBase.AbstractODEAlgorithm
end

function terminate!(istate::Ref{Int})
    # TODO
    if ILLIN[] == 5
        error("[lsoda] repeated occurrence of illegal input. run aborted.. apparent infinite loop\n")
    else
        ILLIN[] += 1
        istate[] = -3
    end
end

function terminate2!(y::Vector{Float64}, t::Ref{Float64})
    yp1 = YH[1]
    for i in 1:n
        y[i] = yp1[i]
    end
    t[] = TN
    ILLIN[] = 0
end

function successreturn!(y::Vector{Float64}, t::Ref{Float64}, itask::Int, ihit::Int, icrit::Float64, istate::Ref{Int})
    yp1 = YH[1]
    for i in 1:n
        y[i] = yp[i]
    end
    t = TN
    if itask == 4 || itask == 5
        ihit ? t = tcrit : nothing
    end
    istate = 2
    ILLIN = 0
end

function DiffEqBase.__solve(prob::ODEProblem{uType,tType,true}, ::LSODA;
                            itask::Int=1, istate::Ref{Int}=Ref(1), iopt::Bool=false,
                            tout=prob.tspan[end], reltol=1e-4, abstol=1e-6,
                            tcrit#=tstop=#=nothing) where {uType,tType}
    mxstp0 = 500
    mxhnl0 = 10
    1 <= istate <= 3 || error("[lsoda] illegal istate = $istate\n")
    @assert !(itask < 1 || itask > 5) "[lsoda] illegal itask = $itask\n"
    @assert !(INIT[] == 0 &&(istate == 2 || istate == 3)) "[lsoda] istate > 1 but lsoda not initialized"

    t = Ref(first(prob.tspan))

    if istate[] == 1
        INIT[] = 0
    end

    if istate[] == 1 || istate[] == 3
        NTREP[] = 0
        @assert neq > 0 "[lsoda] neq = $neq is less than 1\n"
        @assert !(istate[] == 3 && neq > n) "[lsoda] istate = 3 and neq increased"
    end

    if istate[] == 1
		INIT[] = 0
		if tout == t[]
			NTREP[] += 1
			ntrep < 5 && return
			@warn("[lsoda] repeated calls with istate = 1 and tout = t. run aborted.. apparent infinite loop\n")
			return
        end
    end
end

function stoda()
end

end # module
