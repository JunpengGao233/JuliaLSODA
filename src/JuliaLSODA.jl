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
    YP1[] = YH[][1]
    for i in 1:n
        y[i] = YP1[][i]
    end
    t[] = TN[]
    ILLIN[] = 0
end

function successreturn!(y::Vector{Float64}, t::Ref{Float64}, itask::Int, ihit::Int, icrit::Float64, istate::Ref{Int})
    YP1[] = YH[][1]
    for i in 1:n
        y[i] = YP1[i]
    end
    t[] = TN[]
    if itask == 4 || itask == 5
        ihit && (t = tcrit)
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
    1 <= istate[] <= 3 || (@warn("[lsoda] illegal istate = $istate\n") || terminate(istate[]))
    !(itask < 1 || itask > 5) || (@warn( "[lsoda] illegal itask = $itask\n") || terminate(istate[]))
    !(INIT[] == 0 &&(istate[] == 2 || istate[] == 3)) || (@warn("[lsoda] istate > 1 but lsoda not initialized") || terminate(istate[]))

    t = Ref(first(prob.tspan))    
    sizey = size(prob.u0)
    neq = sizey[0] * sizey[1]
    local y = prob.u0

    if istate[] == 1
		INIT[] = 0
		if tout == t[]
			NTREP[] += 1
			NTREP[] < 5 && return
			@warn("[lsoda] repeated calls with istate = 1 and tout = t. run aborted.. apparent infinite loop\n")
			return
        end
    end
    ###Block b ###
    
    if iopt == false 
        IXPR[] = 0
		MXSTEP[] = mxstp0
		MXHNIL = mxhnl0
		HMXI[] = 0.0
		HMIN[] = 0.0
		if (istate == 1) 
			local h0 = 0.0;
			MXORDN[] = MORD[1]
            MXORDS[] = MORD[2]
        end
    #TODO iopt == true
    end 
    (istate[] == 3) && (JSTART = -1)

    ### Block c ###
    if istate[] == 1
        TN[] = t[]
        TSW[] = t[]
        MAXORD[] = MXORDN[]
        if tcrit != nothing
            if (tcrit - tout) * (tout - t[]) < 0
                @warn("tcrit behind tout")
                terminate(istate[])
            end
            if (h0 != 0.0 && (t[] + h0 - tcrit) * h0 > 0.0)
                h0 = tcrit - t[]
            end
        end

        JSTART[] = 0;
        NHNIL[] = 0;
        NST[] = 0;
        NJE[] = 0;
        NSLAST[] = 0;
        HU[] = 0.;
        NQU[] = 0;
        MUSED[] = 0;
        MITER[] = 0;
        CCMAX[] = 0.3;
        MAXCOR[] = 3;
        MSBP[] = 20;
        MXNCF[] = 10;

        #prob.f(du,u,p,t)
        #(double t, double *y, double *ydot, void *data)

        prob.f(YH[2] + 1, y, 0 #=_data/p?=#, t[])
        NFE = 1
        YP1[] = YH[][1]
        for i in 1:n
            YP1[][i] = y[i]
        end
        NQ = 1
        H = 1
        ewset(rtol, atol, y)
        for i in 1:n
            if EWT[][i] <= 0
                @warn("[lsoda] EWT[$i] = $(EWT[][i])")
                terminate2(y, t[])
                return
            end
            EWT[][i] = 1 / EWT[][i]
        end

        #compute h0, the first step size, 
        if h0 == 0.0
            tdist = abs(tout - t[])
            w0 = max(abs(t[]), abs(tout))
            if tdist < 2 *eps() *w0
                @warn("[lsoda] tout too close to t to start integration")
                terminate(istate[])
            end
            local tol = rtol
            if tol <= 0.0
                for i in 1:n
                    if abs(y[i]) != 0
                        tol = max(rtol, atol/abs(y[i]))
                    end
                end
            end
            tol = max(tol, 100 * eps())
            tol = min(tol, 0.001)
            sum = vmnorm(n, YH[][2],EWT)
            sum = 1 / (tol * 100 * eps())
            h0 = 1 / sqrt(sum)
            h0 = min(h0, tdist)
            h0 = h0 * ((tout - t[] >= 0) ? 1 : -1)
        end
        rh = abs(h0) * HMXI[]
        rh > 1 && (h0 /= rh)
        h = h0
        YP1[] = YH[][2]
        for i in 1:n
            YP1[][i] *= h0
        end
    end

    ###Block d###

end

function vmnorm(n::Int, v::Ref{Float64}, w::Ref{Float64})
end

function ewset(rtol::Ref{Float64}, atol::Ref{Float64}, ycur::Ref{Float64})
    #TODO
end

function stoda()
    #TODO
end

end # module
