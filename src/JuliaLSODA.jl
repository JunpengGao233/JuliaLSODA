module JuliaLSODA

using LinearAlgebra

using Reexport: @reexport
@reexport using DiffEqBase

using Printf

export LSODA

macro defconsts(expr, var)
    expr.head === :vect || error("Use it like `@defconsts [ccmax, el0, h, hmin, hmxi, hu, rc, tn] Ref(0.0)`")
    blk = quote end
    for name in expr.args
        push!(blk.args, :(const $name = $var))
    end
    return esc(blk)
end

#const DOPRINT = Ref(false)
# newly added static variables
const MORD = [12, 5]
const SM1 = [0.5, 0.575, 0.55, 0.45, 0.35, 0.25, 0.2, 0.15, 0.1, 0.075, 0.05, 0.025]

#=
@defconsts [CCMAX, EL0, H, HMIN, HMXI, HU, RC, TN,
            TSW, PDNORM,
            PDEST, PDLAST, RATIO,
            CONIT, CRATE, HOLD, RMAX] Ref(0.0)

@defconsts [ML, MU, IMXER,
            ILLIN, INIT, MXSTEP, MXHNIL, NHNIL, NTREP, NSLAST, NYH, IERPJ, IERSL,
            JCUR, JSTART, KFLAG, L, METH, MITER, MAXORD, MAXCOR, MSBP, MXNCF, N, NQ, NST,
            NFE, NJE, NQU,
            IXPR, JTYP, MUSED, MXORDN, MXORDS,
            IALTH, IPUP, LMAX, NSLP,
            ICOUNT, IRFLAG] Ref(0)


const LUresult = Ref{LU{Float64,Array{Float64,2}}}()
const EL = zeros(13)
const integrator.elco = zeros(12, 13)
const integrator.tesco = zeros(12, 3)
const integrator.cm1 = zeros(12)
const integrator.cm2 = zeros(5)

@defconsts [YH, WM] Ref{Matrix{Float64}}(zeros(1,2))
@defconsts [EWT, SAVF, ACOR] Ref{Vector{Float64}}(zeros(2))
=#

mutable struct JLSODAIntegrator
    ccmax::Float64 #in stru
    el0::Float64 #in stru
    h::Float64 #in stru
    hmin::Float64 #in stru 
    hmxi::Float64 #in stru
    hu::Float64 #in stru
    rc::Float64 #in stru
    tn::Float64 #in stru
    tsw::Float64 #in stru
    pdnorm::Float64 #in stru
    pdest::Float64 #in stru
    pdlast::Float64 #in stru 
    ratio::Float64 #in stru
    conit::Float64 #in stru
    crate::Float64 #in stru
    hold::Float64 #in stru
    rmax::Float64 #in stru

    #ml::Int64 #in stru
    #mu::Int64 #in stru
    imxer::Int64 #in stru
    illin::Int64 #in stru
    init::Int64 #in stru
    mxstep::Int64 #in stru
    mxhnil::Int64 #in stru
    nhnil::Int64 #in stru
    ntrep::Int64 #in stru
    nslast::Int64 #in stru
    nyh::Int64 #in stru
    ierpj::Int64 #in stru
    iersl::Int64 #in stru
    jcur::Int64 #in stru
    jstart::Int64 #in stru
    kflag::Int64 #in stru
    l::Int64 #in stru
    meth::Int64 #in stru
    miter::Int64 #in stru
    maxord::Int64 #in stru
    maxcor::Int64 #in stru
    msbp::Int64 #in stru
    mxncf::Int64 #in stru
    n::Int64 #in stru
    nq::Int64 #in stru
    nst::Int64 #in stru
    nfe::Int64 #in stru
    nje::Int64 #in stru
    nqu::Int64 #in stru
    ixpr::Int64 #in stru
    jtyp::Int64 #in stru
    mused::Int64 #in stru
    mxordn::Int64 #in stru
    mxords::Int64 #in stru
    ialth::Int64 #in stru
    ipup::Int64 #in stru
    lmax::Int64 #in stru
    nslp::Int64 #in stru
    icount::Int64 #in stru
    irflag::Int64 #in stru

    LUresult::LU{Float64,Array{Float64,2}}
    el::Vector{Float64}
    elco::Array{Float64,2}
    tesco::Array{Float64,2}
    cm1::Vector{Float64}
    cm2::Vector{Float64}
    yh::Matrix{Float64}
    wm::Matrix{Float64}
    ewt::Vector{Float64}
    savf::Vector{Float64}
    acor::Vector{Float64}
    yp1::Any
    yp2::Any
    function JLSODAIntegrator()
        obj = new()
        obj.el = zeros(13)
        obj.elco = zeros(12, 13)
        obj.tesco = zeros(12, 3)
        obj.cm1 = zeros(12)
        obj.cm2 = zeros(5)
        obj.yh = zeros(1,2)
        obj.wm = zeros(1,2)
        obj.ewt = zeros(2)
        obj.savf = zeros(2)
        obj.acor = zeros(2)
        return obj
    end
end

struct LSODA <: DiffEqBase.AbstractODEAlgorithm
end


function terminate!(istate::Ref{Int}, integrator::JLSODAIntegrator)
    # TODO
    if integrator.illin == 5
        error("[lsoda] repeated occurrence of illegal input. run aborted.. apparent infinite loop\n")
    else
        integrator.illin += 1
        istate[] = -3
    end
    return
end

function terminate2!(y::Vector{Float64}, t::Ref{Float64}, integrator::JLSODAIntegrator)
    integrator.yp1 = @view integrator.yh[1,:]
    for i in 1:integrator.n
        y[i] = integrator.yp1[i]
    end
    t[] = integrator.tn
    integrator.illin = 0
    return
end

function successreturn!(y::Vector{Float64}, t::Ref{Float64}, itask::Int, ihit::Int, tcrit::Float64, istate::Ref{Int}, integrator::JLSODAIntegrator)
    integrator.yp1 = @view integrator.yh[1,:]
    for i in 1:integrator.n
        y[i] = integrator.yp1[i]
    end
    t[] = integrator.tn
    if itask == 4 || itask == 5
        if bool(ihit)
            t = tcrit
        end
    end
    istate[] = 2
    integrator.illin = 0
    return
end

function DiffEqBase.__solve(prob::ODEProblem{uType,tType,true}, ::LSODA;
                            itask::Int=1, istate::Ref{Int}=Ref(1), iopt::Bool=false,
                            tout=prob.tspan[end], rtol=Ref(1e-4), atol=Ref(1e-6),
                            tcrit#=tstop=#=nothing) where {uType,tType}
    #DOPRINT[] = false
    integrator = JLSODAIntegrator()
    integrator.pdnorm = 0
    integrator.jtyp = 2
    mxstp0 = 500
    mxhnl0 = 10
    iflag = Ref(0)
    if istate[] < 1 || istate[] > 3
       @warn("[lsoda] illegal istate = $istate\n")
       terminate!(istate, integrator)
    end
    if itask < 1 || itask > 5
        @warn("[lsoda] illegal itask = $itask\n")
        terminate!(istate, integrator)
    end
    if (integrator.init == 0 && (istate[] == 2 || istate[] == 3))
        @warn("[lsoda] illegal itask = $itask\n")
        terminate!(istate, integrator)
    end
    if (integrator.init == 0 && (istate[] ==2 || istate[] == 3))
        @warn("[lsoda] istate > 1 but lsoda not initialized")
        terminate!(istate, integrator)
    end

    t = Ref(first(prob.tspan))
    neq = length(prob.u0)
    # NOTE!!! this function mutates `prob.u0`
    y = prob.u0
    integrator.n = neq
    if istate[] == 1
        integrator.illin = 0
        if tout == t[]
            integrator.ntrep += 1
            integrator.ntrep < 5 && return
            @warn("[lsoda] repeated calls with istate = 1 and tout = t. run aborted.. apparent infinite loop\n")
            return
        end
    end
    ###Block b ###
    if istate[] == 1 || istate[] == 3
        ntrep = 0
        if neq <= 0
            @warn("[lsoda] neq = %d is less than 1\n", neq)
            terminate!(istate, integrator)
            return
        end
        if istate[] == 3 && neq > integrator.n
            @warn("[lsoda] istate = 3 and neq increased\n")
            terminate!(istate, integrator)
            return
        end
        integrator.n = neq
    end

    if iopt == false
        integrator.ixpr = 0
        integrator.mxstep = mxstp0
        integrator.mxhnil = mxhnl0
        integrator.hmxi = 0.0
        integrator.hmin = 0.0
        if (istate[] == 1)
            h0 = 0.0
            integrator.mxordn = MORD[1]
            integrator.mxords = MORD[2]
        end
    #TODO iopt == true
    end
    (istate[] == 3) && (integrator.jstart = -1)

    ### Block c ###
    if istate[] == 1
        integrator.tn = t[]
        integrator.tsw = t[]
        integrator.maxord = integrator.mxordn
        if tcrit != nothing
            if (tcrit - tout) * (tout - t[]) < 0
                @warn("tcrit behind tout")
                terminate!(istate, integrator)
            end
            if (h0 != 0.0 && (t[] + h0 - tcrit) * h0 > 0.0)
                h0 = tcrit - t[]
            end
        end
        integrator.meth = 2
        integrator.nyh = integrator.n
        lenyh = 1 + max(integrator.mxordn, integrator.mxords)
        integrator.yh = zeros(lenyh, integrator.nyh)
        integrator.wm = zeros(integrator.nyh, integrator.nyh)
        integrator.ewt = zeros(integrator.nyh)
        integrator.savf = zeros(integrator.nyh)
        integrator.acor = zeros(integrator.nyh)

        integrator.jstart = 0
        integrator.nhnil = 0
        integrator.nst = 0
        integrator.nje = 0
        integrator.nslast = 0
        integrator.hu = 0.0
        integrator.nqu = 0
        integrator.mused = 0
        integrator.miter = 0
        integrator.ccmax = 0.3
        integrator.maxcor = 3
        integrator.msbp = 20
        integrator.mxncf = 10
        integrator.nfe = 1
        #prob.f(du,u,p,t)
        #(double t, double *y, double *ydot, void *data)
        @views prob.f(integrator.yh[2,:], y, prob.p, t[])
        integrator.yp1 = @view integrator.yh[1,:]
        for i in 1:integrator.n
            integrator.yp1[i] = y[i]
        end
        integrator.nq = 1
        integrator.h = 1.0
        ewset!(rtol, atol, y, integrator)
        for i in 1:integrator.n
            if integrator.ewt[i] <= 0
                @warn("[lsoda] EWT[$i] = $(integrator.ewt[i]) <= 0.0")
                terminate2!(y, t, integrator)
                return
            end
            integrator.ewt[i] = 1 / integrator.ewt[i]
        end

        #compute h0, the first step size,
        if h0 == 0.0
            tdist = abs(tout - t[])
            w0 = max(abs(t[]), abs(tout))
            if tdist < 2 * eps() *w0
                @warn("[lsoda] tout too close to t to start integration")
                terminate!(istate, integrator)
                return
            end
            tol = rtol[]
            if tol <= 0.0
                for i in 1:integrator.n
                    ayi = abs(y[i])
                    if ayi != 0
                        tol = max(tol[], atol[]/ayi)
                    end
                end
            end
            tol = max(tol, 100 * eps())
            tol = min(tol, 0.001)
            sum = vmnorm(integrator.n, integrator.yh[2,:], integrator.ewt)
            sum = 1 / (tol * w0 * w0) + tol * sum * sum
            h0 = 1 / sqrt(sum)
            h0 = min(h0, tdist)
            # h0 = h0 * ((tout - t[] >= 0) ? 1 : -1)
            h0 = copysign(h0, tout - t[])
        end
        rh = abs(h0) * integrator.hmxi
        rh > 1 && (h0 /= rh)
        integrator.h = h0
        integrator.yp1 = @view integrator.yh[2,:]
        for i in 1:integrator.n
            integrator.yp1[i] *= h0
        end
    end

    ###Block d###
    if (istate[] == 2 || istate[] == 3)
        integrator.nslast = integrator.nst
        if itask == 1
            if ((integrator.tn - tout) * integrator.h >= 0)
                intdy!(tout, 0, y, iflag, integrator)
                if iflag[] != 0
                    @warn("[lsoda] trouble from intdy, itask = $itask, tout = $tout\n")
                    terminate!(istate, integrator)
                    return
                end
                t[] = tout
                istate[] = 2
                integrator.illin = 0
                return
            end
        elseif itask == 3
            tp = integrator.tn - integrator.hu * (1 + 100 * eps())
            if ((tp - tout) * integrator.h > 0)
                @warn("[lsoda] itask = $itask and tout behind tcur - hu")
                terminate!(istate, integrator)
            end
            if ((integrator.tn - tout) * integrator.h >= 0)
                successreturn!(y, t, itask, ihit, tcrit, istate, integrator)
                return
            end
        elseif itask == 4
            if ((tn - tcrit) * integrator.h > 0)
                @warn("[lsoda] itask = 4 or 5 and tcrit behind tcur")
                terminate!(istate, integrator)
                return
            end
            if ((tcrit - tout) * integrator.h < 0)
                @warn("[lsoda] itask = 4 or 5 and tcrit behind tout")
                terminate!(istate, integrator)
                return
            end
            if ((integrator.tn - tout) * integrator.h >= 0)
                intdy!(tout, 0, y, iflag, integrator)
                if iflag != 0
                    @warn("[lsoda] trouble from intdy, itask = $itask, tout = $tout\n")
                    terminate!(istate, integrator)
                    return
                end
                t[] = tout
                istate[] = 2
                integrator.illin = 0
                return
            end
        elseif itask == 5
            if ((tn - tcrit) * integrator.h > 0)
                @warn("[lsoda] itask = 4 or 5 and tcrit behind tcur")
                terminate!(istate, integrator)
                return
            end
            hmx = abs(integrator.tn) + abs(integrator.h)
            ihit = abs(integrator.tn - tcrit) <= (100 * eps() *hmx)
            if Bool(ihit)
                t[] = tcrit
                successreturn!(y, t, itask, ihit, tcrit, istate, integrator)
                return
            end
            tnext = tn + integrator.h * (1 + 4 * eps())
            if (tnext - tcrit) * integrator.h > 0
                integrator.h = (tcrit - tn) * (1 - 4 * eps())
                if istate[] == 2
                    integrator.jstart = -2
                end
            end
        end
    end
    #Block e#
    while true
        if (istate[] != 1 || integrator.nst != 0)
            if ((integrator.nst - integrator.nslast) >= integrator.mxstep)
                @warn("[lsoda] $(integrator.mxstep) steps taken before reaching tout\n")
                istate[] = -1
                terminate2!(y, t, integrator)
                return
            end
            ewset!(rtol, atol, integrator.yh[1,:], integrator)
            for i = 1:integrator.n
                if integrator.ewt[i] <= 0
                    @warn("[lsoda] ewt[$i] = $(integrator.ewt[i]) <= 0.\n")
                    istate[] = -6
                    terminate2!(y, t, integrator)
                    return
                end
                integrator.ewt[i] = 1 / integrator.ewt[i]
            end
        end
        tolsf = eps() * vmnorm(integrator.n, integrator.yh[1,:], integrator.ewt)
        if tolsf > 0.01
            tolsf *= 200
            if integrator.nst == 0
                @warn("""
                lsoda -- at start of problem, too much accuracy
                requested for precision of machine,
                suggested scaling factor = $tolsf
                """)
                terminate!(istate, integrator, integrator)
                return
            end
            @warn("""lsoda -- at t = $(t[]), too much accuracy requested
                     for precision of machine, suggested
                     scaling factor = $tolsf""")
            istate[] = -2
            terminate2!(y, t, integrator)
            return
        end
        if ((integrator.tn + integrator.h) == integrator.tn)
            integrator.nhnil += 1
            if integrator.nhnil <= integrator.mxhnil
                @warn( """lsoda -- warning..internal t = $(integrator.tn) and h = $(integrator.h) are
                         such that in the machine, t + h = t on the next step
                         solver will continue anyway.""")
                if integrator.nhnil == integrator.mxhnil
                    @warn("""lsoda -- above warning has been issued $(integrator.nhnil) times,
                            it will not be issued again for this problem\n""")
                end
            end
        end

        stoda(neq, prob, integrator)
        #Block f#
        if integrator.kflag == 0
            integrator.init = 1
            if integrator.meth != integrator.mused
                integrator.tsw = integrator.tn
                integrator.maxord = integrator.mxordn
                if integrator.meth == 2
                    integrator.maxord = integrator.mxords
                end
                integrator.jstart = -1
                if Bool(integrator.ixpr)
                    integrator.meth == 2 && @warn("[lsoda] a switch to the stiff method has occurred")
                    integrator.meth == 1 && @warn("[lsoda] a switch to the nonstiff method has occurred")
                    @warn("at t = $(integrator.tn), tentative step size h = $(integrator.h), step nst = $(integrator.nst)\n")
                end
            end
            #itask == 1
            if itask == 1
                if (integrator.tn - tout) * integrator.h < 0
                    continue
                end
                intdy!(tout, 0, y, iflag, integrator)
                t[] = tout
                istate[] = 2
                integrator.illin = 0
                return
            end
            if itask == 2
                successreturn(y, t, itask, ihit, tcrit, istate, integrator)
                return
            end
            if itask == 3
                if (integrator.tn - tout) * integrator.h >= 0
                    successreturn(y, t, itask, ihit, tcrit, istate, integrator)
                    return
                end
                continue
            end
            if itask == 4
                if (integrator.tn - tout) * integrator.h >= 0
                    intdy!(tout, 0, y, iflag, integrator)
                    t[] = tout
                    istate[] =2
                    integrator.illin = 0
                    return
                else
                    hmx = abs(integrator.tn) + abs(integrator.h)
                    ihit = abs(tn - tcrit) <= 100 * eps() *hmx
                    if Bool(ihit)
                        successreturn(y, t, itask, ihit, tcrit, istate, integrator)
                        return
                    end
                    tnext = integrator.tn + integrator.h * (1 + 4 * eps())
                    if ((tnext - tcrit) * integrator.h <= 0)
                        continue
                    end
                    integrator.h = (tcrit - integrator.tn) * (1 - 4 * eps())
                    integrator.jstart = -2
                    continue
                end
            end
            if itask == 5
                hmx = abs(integrator.tn) + abs(integrator.h)
                ihit = abs(integrator.tn - tcrit) <= (100 * eps() * hmx)
                successreturn(y, t, itask, ihit, tcrit, istate, integrator)
                return
            end
        end

        if (integrator.kflag == -1 || integrator.kflag == -2)
            if integrator.kflag == -1
                @warn("""at t = $(integrator.tn), tentative step size h = $(integrator.h), step nst = $(integrator.nst)\n
                 error test failed repeatedly or with fabs(h) = hmin""")
                istate[] = -4
            end
            if integrator.kflag == -2
                @warn("""         corrector convergence failed repeatedly or
                         with fabs(h) = hmin""")
                istate[] = -5
            end
            big = 0
            integrator.imxer = 1
            for i in 1:integrator.n
                sizing = abs(integrator.acor[i]) * integrator.ewt[i]
                if big < sizing
                    big = sizing
                    integrator.imxer = 1
                end
            end
            terminate2!(y, t, integrator)
            return
        end
    end
    return
end

function stoda(neq::Int, prob, integrator::JLSODAIntegrator)
    y = prob.u0
    integrator.kflag = 0
    told = Ref(integrator.tn)
    corflag = Ref(0)
    ncf = Ref(0)
    delp = Ref(0.0)
    del = Ref(0.0)
    m = Ref(0)
    pdh = Ref(0.0)
    rh = Ref(0.0)
    orderflag = Ref(0)
    integrator.ierpj = 0
    integrator.iersl = 0
    integrator.jcur = 0
    if integrator.jstart == 0
        integrator.lmax = integrator.maxord + 1
        integrator.nq = 1
        integrator.l = 2
        integrator.ialth = 2
        integrator.rmax = 10000.0
        integrator.rc = 0
        integrator.el0 = 1.0
        integrator.crate = 0.7
        integrator.hold = integrator.h
        integrator.nslp = 0
        integrator.ipup = integrator.miter
        integrator.icount = 20
        integrator.irflag = 0
        integrator.pdest = 0.0
        integrator.pdlast = 0.0
        integrator.ratio = 5.0
        cfode(2, integrator)
        for i in 1:5
            integrator.cm2[i] = integrator.tesco[i, 2] * integrator.elco[i, i + 1]
        end
        cfode(1, integrator)
        for i in 1:12
            integrator.cm1[i] = integrator.tesco[i, 2] * integrator.elco[i, i + 1]
        end
        resetcoeff(integrator)
    end
    if integrator.jstart == -1
        integrator.ipup = integrator.miter
        integrator.lmax = integrator.maxord + 1
        if integrator.ialth == 1
            integrator.ialth = 2
        end
        if integrator.meth != integrator.mused
            cfode(integrator.meth, integrator)
            integrator.ialth = integrator.l
            resetcoeff(integrator)
        end
        if integrator.h != integrator.hold
            rh[] = integrator.h / integrator.hold
            integrator.h = integrator.hold
            scaleh(rh, pdh, integrator)
        end
    end
    if integrator.jstart == -2
        if integrator.h != integrator.hold
            rh[] = integrator.h / integrator.hold
            integrator.h = integrator.hold
            scaleh(rh, pdh, integrator)
        end
    end
    while true
        local pnorm
        while true
            if abs(integrator.rc - 1) > integrator.ccmax
                integrator.ipup = integrator.miter
            end
            if integrator.nst >= integrator.nslp + integrator.msbp
                integrator.ipup = integrator.miter
            end
            integrator.tn += integrator.h
            for j in integrator.nq : -1 : 1
                for i1 in j : integrator.nq
                    integrator.yp1 = @view integrator.yh[i1, :]
                    integrator.yp2 = @view integrator.yh[i1 + 1, :]
                    for i in 1:integrator.n
                        integrator.yp1[i] += integrator.yp2[i]
                    end
                end
            end
            pnorm = vmnorm(integrator.n, integrator.yh[1,:], integrator.ewt)
            #Ref??? y
            #integrator.nfe >= 80 && (DOPRINT[] = true)
            correction(neq, prob, corflag, pnorm, del, delp, told, ncf, rh, m, integrator)
            #DOPRINT[] && @printf(stderr, "tn = %f, del = %f, nfe = %d, method = %d, y[1] = %.12f\n", integrator.tn, del[], integrator.nfe, integrator.meth, y[1]);
            if corflag[] == 0
                break
            end
            if corflag[] == 1
                rh[] = max(rh[], integrator.hmin/abs(integrator.h))
                scaleh(rh, pdh, integrator)
                continue
            end
            if corflag[] == 2
                integrator.kflag = -2
                integrator.hold = integrator.h
                integrator.jstart = 1
                return
            end
        end
        integrator.jcur = 0
        if m[] == 0
            dsm = del[] / integrator.tesco[integrator.nq, 2]
        end
        if m[] > 0
            dsm = vmnorm(integrator.n, integrator.acor, integrator.ewt) / integrator.tesco[integrator.nq,2]
        end
        if dsm <= 1.0
            integrator.kflag = 0
            integrator.nst += 1
            integrator.hu = integrator.h
            integrator.nqu = integrator.nq
            integrator.mused = integrator.meth
            for j = 1:integrator.l
                integrator.yp1 = @view integrator.yh[j, :]
                r = integrator.el[j]
                for i = 1:integrator.n
                    integrator.yp1[i] += r * integrator.acor[i]
                end
            end
            integrator.icount -= 1
            if integrator.icount < 0
                methodswitch(dsm, pnorm, pdh, rh, integrator)
                if integrator.meth != integrator.mused
                    rh[] = max(rh[], integrator.hmin / abs(integrator.h))
                    scaleh(rh, pdh, integrator)
                    integrator.rmax = 10.0
                    endstoda(integrator)
                    break
                end
            end
            integrator.ialth -= 1
            if integrator.ialth == 0
                rhup = Ref(0.0)
                if integrator.l != integrator.lmax
                    integrator.yp1 = @view integrator.yh[integrator.lmax, :]
                    for i in 1:integrator.n
                        integrator.savf[i] = integrator.acor[i] - integrator.yp1[i]
                    end
                    dup = vmnorm(integrator.n, integrator.savf, integrator.ewt) / integrator.tesco[integrator.nq, 3]
                    exup = 1 / (integrator.l + 1)
                    rhup[] = 1 / (1.4 * dup ^ exup +0.0000014)
                end
                orderswitch(rhup, dsm, pdh, rh, orderflag, integrator)
                if orderflag[] == 0
                    endstoda(integrator)
                    break
                end
                if orderflag[] == 1
                    rh[] = max(rh[], integrator.hmin / abs(integrator.h))
                    scaleh(rh, pdh, integrator)
                    integrator.rmax = 10.0
                    endstoda(integrator)
                    break
                end
                if orderflag[] == 2
                    resetcoeff(integrator)
                    rh[] = max(rh[], integrator.hmin / abs(integrator.h))
                    scaleh(rh, pdh, integrator)
                    integrator.rmax = 10
                    endstoda(integrator)
                    break
                end
            end
            if integrator.ialth > 1 || integrator.l == integrator.lmax
                endstoda(integrator)
                break
            end
            integrator.yp1 = @view integrator.yh[integrator.lmax, :]
            for i in 1:integrator.n
                integrator.yp1[i] = integrator.acor[i]
            end
            endstoda(integrator)
            break
        else
            integrator.kflag -= 1
            integrator.tn = told[]
            for j in integrator.nq : -1 : 1
                for i1 in j:integrator.nq
                    integrator.yp1 = @view integrator.yh[i1, :]
                    integrator.yp2 = @view integrator.yh[i1+1, :]
                    for i in 1:integrator.n
                        integrator.yp1[i] -= integrator.yp2[i]
                    end
                end
            end
            integrator.rmax = 2.0
            if abs(integrator.h) <= integrator.hmin * 1.00001
                integrator.kflag = -1
                integrator.hold = integrator.h
                integrator.jstart = 1
                break
            end
            if integrator.kflag > -3
                rhup = Ref(0.0)
                orderswitch(rhup, dsm, pdh, rh, orderflag, integrator)
                if orderflag[] == 1 || orderflag[] == 0
                    if orderflag[] == 0
                        rh[] = min(rh[], 0.2)
                    end
                    rh[] = max(rh[], integrator.hmin / abs(integrator.h))
                    scaleh(rh, pdh, integrator)
                end
                if orderflag[] == 2
                    resetcoeff(integrator)
                    rh[] = max(rh[], integrator.hmin / abs(integrator.h))
                    scaleh(rh, pdh, integrator)
                end
                continue
            else
                if integrator.kflag == -10
                    integrator.kflag = -1
                    integrator.hold = integrator.h
                    integrator.jstart = 1
                    break
                else
                    rh[] = 0.1
                    rh[] = max(integrator.hmin / abs(integrator.h), rh[])
                    integrator.h *= rh[]
                    integrator.yp1 = @view integrator.yh[1,:]
                    for i in 1:integrator.n
                        y[i] = integrator.yp1[i]
                    end
                    @views prob.f(integrator.savf, y, prob.p, integrator.tn)
                    integrator.nfe += 1
                    integrator.yp1 = @view integrator.yh[2,:]
                    for i in 1:integrator.n
                        integrator.yp1[i] = integrator.h * integrator.savf[i]
                    end
                    integrator.ipup = integrator.miter
                    integrator.ialth = 5
                    if integrator.nq == 1
                        continue
                    end
                    integrator.nq = 1
                    integrator.l = 2
                    resetcoeff(integrator)
                    continue
                end
            end
        end
    end
end

function cfode(meth::Int, integrator::JLSODAIntegrator)
    pc = zeros(12)
    if meth == 1
        integrator.elco[1, 1] = 1.0
        integrator.elco[1, 2] = 1.0
        integrator.tesco[1, 1] = 0.0
        integrator.tesco[1, 2] = 2.0
        integrator.tesco[2, 1] = 1.0
        integrator.tesco[12, 3] = 0.0
        pc[1] = 1.0
        rqfac = 1.0
        for nq = 2:12
            rq1fac = rqfac
            rqfac = rqfac / nq
            nqm1 = nq - 1
            fnqm1 = Float64(nqm1)
            nqp1 = nq + 1
            pc[nq] = 0.0
            for i in nq : -1 : 2
                pc[i] = pc[i - 1] + fnqm1 * pc[i]
            end
            pc[1] = fnqm1 * pc[1]
            pint = pc[1]
            xpin = pc[1] / 2.0
            tsign = 1.0
            for i in 2:nq
                tsign = -tsign
                pint += tsign * pc[i] / i
                xpin += tsign * pc[i] / (i + 1)
            end
            integrator.elco[nq, 1] = pint * rq1fac
            integrator.elco[nq, 2] = 1.0
            for i in 2: nq
                integrator.elco[nq, i + 1] = rq1fac * pc[i] / i
            end
            agamq = rqfac * xpin
            ragq = 1.0 / agamq
            integrator.tesco[nq, 2] = ragq
            if nq < 12
                integrator.tesco[nqp1, 1] = ragq * rqfac / nqp1
            end
            integrator.tesco[nqm1, 3] = ragq
        end
        return
    end
    pc[1] = 1.0
    rq1fac = 1.0
    for nq in 1:5
        fnq = Float64(nq)
        nqp1 = nq + 1
        pc[nqp1] = 0
        for i in nq+1  : -1 : 2
            pc[i] = pc[i - 1] + fnq * pc[i]
        end
        pc[1] *= fnq
        for i = 1:nqp1
            integrator.elco[nq, i] = pc[i] / pc[2]
        end
        integrator.elco[nq, 2] = 1.0
        integrator.tesco[nq, 1] = rq1fac
        integrator.tesco[nq, 2] = Float64(nqp1) / integrator.elco[nq, 1]
        integrator.tesco[nq, 3] = Float64(nq + 2) / integrator.elco[nq, 1]
        rq1fac /= fnq
    end
    return
end

function scaleh(rh::Ref{Float64}, pdh::Ref{Float64}, integrator::JLSODAIntegrator)
    rh[] = min(rh[], integrator.rmax)
    rh[] = rh[] / max(1, abs(integrator.h * integrator.hmxi * rh[]))
    if integrator.meth ==1
        integrator.irflag = 0
        pdh[] = max(abs(integrator.h) * integrator.pdlast, 0.000001)
        if rh[] * pdh[] * 1.00001 >= SM1[integrator.nq]
            rh[] = SM1[integrator.nq] / pdh[]
            integrator.irflag = 1
        end
    end
    r = 1.0
    for j in 2:integrator.l
        r *= rh[]
        integrator.yp1 = @view integrator.yh[j, :]
        for i = 1:integrator.n
            integrator.yp1[i] *= r
        end
    end
    integrator.h *= rh[]
    integrator.rc *= rh[]
    integrator.ialth = integrator.l
end

function resetcoeff(integrator::JLSODAIntegrator)
    ep1 = integrator.elco[integrator.nq, :]
    for i in 1:integrator.l
        integrator.el[i] = ep1[i]
    end
    integrator.rc = integrator.rc * integrator.el[1] / integrator.el0
    integrator.el0 = integrator.el[1]
    integrator.conit = 0.5 / (integrator.nq + 2)
    return
end

function vmnorm(n::Int, v::Vector{Float64}, w::Vector{Float64})
    vm = 0.0
    for i in 1:n
        vm = max(vm, abs(v[i]) * w[i])
    end
    return vm
end

function ewset!(rtol::Ref{Float64}, atol::Ref{Float64}, ycur::Vector{Float64}, integrator::JLSODAIntegrator)
    for i in 1:integrator.n
        integrator.ewt[i] = rtol[] * abs(ycur[i]) + atol[]
    end
end

function intdy!(t::Float64, k::Int, dky::Vector{Float64}, iflag::Ref{Int}, integrator::JLSODAIntegrator)
    iflag[] = 0
    if (k < 0 || k > integrator.nq)
        @warn("[intdy] k = $k illegal\n")
        iflag[] = -1
        return
    end
    tp = integrator.tn - integrator.hu - 100 * eps() * (integrator.tn + integrator.hu)
    if (t - tp) * (t - integrator.tn) > 0
        @warn("intdy -- t = $t illegal. t not in interval tcur - hu to tcur\n")
        iflag[] = -2
        return
    end
    s = (t - integrator.tn) / integrator.h
    c = 1
    for jj in (integrator.l - k):integrator.nq
        c *= jj
    end
    integrator.yp1 = @view integrator.yh[integrator.l,:]
    for i in 1:integrator.n
        dky[i] = c * integrator.yp1[i]
    end
    for j in (integrator.nq -1 : -1 : k)
        jp1 = j + 1
        c = 1
        for jj in jp1 - k : j
            c *= jj
        end
        integrator.yp1 = @view integrator.yh[jp1, :]
        for i = 1 : integrator.n
            dky[i] = c * integrator.yp1[i] + s *dky[i]
        end
    end
    if k == 0
        return
    end
    r = h ^ (-k)
    for i in 1 : integrator.n
        dky[i] *= r
    end
    return
end

function prja(neq::Int, prob, integrator::JLSODAIntegrator)
    y = prob.u0
    integrator.nje += 1
    integrator.ierpj = 0
    integrator.jcur = 1
    hl0 = integrator.h * integrator.el0
    if integrator.miter != 2
        @warn("[prja] miter != 2\n")
        return
    end
    if integrator.miter == 2
        fac = vmnorm(integrator.n, integrator.savf, integrator.ewt)
        r0 = 1000 * abs(integrator.h) * eps() * integrator.n *fac
        if r0 == 0.0
            r0 = 1.0
        end
        for j in 1:integrator.n
            yj = y[j] ## y need to be changed
            r = max(sqrt(eps()) * abs(yj), r0/integrator.ewt[j])
            y[j] += r
            fac = -hl0 / r
            @views prob.f(integrator.acor, y, prob.p, integrator.tn)
            for i in 1:integrator.n
                integrator.wm[i, j] = (integrator.acor[i] - integrator.savf[i]) * fac
            end
            y[j] = yj
        end
        integrator.nfe += integrator.n
        integrator.pdnorm = fnorm(integrator.n, integrator.wm, integrator.ewt) / abs(hl0)
        for i in 1:integrator.n
            integrator.wm[i, i] += 1.0
        end
        integrator.LUresult = lu!(integrator.wm)
        issuccess(integrator.LUresult) || (integrator.ierpj = 1)
        return
    end
end

function fnorm(n::Int, a::Matrix{Float64}, w::Vector{Float64})
    an = 0
    for i in 1:n
        sum = 0
        ap1 = a[i,:]
        for j in 1:n
            sum += abs(ap1[j]) / w[j]
        end
        an = max(an, sum * w[i])
    end
    return an
end

function correction(neq::Int, prob::ODEProblem, corflag::Ref{Int}, pnorm::Float64,
    del::Ref{Float64}, delp::Ref{Float64}, told::Ref{Float64}, ncf::Ref{Int}, rh::Ref{Float64},
    m::Ref{Int}, integrator::JLSODAIntegrator)
    y = prob.u0
    m[] = 0
    corflag[] = 0
    rate = 0.0
    del[] = 0
    integrator.yp1 = @view integrator.yh[1, :]
    for i in 1:integrator.n
        y[i] = integrator.yp1[i]
    end
    @views prob.f(integrator.savf, y, prob.p, integrator.tn)
    integrator.nfe += 1
    while true
        if m[] == 0
            if integrator.ipup > 0
                prja(neq, prob, integrator)
                integrator.ipup = 0
                integrator.rc = 1.0
                integrator.nslp = integrator.nst
                integrator.crate = 0.7
                if integrator.ierpj != 0
                    corfailure(told, rh, ncf, corflag, integrator)
                    return
                end
            end
            for i in 1:integrator.n
                integrator.acor[i] = 0.0
            end
        end
        if integrator.miter == 0
            integrator.yp1 = @view integrator.yh[2, :]
            for i in 1:integrator.n
                integrator.savf[i] = integrator.h * integrator.savf[i] - integrator.yp1[i]
                y[i] = integrator.savf[i] - integrator.acor[i]
            end
            del[] = vmnorm(integrator.n, y, integrator.ewt)
            integrator.yp1 = @view integrator.yh[1, :]
            for i = 1:integrator.n
                y[i] = integrator.yp1[i] + integrator.el[1] * integrator.savf[i]
                integrator.acor[i] = integrator.savf[i]
            end
        else
            integrator.yp1 = @view integrator.yh[2, :]
            for i in 1:integrator.n
                y[i] = integrator.h * integrator.savf[i] - (integrator.yp1[i] + integrator.acor[i])
            end
            y[:] = solsy(y, integrator)
            del[] = vmnorm(integrator.n, y, integrator.ewt)
            integrator.yp1 = @view integrator.yh[1, :]
            for i in 1:integrator.n
                integrator.acor[i] += y[i]
                y[i] = integrator.yp1[i] + integrator.el[1] *integrator.acor[i]
            end
        end
        if del[] <= 100 *pnorm *eps()
            break
        end
        if m[] != 0 || integrator.meth != 1
            if m[] != 0
                rm = 1024
                if del[] <= (1024 * delp[])
                    rm = del[] / delp[]
                end
                rate = max(rate, rm)
                integrator.crate = max(0.2 * integrator.crate, rm)
            end
            dcon = del[] * min(1.0, 1.5 * integrator.crate) / (integrator.tesco[integrator.nq, 2] * integrator.conit)
            if dcon <= 1.0
                integrator.pdest = max(integrator.pdest, rate / abs(integrator.h * integrator.el[1]))
                if integrator.pdest != 0
                    integrator.pdlast = integrator.pdest
                end
                break
            end
        end
        m[] += 1
        if m[] == integrator.maxcor || (m[] >= 2 && del[] > 2 * delp[])
            if integrator.miter == 0 || integrator.jcur == 1
                corfailure(told, rh, ncf, corflag, integrator)
                return
            end
            integrator.ipup = integrator.miter
            m[] = 0
            rate = 0
            del[] = 0
            integrator.yp1 = @view integrator.yh[1, :]
            for i in 1:integrator.n
                y[i] = integrator.yp1[i]
            end
            @views prob.f(integrator.savf, y, prob.p, integrator.tn)
            integrator.nfe += 1
        else
            delp[] = del[]
            @views prob.f(integrator.savf, y, prob.p, integrator.tn)
            integrator.nfe += 1
        end
    end
end

function corfailure(told::Ref{Float64}, rh::Ref{Float64}, ncf::Ref{Int},
    corflag::Ref{Int}, integrator::JLSODAIntegrator)
    ncf[] += 1
    integrator.rmax = 2.0
    integrator.tn = told[]
    for j in integrator.nq : -1 : 1
        for i1 in j : integrator.nq
            integrator.yp1 = @view integrator.yh[i1, :]
            integrator.yp2 = @view integrator.yh[i1 + 1, :]
            for i in 1 : integrator.n
                integrator.yp1[i] -= integrator.yp2[i]
            end
        end
    end
    if (abs(integrator.h) <= integrator.hmin * 1.00001) || (ncf[] == integrator.mxncf)
        corflag[] = 2
        return
    end
    corflag[] = 1
    rh[] = 0.25
    integrator.ipup = integrator.miter
end

function solsy(y::Vector{Float64}, integrator::JLSODAIntegrator)
    integrator.iersl = 0
    if integrator.miter != 2
        print("solsy -- miter != 2\n")
        return
    end
    if integrator.miter == 2
        return ldiv!(integrator.LUresult, y)
    end
end

function methodswitch(dsm::Float64, pnorm::Float64, pdh::Ref{Float64}, rh::Ref{Float64}, integrator::JLSODAIntegrator)
    if (integrator.meth == 1)
        if (integrator.nq > 5)
            return
        end
        if (dsm <= (100 * pnorm * eps()) || integrator.pdest == 0)
            if (integrator.irflag == 0)
                return
            end
            rh2 = 2.0
            nqm2 = min(integrator.nq, integrator.mxords)
        else
            exsm = 1 / integrator.l
            rh1 = 1 / (1.2 * (dsm ^ exsm) + 0.0000012)
            rh1it = 2 * rh1
            pdh[] = integrator.pdlast * abs(integrator.h)
            if (pdh[] * rh1) > 0.00001
                rh1it = SM1[integrator.nq] / pdh[]
            end
            rh1 = min(rh1, rh1it)
            if (integrator.nq > integrator.mxords)
                nqm2 = integrator.mxords
                lm2 = integrator.mxords + 1
                exm2 = 1 / lm2
                lm2p1 = lm2 + 1
                dm2 = vmnorm(integrator.n, integrator.yh[lm2p1,:], integrator.ewt) / integrator.cm2[integrator.mxords]
                rh2 = 1 / (1.2 * (dm2 ^ exm2) + 0.0000012)
            else
                dm2 = dsm * (integrator.cm1[integrator.nq] / integrator.cm2[integrator.nq])
                rh2 = 1 / (1.2 * (dm2 ^ exsm) + 0.0000012)
                nqm2 = integrator.nq
            end
            if (rh2 < integrator.ratio * rh1)
                return
            end
        end

        rh[] = rh2
        integrator.icount = 20
        integrator.meth = 2
        integrator.miter = integrator.jtyp
        integrator.pdlast = 0.0
        integrator.nq = nqm2
        integrator.l = integrator.nq + 1
        return
    end

    exsm = 1 / integrator.l
    if integrator.mxordn < integrator.nq
        nqm1 = integrator.mxordn
        lm1 = integrator.mxordn + 1
        exm1 = 1 / lm1
        lm1p1 = lm1 + 1
        dm1 = vmnorm(integrator.n, integrator.yh[lm1p1,:], integrator.ewt) / integrator.cm1[MXORDN]
        rh1 = 1 / (1.2 * (dm1 ^ exm1) + 0.0000012)
    else
        #@show dsm, integrator.nq, integrator.cm2[integrator.nq], integrator.cm1[integrator.nq]
        dm1 = dsm * ((integrator.cm2[integrator.nq] / integrator.cm1[integrator.nq]))
        rh1 = 1 / (1.2 * (dm1 ^ exsm) + 0.0000012)
        nqm1 = integrator.nq
        exm1 = exsm
    end
    rh1it = 2 * rh1
    pdh[] = integrator.pdnorm * abs(integrator.h)
    if (pdh[] * rh1) > 0.00001
        rh1it = SM1[nqm1] / pdh[]
    end
    rh1 = min(rh1, rh1it)
    rh2 = 1 / (1.2 * (dsm ^ exsm) + 0.0000012)
    if (rh1 * integrator.ratio) < (5 * rh2)
        return
    end
    alpha = max(0.001, rh1)
    dm1 *= alpha ^ exm1
    if dm1 <= 1000 * eps() * pnorm
        return
    end
    rh[] = rh1
    integrator.icount = 20
    integrator.meth = 1
    integrator.miter = 0
    integrator.pdlast = 0.0
    integrator.nq = nqm1
    integrator.l = integrator.nq + 1
end

function endstoda(integrator::JLSODAIntegrator)
    r = 1 / integrator.tesco[integrator.nqu, 2]
    for i in 1:integrator.n
        integrator.acor[i] *= r
    end
    integrator.hold = integrator.h
    integrator.jstart = 1
    return
end


function orderswitch(rhup::Ref{Float64}, dsm::Float64, pdh::Ref{Float64}, rh::Ref{Float64}, orderflag::Ref{Int}, integrator::JLSODAIntegrator)
    orderflag[] = 0
    exsm = 1 / integrator.l
    rhsm = 1 / (1.2 * (dsm ^ exsm) + 0.0000012)
    rhdn = 0
    if integrator.nq != 1
        ddn = vmnorm(integrator.n, integrator.yh[integrator.l, :], integrator.ewt) / integrator.tesco[integrator.nq, 1]
        exdn = 1 / integrator.nq
        rhdn = 1 / (1.3 * (ddn ^ exdn) + 0.0000013)
    end

    if integrator.meth == 1
        pdh[] = max(abs(integrator.h) * integrator.pdlast, 0.000001)
        if integrator.l < integrator.lmax
            rhup[] = min(rhup[], SM1[integrator.l] / pdh[])
        end
        rhsm = min(rhsm, SM1[integrator.nq] / pdh[])
        if integrator.nq > 1
            rhdn = min(rhdn, SM1[integrator.nq - 1] / pdh[])
        end
        integrator.pdest = 0
    end
    if rhsm >= rhup[]
        if rhsm >= rhdn
            newq = integrator.nq
            rh[] = rhsm
        else
            newq = integrator.nq - 1
            rh[] = rhdn
            if integrator.kflag < 0 && rh[] > 1
                rh[] = 1
            end
        end
    else
        if rhup[] <= rhdn
            newq = integrator.nq - 1
            rh[] = rhdn
            if integrator.kflag < 0 && rh[] > 1
                rh[] = 1
            end
        else
            rh[] = rhup[]
            if rh[] >= 1.1
                r = integrator.el[integrator.l] / integrator.l
                integrator.nq = integrator.l
                integrator.l = integrator.nq + 1
                integrator.yp1 = @view integrator.yh[integrator.l,:]
                for i in 1:integrator.n
                    integrator.yp1[i] = integrator.acor[i] * r
                end
                orderflag[] = 2
                return
            else
                integrator.ialth = 3
                return
            end
        end
    end
    if integrator.meth == 1
        if rh[] * pdh[] * 1.00001 < SM1[newq]
            if integrator.kflag == 0 && rh[] < 1.1
                integrator.ialth = 3
                return
            end
        end
    else
        if integrator.kflag == 0 && rh[] < 1.1
            integrator.ialth = 3
            return
        end
    end
    if integrator.kflag <= -2
        rh[] = min(rh[], 0.2)
    end
    if newq == integrator.nq
        orderflag[] = 1
        return
    end
    integrator.nq = newq
    integrator.l = integrator.nq + 1
    orderflag[] = 2
    return
end

end  #module
