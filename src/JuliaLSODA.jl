module JuliaLSODA

using LinearAlgebra
using Reexport: @reexport
@reexport using DiffEqBase

using Printf
using Parameters: @unpack
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
mutable struct JLOptions{Rtol,Atol}
    rtol::Rtol
    atol::Atol
end

mutable struct JLSODAIntegrator{uType,tType,uElType,solType,Rtol,Atol}
    ccmax::uElType #in stru  #When rc differs from 1 by more than ccmax, ipup is set to miter
    el0::uElType #in stru #leading coeff
    h::tType #in stru
    hmin::tType #in stru
    hmxi::tType #in stru
    hu::tType #in stru
    rc::uElType #in stru #rc is the ratio of new to old values of the coefficient h * el[1].
    tn::tType #in stru
    tsw::tType #in stru
    pdnorm::tType #in stru what type
    pdest::tType #in stru  ???
    pdlast::tType #in stru ???
    ratio::uElType #in stru
    conit::uElType #in stru
    crate::uElType #in stru
    hold::tType #in stru
    rmax::uElType #in stru
    tfirst::tType

    #ml::Int64 #in stru
    #mu::Int64 #in stru
    imxer::Int #in stru
    illin::Int #in stru
    init::Int #in stru
    mxstep::Int #in stru
    mxhnil::Int #in stru
    nhnil::Int #in stru
    ntrep::Int #in stru
    nslast::Int #in stru
    nyh::Int #in stru
    ierpj::Int #in stru
    iersl::Int #in stru
    jcur::Int #in stru
    jstart::Int #in stru
    kflag::Int #in stru
    l::Int #in stru
    meth::Int #in stru
    miter::Int #in stru #corrector iteration method
    maxord::Int #in stru
    maxcor::Int #in stru
    msbp::Int #in stru
    mxncf::Int #in stru
    n::Int #in stru
    nq::Int #in stru
    nst::Int #in stru # number of steps
    nfe::Int #in stru # Number of f called
    nje::Int #in stru
    nqu::Int #in stru
    ixpr::Int #in stru
    jtyp::Int #in stru
    mused::Int #in stru
    mxordn::Int #in stru
    mxords::Int #in stru
    ialth::Int #in stru
    ipup::Int #in stru #force a matrix update
    lmax::Int #in stru
    nslp::Int #in stru
    icount::Int #in stru
    irflag::Int #in stru

    LUresult::LU{uElType,Array{uElType,2}}
    el::Vector{uElType} #  l(x) = el[1] + el[2]*x + ... + el[nq+1]*x^nq.
    elco::Array{uElType,2} # the method of order nq are stored in elco[nq][i]
    tesco::Array{uElType,2} #test constants
    cm1::Vector{uElType}
    cm2::Vector{uElType}
    yh::Matrix{uElType}
    wm::Matrix{uElType}
    ewt::uType
    savf::uType
    acor::uType #altered correction
    sol::uType
    solreturn::solType
    opts::JLOptions{Rtol,Atol}
    function JLSODAIntegrator(prob::ODEProblem, ::solType, ::Rtol, ::Atol) where{solType,Rtol,Atol}
        @unpack u0, tspan, p = prob
        obj = new{typeof(u0),eltype(tspan),eltype(u0),solType,Rtol,Atol}()
        uElType = eltype(u0)
        obj.el = zeros(uElType,13)
        obj.elco = zeros(uElType,12, 13)
        obj.tesco = zeros(uElType,12, 3)
        obj.cm1 = zeros(uElType,12)
        obj.cm2 = zeros(uElType,5)
        obj.yh = zeros(uElType,1,2)
        obj.wm = zeros(uElType,1,2)
        return obj
    end
end

struct LSODA <: DiffEqBase.AbstractODEAlgorithm
end

#=
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
    for i in 1:integrator.n
        y[i] = integrator.yh[1,:][i]
    end
    integrator.tfirst = integrator.tn
    integrator.illin = 0
    return
end
=#

function passtoy!(integrator)
    for i in 1:integrator.n
        integrator.sol[i] = integrator.yh[1,:][i]
    end
    return
end
#=
function successreturn!(y::Vector{Float64}, t::Ref{Float64}, itask::Int, ihit::Int, tcrit::Float64, istate::Ref{Int}, integrator::JLSODAIntegrator)
    for i in 1:integrator.n
        y[i] = integrator.yh[1,:][i]
    end
    integrator.tfirst = integrator.tn
    if itask == 4 || itask == 5
        if bool(ihit)
            t = tcrit
        end
    end
    #istate[] = 2
    integrator.illin = 0
    return
end
=#
function DiffEqBase.__solve(prob::ODEProblem{uType,tType,true}, alg::LSODA, timeseries=[], ts=[], ks=[];
                            callback = nothing, saveat= [], itask::Int=1, iopt::Bool=false,
                            save_everystep = isempty(saveat), dense = save_everystep && isempty(saveat),
                            save_start = save_everystep || isempty(saveat) || typeof(saveat) <: Number ? true : prob.tspan[1] in saveat,
                            save_timeseries = nothing,
                            alias_u0=false,
                            tout=prob.tspan[end], rtol =1e-4, atol = 1e-6,
                            tstops#=tstop=#=nothing) where {uType,tType}
    #DOPRINT[] = false
    @unpack f, u0, tspan, p = prob
    uElType = eltype(u0)
    timeseries = uType[]
    ttType = eltype(tspan)
    ts = ttType[]
    solreturn = DiffEqBase.build_solution(prob,alg,ts,timeseries)
    integrator = JLSODAIntegrator(prob, solreturn, rtol, atol)
    integrator.opts = JLOptions(rtol, atol)
    rType = typeof(rtol)
    aType = typeof(atol)
    integrator.pdnorm = 0
    integrator.jtyp = 2
    mxstp0 = 1e5
    mxhnl0 = 10
    iflag = Ref(0)
    countstop = 1
    if tstops != nothing && typeof(tstops) <: Number
        tcrit = tstops
    elseif tstops != nothing
        tcrit = tstops[1]
    else
        tcrit = nothing
    end



    #=
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
    =#
    integrator.tfirst = first(prob.tspan)
    neq = length(prob.u0)
    integrator.sol = copy(prob.u0)
    integrator.acor = copy(prob.u0)
    integrator.savf = copy(prob.u0)
    integrator.ewt = copy(prob.u0)
    y = integrator.sol
    integrator.n = neq
    integrator.illin = 0
    if tout == integrator.tfirst
        integrator.ntrep += 1
        integrator.ntrep < 5 && return
        @warn("[lsoda] repeated calls with istate = 1 and tout = t. run aborted.. apparent infinite loop\n")
        return
    end
    ###Block b ###
    #if istate[] == 1 || istate[] == 3
    ntrep = 0
    if neq <= 0
        @warn("[lsoda] neq = %d is less than 1\n", neq)
        return
    end
    #=
    if istate[] == 3 && neq > integrator.n
        @warn("[lsoda] istate = 3 and neq increased\n")
        return
    end
    =#

    #if iopt == false
    integrator.ixpr = 0
    integrator.mxstep = mxstp0
    integrator.mxhnil = mxhnl0
    integrator.hmxi = 0.0
    integrator.hmin = 0.0
    h0 = ttType(0.0)
    integrator.mxordn = MORD[1]
    integrator.mxords = MORD[2]
    #TODO iopt == true
    #(istate[] == 3) && (integrator.jstart = -1)

    ### Block c ###
    #if istate[] == 1
    integrator.tn = integrator.tfirst
    integrator.tsw = integrator.tfirst
    integrator.maxord = integrator.mxordn
    if tcrit != nothing
        #=if (tcrit - tout) * (tout - integrator.tfirst) < 0
            @warn("tcrit behind tout")
        end=#
        if (h0 != 0.0 && (integrator.tfirst + h0 - tcrit) * h0 > tType(0.0))
            h0 = tcrit - integrator.tfirst
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
    @views prob.f(integrator.yh[2,:], y, prob.p, integrator.tfirst)
    @views for i in 1:integrator.n
        integrator.yh[1,:][i] = y[i]
    end
    integrator.nq = 1
    integrator.h = 1.0
    ewset!(y, integrator)
    for i in 1:integrator.n
        if integrator.ewt[i] <= 0
            @warn("[lsoda] EWT[$i] = $(integrator.ewt[i]) <= 0.0")
            #terminate2!(y, t, integrator)
            return
        end
        integrator.ewt[i] = 1 / integrator.ewt[i]
    end

    #compute h0, the first step size,
    if h0 == 0.0
        tdist = abs(tout - integrator.tfirst)
        w0 = max(abs(integrator.tfirst), abs(tout))
        if tdist < 2 * eps(ttType) * w0
            @warn("[lsoda] tout too close to t to start integration")
            return
        end
        if typeof(integrator.opts.rtol) <: Number
            tol = integrator.opts.rtol
        else
            tol = maximum(integrator.opts.rtol)
        end
        if tol <= 0.0
            integrator.opts.atol <: Number ? atoli = integrator.opts.atol : integrator.atol[1]
            for i in 1:integrator.n
                ayi = abs(y[i])
                if typeof(integrator.opts.atol) <: Vector
                    atoli = integrator.opts.atol[i]
                end
                if ayi != 0
                    tol = max(tol, atoli/ayi)
                end
            end
        end
        if isa(tol,rType)
            tol = max(tol, 100 * eps(rType))
            tol = min(tol, rType(0.001))
        else
            tol = max(tol, 100 * eps(aType))
            tol = min(tol, aType(0.001))
        end
        #@show typeof(tol)
        sum = vmnorm(integrator.n, integrator.yh[2,:], integrator.ewt)
        #@show typeof(sum),111111,integrator.ewt
        sum = 1 / (tol * w0 * w0) + tol * sum * sum
        #@show typeof(sum),sum
        #@show typeof(integrator.yh[2,:])
        #@show integrator.ewt, typeof(tol)
        h0 = ttType(1 / sqrt(sum))
        h0 = min(h0, tdist)
        # h0 = h0 * ((tout - integrator.tfirst >= 0) ? 1 : -1)
        h0 = copysign(h0, tout - integrator.tfirst)
    end
    rh = abs(h0) * integrator.hmxi
    rh > 1 && (h0 /= rh)
    integrator.h = h0
    @views for i in 1:integrator.n
        integrator.yh[2,:][i] *= h0
    end
    #@show typeof(h0)
    ###Block d###
    #=if (istate[] == 2 || istate[] == 3)
        integrator.nslast = integrator.nst
        if itask == 1
            if ((integrator.tn - tout) * integrator.h >= 0)
                intdy!(tout, 0, integrator.sol, iflag, integrator)
                if iflag[] != 0
                    @warn("[lsoda] trouble from intdy, itask = $itask, tout = $tout\n")
                    terminate!(istate, integrator)
                    return
                end
                integrator.tfirst = tout
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
                intdy!(tout, 0, integrator.sol, iflag, integrator)
                if iflag != 0
                    @warn("[lsoda] trouble from intdy, itask = $itask, tout = $tout\n")
                    terminate!(istate, integrator)
                    return
                end
                integrator.tfirst = tout
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
                integrator.tfirst = tcrit
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
    end =#
    #saveat t#
    ttype = eltype(tspan)
    #@show saveat
    if typeof(saveat) <: Number
        if (tspan[1]:saveat:tspan[end])[end] == tspan[end]
          saveat_vec = convert(Vector{ttype},collect(ttype,tspan[1]+saveat:saveat:tspan[end] - saveat))
        else
          saveat_vec = convert(Vector{ttype},collect(ttype,tspan[1]+saveat:saveat:(tspan[end])))
        end
    else
        saveat_vec =  convert(Vector{ttype},collect(saveat))
    end

    if !isempty(saveat_vec) && saveat_vec[end] == tspan[2]
        pop!(saveat_vec)
    end
    saveat_vec = sort(unique(saveat_vec))
    #@show saveat_vec
    #=if !isempty(saveat_vec) && saveat_vec[1] == tspan[1]
        save_ts = sort(unique([saveat_vec;tout]))
    else
        save_ts = sort(unique([integrator.tfirst;saveat_vec;tout]))
    end

    if !isempty(saveat_vec) && saveat_vec[1] == tspan[1]
        save_ts = sort(unique([saveat_vec;tout]))
    else
        save_ts = sort(unique([integrator.tfirst;saveat_vec;tout]))
    end=#
    if !isempty(saveat_vec) && integrator.tfirst > saveat_vec[1]
        error("First saving timepoint is before the solving timespan")
    end

    if !isempty(saveat_vec) && tout < saveat_vec[end]
        error("Final saving timepoint is past the solving timespan")
    end

    if typeof(prob.u0) <: Number
        u0 = [prob.u0]
    else
        if alias_u0
            u0 = vec(prob.u0)
        else
            u0 = vec(deepcopy(prob.u0))
        end
    end

    ures = uType[]
    #save_start ? tsave = integrator.tfirst : tsave = typeof(integrator.tfirst)
    tsave = integrator.tfirst
    countsav = 1
    while true
        if integrator.nst != 0
            if ((integrator.nst - integrator.nslast) >= integrator.mxstep)
                @warn("[lsoda] $(integrator.mxstep) steps taken before reaching tout\n")
                #terminate2!(y, t, integrator)
                passtoy!(integrator)
                return
            end
            ewset!(integrator.yh[1,:], integrator)
            for i = 1:integrator.n
                if integrator.ewt[i] <= 0
                    @warn("[lsoda] ewt[$i] = $(integrator.ewt[i]) <= 0.\n")
                    passtoy!(integrator)
                    #terminate2!(y, t, integrator)
                    return
                end
                integrator.ewt[i] = 1 / integrator.ewt[i]
            end
        end
        tolsf = eps(eltype(integrator.sol)) * vmnorm(integrator.n, integrator.yh[1,:], integrator.ewt)
        if tolsf > 0.01
            tolsf *= 200
            if integrator.nst == 0
                @warn("""
                lsoda -- at start of problem, too much accuracy
                requested for precision of machine,
                suggested scaling factor = $tolsf
                """)
                return
            end
            @warn("""lsoda -- at t = $(integrator.tfirst), too much accuracy requested
                     for precision of machine, suggested
                     scaling factor = $tolsf""")
            passtoy!(integrator)
            #terminate2!(y, t, integrator)
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
            #itask ==

            if (integrator.tn - tout) * integrator.h < 0
                if save_everystep
                    push!(ures, copy(integrator.sol))
                    push!(ts, integrator.tn)
                elseif (integrator.tn - tsave) * integrator.h >= 0
                    if !(tcrit != nothing && tsave > tcrit)
                        interp = copy(integrator.sol)
                        intdy!(tsave, 0, interp, iflag, integrator)
                        push!(ures, interp)
                        push!(ts, tsave)
                        if tsave == saveat_vec[end]
                            tsave = tout
                        else
                            tsave = saveat_vec[countsav]
                            countsav += 1
                        end
                    end
                end

                if tcrit != nothing
                    hmx = abs(integrator.tn) + abs(integrator.h)
                    ihit = abs(integrator.tn - tcrit) <= 100 * eps(typeof(integrator.tn)) *hmx
                    if Bool(ihit)
                        integrator.tn = tcrit
                        integrator.sol = integrator.yh[1,:]
                        push!(ures, integrator.sol)
                        push!(ts, integrator.tn)
                        countstop += 1

                        if countstop <= length(tstops)
                            tcrit = tstops[countstop]
                        else
                            tcrit = nothing
                        end
                       # successreturn(y, t, itask, ihit, tcrit, istate, integrator)
                        continue
                    end
                    tnext = integrator.tn + integrator.h * (1 + 4 * eps(typeof(integrator.tn)))
                    if ((tnext - tcrit) * integrator.h <= 0)
                        continue
                    end
                    integrator.h = (tcrit - integrator.tn) * (1 - 4 * eps(typeof(integrator.tn)))
                    integrator.jstart = -2
                    continue
                end
                continue
            else
                intdy!(tout, 0, integrator.sol, iflag, integrator)
                push!(ures, copy(integrator.sol))
                push!(ts, tout)
                sizeu = size(prob.u0)
                timeseries = uType[]
                save_start ? start_idx = 1 : start_idx = 2
                if typeof(prob.u0)<:Number
                    for i=start_idx:length(ures)
                        push!(timeseries,ures[i][1])
                    end
                else
                    for i=start_idx:length(ures)
                        push!(timeseries,reshape(ures[i],sizeu))
                    end
                end
                solreturn = DiffEqBase.build_solution(prob, alg, ts, ures,
                retcode = :Success)
                return solreturn
            end


            #if itask == 2
            #    passtoy!(integrator)
                #successreturn(y, t, itask, ihit, tcrit, istate, integrator)

            #    return integrator.sol
            #end
            #if itask == 3
            #    if (integrator.tn - tout) * integrator.h >= 0
            #        passtoy!(integrator)
                    #successreturn(y, t, itask, ihit, tcrit, istate, integrator)
            #        return integrator.sol
            #    end
            #    continue
            #end
            #if itask == 4
            #=
                if (integrator.tn - tout) * integrator.h >= 0
                    intdy!(tout, 0, integrator.sol, iflag, integrator)
                    integrator.tfirst = tout
                    #istate[] =2
                    #integrator.illin = 0
                    return integrator.sol
                else
                    hmx = abs(integrator.tn) + abs(integrator.h)
                    ihit = abs(tn - tcrit) <= 100 * eps() *hmx
                    if Bool(ihit)

                       # successreturn(y, t, itask, ihit, tcrit, istate, integrator)
                        return integrator.sol
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
                passtoy!(integrator)
                #successreturn(y, t, itask, ihit, tcrit, istate, integrator)
                return integrator.sol
            end
            =#
        end

        if (integrator.kflag == -1 || integrator.kflag == -2)
            if integrator.kflag == -1
                @warn("""at t = $(integrator.tn), tentative step size h = $(integrator.h), step nst = $(integrator.nst)\n
                 error test failed repeatedly or with fabs(h) = hmin""")
            end
            if integrator.kflag == -2
                @warn("""         corrector convergence failed repeatedly or
                         with fabs(h) = hmin""")
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
            passtoy!(integrator)
            #terminate2!(y, t, integrator)
            return
        end
    end
    return
end

mutable struct StodaRef{tType,uType}
    corflag::Int
    ncf::Int
    m::Int
    orderflag::Int
    delp::uType
    del::uType
    pdh::tType
    rh::tType
    told:: tType
    function StodaRef(told::tType,::uType) where {tType,uType}
        obj = new{typeof(told),uType}()
        obj.corflag = 0
        obj.ncf = 0
        obj.m = 0
        obj.orderflag = 0
        obj.delp = 0
        obj.del = 0
        obj.pdh = 0.0
        obj.rh = 0.0
        obj.told = told
        return obj
    end
end

function stoda(neq::Int, prob::ODEProblem, integrator::JLSODAIntegrator)
    y = integrator.sol
    integrator.kflag = 0
    stodaref = StodaRef(integrator.tn,integrator.sol[1])
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
            stodaref.rh = integrator.h / integrator.hold
            integrator.h = integrator.hold
            scaleh(integrator, stodaref)
        end
    end
    if integrator.jstart == -2
        if integrator.h != integrator.hold
            stodaref.rh = integrator.h / integrator.hold
            integrator.h = integrator.hold
            scaleh(integrator, stodaref)
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
            @views for j in integrator.nq : -1 : 1
                for i1 in j : integrator.nq
                    for i in 1:integrator.n
                        integrator.yh[i1, :][i] += integrator.yh[i1 + 1, :][i]
                    end
                end
            end
            pnorm = vmnorm(integrator.n, integrator.yh[1,:], integrator.ewt)
            #Ref??? y
            #integrator.nfe >= 80 && (DOPRINT[] = true)
            correction(neq, prob, pnorm, integrator, stodaref)
            #DOPRINT[] && @printf(stderr, "tn = %f, del = %f, nfe = %d, method = %d, y[1] = %.12f\n", integrator.tn, stodaref.del, integrator.nfe, integrator.meth, y[1]);
            if stodaref.corflag == 0
                break
            end
            if stodaref.corflag == 1
                stodaref.rh = max(stodaref.rh, integrator.hmin/abs(integrator.h))
                scaleh(integrator, stodaref)
                continue
            end
            if stodaref.corflag == 2
                integrator.kflag = -2
                integrator.hold = integrator.h
                integrator.jstart = 1
                return
            end
        end
        integrator.jcur = 0
        #@show stodaref.m
        if stodaref.m == 0
            dsm = stodaref.del / integrator.tesco[integrator.nq, 2]
        end
        if stodaref.m > 0
            dsm = vmnorm(integrator.n, integrator.acor, integrator.ewt) / integrator.tesco[integrator.nq,2]
        end
        #@show dsm
        #@show integrator.tn,integrator.h
        #@show integrator.hu
        if dsm <= 1.0
            integrator.kflag = 0
            integrator.nst += 1
            integrator.hu = integrator.h
            integrator.nqu = integrator.nq
            integrator.mused = integrator.meth
            @views for j = 1:integrator.l
                r = integrator.el[j]
                for i = 1:integrator.n
                    integrator.yh[j,:][i] += r * integrator.acor[i]
                end
            end
            integrator.icount -= 1
            if integrator.icount < 0
                methodswitch(dsm, pnorm, integrator, stodaref)
                if integrator.meth != integrator.mused
                    stodaref.rh = max(stodaref.rh, integrator.hmin / abs(integrator.h))
                    scaleh(integrator, stodaref)
                    integrator.rmax = 10.0
                    endstoda(integrator)
                    break
                end
            end
            integrator.ialth -= 1
            if integrator.ialth == 0
                rhup = Ref(0.0)
                if integrator.l != integrator.lmax
                    for i in 1:integrator.n
                        integrator.savf[i] = integrator.acor[i] - integrator.yh[integrator.lmax, :][i]
                    end
                    dup = vmnorm(integrator.n, integrator.savf, integrator.ewt) / integrator.tesco[integrator.nq, 3]
                    exup = 1 / (integrator.l + 1)
                    rhup[] = 1 / (1.4 * dup ^ exup +0.0000014)
                end
                orderswitch(rhup, dsm, integrator, stodaref)
                if stodaref.orderflag == 0
                    endstoda(integrator)
                    break
                end
                if stodaref.orderflag == 1
                    stodaref.rh = max(stodaref.rh, integrator.hmin / abs(integrator.h))
                    scaleh(integrator, stodaref)
                    integrator.rmax = 10.0
                    endstoda(integrator)
                    break
                end
                if stodaref.orderflag == 2
                    resetcoeff(integrator)
                    stodaref.rh = max(stodaref.rh, integrator.hmin / abs(integrator.h))
                    scaleh(integrator, stodaref)
                    integrator.rmax = 10
                    endstoda(integrator)
                    break
                end
            end
            if integrator.ialth > 1 || integrator.l == integrator.lmax
                endstoda(integrator)
                break
            end
            @views for i in 1:integrator.n
                integrator.yh[integrator.lmax, :][i] = integrator.acor[i]
            end
            endstoda(integrator)
            break
        else
            integrator.kflag -= 1
            integrator.tn = stodaref.told
            for j in integrator.nq : -1 : 1
                for i1 in j:integrator.nq
                    @views for i in 1:integrator.n
                        integrator.yh[i1, :][i] -= integrator.yh[i1+1, :][i]
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
                orderswitch(rhup, dsm, integrator, stodaref)
                if stodaref.orderflag == 1 || stodaref.orderflag == 0
                    if stodaref.orderflag == 0
                        stodaref.rh = min(stodaref.rh, 0.2)
                    end
                    stodaref.rh = max(stodaref.rh, integrator.hmin / abs(integrator.h))
                    scaleh(integrator, stodaref)
                end
                if stodaref.orderflag == 2
                    resetcoeff(integrator)
                    stodaref.rh = max(stodaref.rh, integrator.hmin / abs(integrator.h))
                    scaleh(integrator, stodaref)
                end
                continue
            else
                if integrator.kflag == -10
                    integrator.kflag = -1
                    integrator.hold = integrator.h
                    integrator.jstart = 1
                    break
                else
                    stodaref.rh = 0.1
                    stodaref.rh = max(integrator.hmin / abs(integrator.h), stodaref.rh)
                    integrator.h *= stodaref.rh
                    for i in 1:integrator.n
                        y[i] = integrator.yh[1,:][i]
                    end
                    @views prob.f(integrator.savf, y, prob.p, integrator.tn)
                    integrator.nfe += 1
                    @views for i in 1:integrator.n
                        integrator.yh[2,:][i] = integrator.h * integrator.savf[i]
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
            fnqm1 = convert(eltype(integrator.sol),nqm1)
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
        fnq = convert(eltype(integrator.sol),nq)
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
        integrator.tesco[nq, 2] = nqp1 / integrator.elco[nq, 1]
        integrator.tesco[nq, 3] = (nq + 2) / integrator.elco[nq, 1]
        rq1fac /= fnq
    end
    return
end

function scaleh(integrator::JLSODAIntegrator, stodaref::StodaRef)
    stodaref.rh = min(stodaref.rh, integrator.rmax)
    stodaref.rh = stodaref.rh / max(1, abs(integrator.h * integrator.hmxi * stodaref.rh))
    if integrator.meth ==1
        integrator.irflag = 0
        stodaref.pdh = max(abs(integrator.h) * integrator.pdlast, 0.000001)
        if stodaref.rh * stodaref.pdh * 1.00001 >= SM1[integrator.nq]
            stodaref.rh = SM1[integrator.nq] / stodaref.pdh
            integrator.irflag = 1
        end
    end
    r = 1.0
    for j in 2:integrator.l
        r *= stodaref.rh
        @views for i = 1:integrator.n
            integrator.yh[j,:][i] *= r
        end
    end
    integrator.h *= stodaref.rh
    integrator.rc *= stodaref.rh
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

function vmnorm(n::Int, v::Vector, w::Vector)
    eltypev = eltype(v)
    vm = eltypev(0.0)
    for i in 1:n
        vm = max(vm, abs(v[i]) * w[i])
    end
    return vm
end

function ewset!(ycur, integrator::JLSODAIntegrator)
    if typeof(integrator.opts.atol) <: Number && typeof(integrator.opts.rtol) <: Number
        for i in 1:integrator.n
            integrator.ewt[i] = integrator.opts.rtol * abs(ycur[i]) + integrator.opts.atol
        end
    elseif typeof(integrator.opts.atol) <:Number && typeof(integrator.opts.rtol) <:Vector
        for i in 1:integrator.n
            integrator.ewt[i] = integrator.opts.rtol[i] * abs(ycur[i]) + integrator.opts.atol
        end
    elseif typeof(integrator.opts.atol) <: Vector && typeof(integrator.opts.rtol) <:Vector
        for i in 1:integrator.n
            integrator.ewt[i] = integrator.opts.rtol[i] * abs(ycur[i]) + integrator.opts.atol[i]
        end
    elseif typeof(integrator.opts.atol) <:Vector && typeof(integrator.opts.rtol) <:Number
        for i in 1:integrator.n
            integrator.ewt[i] = integrator.opts.rtol * abs(ycur[i]) + integrator.opts.atol[i]
        end
    end
end

function intdy!(t :: Real, k::Int, dky, iflag::Ref{Int}, integrator::JLSODAIntegrator)
    iflag[] = 0
    if (k < 0 || k > integrator.nq)
        @warn("[intdy] k = $k illegal\n")
        iflag[] = -1
        return
    end
    tp = integrator.tn - integrator.hu - 100 * eps(typeof(integrator.tn)) * (integrator.tn + integrator.hu)
    if (t - tp) * (t - integrator.tn) > 0
        @warn("intdy -- t = $t illegal.
        t not in interval tcur - hu to tcur\n")
        iflag[] = -2
        return
    end
    s = (t - integrator.tn) / integrator.h
    c = 1
    for jj in (integrator.l - k):integrator.nq
        c *= jj
    end
    for i in 1:integrator.n
        dky[i] = c * integrator.yh[integrator.l,:][i]
    end
    for j in (integrator.nq -1 : -1 : k)
        jp1 = j + 1
        c = 1
        for jj in jp1 - k : j
            c *= jj
        end
        for i = 1 : integrator.n
            dky[i] = c * integrator.yh[jp1,:][i] + s *dky[i]
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

function prja(neq::Int, prob::ODEProblem, integrator::JLSODAIntegrator)
    y = integrator.sol
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
        r0 = 1000 * abs(integrator.h) * eps(typeof(integrator.tn)) * integrator.n *fac
        if r0 == 0.0
            r0 = 1.0
        end
        for j in 1:integrator.n
            yj = y[j] ## y need to be changed
            r = max(sqrt(eps(eltype(integrator.sol))) * abs(yj), r0/integrator.ewt[j])
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

function fnorm(n::Int, a, w)
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

function correction(neq::Int, prob::ODEProblem, pnorm,
    integrator::JLSODAIntegrator, stodaref::StodaRef)
    y = integrator.sol
    stodaref.m = 0
    stodaref.corflag = 0
    rate = 0.0
    stodaref.del = 0
    for i in 1:integrator.n
        y[i] = integrator.yh[1,:][i]
    end
    @views prob.f(integrator.savf, y, prob.p, integrator.tn)
    integrator.nfe += 1
    while true
        if stodaref.m == 0
            if integrator.ipup > 0
                prja(neq, prob, integrator)
                integrator.ipup = 0
                integrator.rc = 1.0
                integrator.nslp = integrator.nst
                integrator.crate = 0.7
                if integrator.ierpj != 0
                    corfailure(integrator, stodaref)
                    return
                end
            end
            for i in 1:integrator.n
                integrator.acor[i] = 0.0
            end
        end
        if integrator.miter == 0
            for i in 1:integrator.n
                integrator.savf[i] = integrator.h * integrator.savf[i] - integrator.yh[2,:][i]
                y[i] = integrator.savf[i] - integrator.acor[i]
            end
            stodaref.del = vmnorm(integrator.n, y, integrator.ewt)
            for i = 1:integrator.n
                y[i] = integrator.yh[1,:][i] + integrator.el[1] * integrator.savf[i]
                integrator.acor[i] = integrator.savf[i]
            end
        else
            for i in 1:integrator.n
                y[i] = integrator.h * integrator.savf[i] - (integrator.yh[2,:][i] + integrator.acor[i])
            end
            y[:] = solsy(y, integrator)
            stodaref.del = vmnorm(integrator.n, y, integrator.ewt)
            for i in 1:integrator.n
                integrator.acor[i] += y[i]
                y[i] = integrator.yh[1, :][i] + integrator.el[1] *integrator.acor[i]
            end
        end
        if stodaref.del <= 100 *pnorm *eps(eltype(integrator.sol)) #= not sure =#
            break
        end
        if stodaref.m != 0 || integrator.meth != 1
            if stodaref.m != 0
                rm = 1024
                if stodaref.del <= (1024 * stodaref.delp)
                    rm = stodaref.del / stodaref.delp
                end
                rate = max(rate, rm)
                integrator.crate = max(0.2 * integrator.crate, rm)
            end
            dcon = stodaref.del * min(1.0, 1.5 * integrator.crate) / (integrator.tesco[integrator.nq, 2] * integrator.conit)
            if dcon <= 1.0
                integrator.pdest = max(integrator.pdest, rate / abs(integrator.h * integrator.el[1]))
                if integrator.pdest != 0
                    integrator.pdlast = integrator.pdest
                end
                break
            end
        end
        stodaref.m += 1
        if stodaref.m == integrator.maxcor || (stodaref.m >= 2 && stodaref.del > 2 * stodaref.delp)
            if integrator.miter == 0 || integrator.jcur == 1
                corfailure(integrator, stodaref)
                return
            end
            integrator.ipup = integrator.miter
            stodaref.m = 0
            rate = 0
            stodaref.del = 0
            for i in 1:integrator.n
                y[i] = integrator.yh[1, :][i]
            end
            @views prob.f(integrator.savf, y, prob.p, integrator.tn)
            integrator.nfe += 1
        else
            stodaref.delp = stodaref.del
            @views prob.f(integrator.savf, y, prob.p, integrator.tn)
            integrator.nfe += 1
        end
    end
end

function corfailure(integrator::JLSODAIntegrator, stodaref::StodaRef)
    stodaref.ncf += 1
    integrator.rmax = 2.0
    integrator.tn = stodaref.told
    for j in integrator.nq : -1 : 1
        for i1 in j : integrator.nq
            @views for i in 1 : integrator.n
                integrator.yh[i1, :][i] -= integrator.yh[i1 + 1, :][i]
            end
        end
    end
    if (abs(integrator.h) <= integrator.hmin * 1.00001) || (stodaref.ncf == integrator.mxncf)
        stodaref.corflag = 2
        return
    end
    stodaref.corflag = 1
    stodaref.rh = 0.25
    integrator.ipup = integrator.miter
end

function solsy(y::Vector, integrator::JLSODAIntegrator)
    integrator.iersl = 0
    if integrator.miter != 2
        print("solsy -- miter != 2\n")
        return
    end
    if integrator.miter == 2
        return ldiv!(integrator.LUresult, y)
    end
end

function methodswitch(dsm, pnorm, integrator::JLSODAIntegrator, stodaref::StodaRef)
    if (integrator.meth == 1)
        if (integrator.nq > 5)
            return
        end
        if (dsm <= (100 * pnorm * eps(eltype(integrator.sol #=not sure=#))) || integrator.pdest == 0)
            if (integrator.irflag == 0)
                return
            end
            rh2 = 2.0
            nqm2 = min(integrator.nq, integrator.mxords)
        else
            exsm = 1 / integrator.l
            rh1 = 1 / (1.2 * (dsm ^ exsm) + 0.0000012)
            rh1it = 2 * rh1
            stodaref.pdh = integrator.pdlast * abs(integrator.h)
            if (stodaref.pdh * rh1) > 0.00001
                rh1it = SM1[integrator.nq] / stodaref.pdh
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

        stodaref.rh = rh2
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
    stodaref.pdh = integrator.pdnorm * abs(integrator.h)
    if (stodaref.pdh * rh1) > 0.00001
        rh1it = SM1[nqm1] / stodaref.pdh
    end
    rh1 = min(rh1, rh1it)
    rh2 = 1 / (1.2 * (dsm ^ exsm) + 0.0000012)
    if (rh1 * integrator.ratio) < (5 * rh2)
        return
    end
    alpha = max(0.001, rh1)
    dm1 *= alpha ^ exm1
    if dm1 <= 1000 * eps(eltype(integrator.sol)) #= not sure =# * pnorm
        return
    end
    stodaref.rh = rh1
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


function orderswitch(rhup, dsm, integrator::JLSODAIntegrator, stodaref::StodaRef)
    stodaref.orderflag = 0
    exsm = 1 / integrator.l
    rhsm = 1 / (1.2 * (dsm ^ exsm) + 0.0000012)
    rhdn = 0
    if integrator.nq != 1
        ddn = vmnorm(integrator.n, integrator.yh[integrator.l, :], integrator.ewt) / integrator.tesco[integrator.nq, 1]
        exdn = 1 / integrator.nq
        rhdn = 1 / (1.3 * (ddn ^ exdn) + 0.0000013)
    end

    if integrator.meth == 1
        stodaref.pdh = max(abs(integrator.h) * integrator.pdlast, 0.000001)
        if integrator.l < integrator.lmax
            rhup[] = min(rhup[], SM1[integrator.l] / stodaref.pdh)
        end
        rhsm = min(rhsm, SM1[integrator.nq] / stodaref.pdh)
        if integrator.nq > 1
            rhdn = min(rhdn, SM1[integrator.nq - 1] / stodaref.pdh)
        end
        integrator.pdest = 0
    end
    if rhsm >= rhup[]
        if rhsm >= rhdn
            newq = integrator.nq
            stodaref.rh = rhsm
        else
            newq = integrator.nq - 1
            stodaref.rh = rhdn
            if integrator.kflag < 0 && stodaref.rh > 1
                stodaref.rh = 1
            end
        end
    else
        if rhup[] <= rhdn
            newq = integrator.nq - 1
            stodaref.rh = rhdn
            if integrator.kflag < 0 && stodaref.rh > 1
                stodaref.rh = 1
            end
        else
            stodaref.rh = rhup[]
            if stodaref.rh >= 1.1
                r = integrator.el[integrator.l] / integrator.l
                integrator.nq = integrator.l
                integrator.l = integrator.nq + 1
                @views for i in 1:integrator.n
                    integrator.yh[integrator.l, :][i] = integrator.acor[i] * r
                end
                stodaref.orderflag = 2
                return
            else
                integrator.ialth = 3
                return
            end
        end
    end
    if integrator.meth == 1
        if stodaref.rh * stodaref.pdh * 1.00001 < SM1[newq]
            if integrator.kflag == 0 && stodaref.rh < 1.1
                integrator.ialth = 3
                return
            end
        end
    else
        if integrator.kflag == 0 && stodaref.rh < 1.1
            integrator.ialth = 3
            return
        end
    end
    if integrator.kflag <= -2
        stodaref.rh = min(stodaref.rh, 0.2)
    end
    if newq == integrator.nq
        stodaref.orderflag = 1
        return
    end
    integrator.nq = newq
    integrator.l = integrator.nq + 1
    stodaref.orderflag = 2
    return
end

end  #module
