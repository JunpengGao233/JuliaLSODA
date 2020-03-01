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
    return
end

function terminate2!(y::Vector{Float64}, t::Ref{Float64})
    YP1[] = YH[][1,:]
    for i in 1:N[]
        y[i] = YP1[][i]
    end
    t[] = TN[]
    ILLIN[] = 0
    return
end

function successreturn!(y::Vector{Float64}, t::Ref{Float64}, itask::Int, ihit::Int, icrit::Float64, istate::Ref{Int})
    YP[] = YH[][1,:]
    for i in 1:N[]
        y[i] = YP1[i]
    end
    t[] = TN[]
    if itask == 4 || itask == 5
        ihit && (t = tcrit)
    end
    istate[] = 2
    ILLIN[] = 0
    return
end

function DiffEqBase.__solve(prob::ODEProblem{uType,tType,true}, ::LSODA;
                            itask::Int=1, istate::Ref{Int}=Ref(1), iopt::Bool=false,
                            tout=prob.tspan[end], reltol=1e-4, abstol=1e-6,
                            tcrit#=tstop=#=nothing) where {uType,tType}
    mxstp0 = 500
    mxhnl0 = 10
    iflag[] = Ref(0)
    1 <= istate[] <= 3 || (@warn("[lsoda] illegal istate = $istate\n") || terminate(istate[]))
    !(itask < 1 || itask > 5) || (@warn( "[lsoda] illegal itask = $itask\n") || terminate(istate[]))
    !(INIT[] == 0 &&(istate[] == 2 || istate[] == 3)) || (@warn("[lsoda] istate > 1 but lsoda not initialized") || terminate(istate[]))

    t = Ref(first(prob.tspan))
    neq = length(prob.u0)

    # NOTE!!! this function mutates `prob.u0`
    y = prob.u0
    N[] = neq
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
		MXHNIL[] = mxhnl0
		HMXI[] = 0.0
		HMIN[] = 0.0
		if (istate == 1)
			h0 = 0.0
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
                terminate!(istate)
            end
            if (h0 != 0.0 && (t[] + h0 - tcrit) * h0 > 0.0)
                h0 = tcrit - t[]
            end
        end

        JSTART[] = 0
        NHNIL[] = 0
        NST[] = 0
        NJE[] = 0
        NSLAST[] = 0
        HU[] = 0.0
        NQU[] = 0
        MUSED[] = 0
        MITER[] = 0
        CCMAX[] = 0.3
        MAXCOR[] = 3
        MSBP[] = 20
        MXNCF[] = 10

        #prob.f(du,u,p,t)
        #(double t, double *y, double *ydot, void *data)

        prob.f(YH[][2,:], y, prob.p, t[])
        NFE[] = 1
        YP1[] = YH[][1,:]
        for i in 1:N[]
            YP1[][i] = y[i]
        end
        NQ[] = 1
        H[] = 1.0
        EWT[] = similar(y)
        ewset!(rtol, atol, y)
        for i in 1:N[]
            if EWT[][i] <= 0
                @warn("[lsoda] EWT[$i] = $(EWT[][i]) <= 0.0")
                terminate2!(y, t)
                return
            end
            EWT[][i] = 1 / EWT[][i]
        end

        #compute h0, the first step size,
        if h0 == 0.0
            tdist = abs(tout - t[])
            w0 = max(abs(t[]), abs(tout))
            if tdist < 2 * eps() *w0
                @warn("[lsoda] tout too close to t to start integration")
                terminate!(istate)
            end
            tol = rtol
            if tol <= 0.0
                for i in 1:N[]
                    ayi = abs(y[i])
                    if ayi != 0
                        tol = max(rtol, atol/ayi)
                    end
                end
            end
            tol = max(tol, 100 * eps())
            tol = min(tol, 0.001)
            sum = vmnorm(N[], YH[][2,:], EWT[])
            sum = 1 / (tol * 100 * eps())
            h0 = 1 / sqrt(sum)
            h0 = min(h0, tdist)
            # h0 = h0 * ((tout - t[] >= 0) ? 1 : -1)
            h0 = copysign(h0, tout - t[])
        end
        rh = abs(h0) * HMXI[]
        rh > 1 && (h0 /= rh)
        H[] = h0
        YP1[] = YH[][2,:]
        for i in 1:N[]
            YP1[][i] *= h0
        end
    end

    ###Block d###
    if (istate[] == 2 || istate[] == 3)
        NSLAST[] = NST[]
        if itask == 1
            if ((TN[] - tout) * H[] >= 0)
                intdy!(tout, 0, y, iflag[])
                if iflag[] != 0
                    @warn("[lsoda] trouble from intdy, itask = $itask, tout = $tout\n")
                    terminate!(istate)
                end
                t[] = tout
                istate[] = 2
                ILLIN[] = 0
                return
            end
        elseif itask == 3
            tp = TN[] - HU[] * (1 + 100 * eps())
            if ((tp - tout) * H[] > 0)
                @warn("[lsoda] itask = $itask and tout behind tcur - hu")
                terminate!(istate)
            end
            if ((TN[] - tout) * H[] >= 0)
                successreturn!(y, t, itask, ihit, tcrit, istate)
                return
            end
        elseif itask == 4
            if ((tn - tcrit) * H[] > 0)
                @warn("[lsoda] itask = 4 or 5 and tcrit behind tcur")
                terminate!(istate)
                return
            end
            if ((tcrit - tout) * H[] < 0)
                @warn("[lsoda] itask = 4 or 5 and tcrit behind tout")
                terminate!(istate)
                return
            end
            if ((TN[] - tout) * H[] >= 0)
                intdy!(tout, 0, y, iflag)
                if iflag != 0
                    @warn("[lsoda] trouble from intdy, itask = $itask, tout = $tout\n")
                    terminate!(istate)
                    return
                end
                t[] = tout
                istate[] = 2
                ILLIN[] = 0
                return
            end
        elseif itask == 5
            if ((tn - tcrit) * H[] > 0)
                @warn("[lsoda] itask = 4 or 5 and tcrit behind tcur")
                terminate!(istate)
                return
            end
            hmx = abs(TN[]) + abs(h)
            ihit = abs(TN[] - tcrit) <= (100 * eps() *hmx)
            if ihit
                t[] = tcrit
                successreturn!(y, t, itask, tcrit, istate)
                return
            end
            tnext = tn + H[] * (1 + 4 * eps())
            if (tnext - tcrit) * H[] > 0
                H[] = (tcrit - tn) * (1 - 4 * eps())
                if istate[] == 2
                    JSTART[] = -2
                end
            end
        end
    end
    #Block e#
    while 1
        if (istate[] != 1 || NST[] != 0)
            if ((NST[]-NSLAST[]) >= MXSTEP[])
                @warn("[lsoda] $(MXSTEP[]) steps taken before reaching tout\n")
                istate[] = -1
                terminate2!(y, t)
                return
            end
            ewset!(rtol, atol, YH[1,:])
            for i = 1:N[]
                if EWT[][i] <= 0
                    @warn("[lsoda] ewt[$i] = $(EWT[][i]) <= 0.\n")
                    istate[] = -6
                    terminate2!(y, t)
                    return
                end
                EWT[][i] = 1 / EWT[][i]
            end
        end
        tolsf = eps() * vmnorm(N[], YH[][1,:], EWT[])
        if tolsf > 0.01
            tolsf *= 200
            if NST[] == 0
                @warn("""
                lsoda -- at start of problem, too much accuracy
                requested for precision of machine,
                suggested scaling factor = $tolsf
                """)
                terminate!(istate)
                return
            end
            @warn("""lsoda -- at t = $(t[]), too much accuracy requested
                     for precision of machine, suggested
                     scaling factor = $tolsf""")
            istate[] = -2
            terminate2!(y, t)
            return
        end
        if ((TN[] + H[]) == TN[])
            NHNIL[] += 1
            if NHNIL[] <= MXHNIL[]
                @warn( """lsoda -- warning..internal t = $(TN[]) and h = $(H[]) are
                         such that in the machine, t + h = t on the next step
                         solver will continue anyway.""")
                if NHNIL[] == MXHNIL[]
                    @warn("""lsoda -- above warning has been issued $(NHNIL[]) times,
                            it will not be issued again for this problem\n""")
                end
            end
        end

        stoda(neq, prob)
        
        #Block f#
        if KFLAG[] == 0
            INIT[] = 1
            if METH[] != MUSED[]
                TSW[] = TN[]
                MAXCOR[] = MXORDS[]
                if METH[] == 2
                    MAXORD[] = MXORDS[]
                end
                JSTART[] = -1
                if IXPR[]
                    METH[] == 2 && @warn("[lsoda] a switch to the stiff method has occurred")
                    METH[] == 1 && @warn("[lsoda] a switch to the nonstiff method has occurred")
                    @warn("at t = $(TN[]), tentative step size h = $(H[]), step nst = $(NST[])\n")
                end
            end
            #itask == 1
            if itask == 1
                if (TN[] - tout) * H[] < 0
                    continue
                end
                intdy(tout, 0, y, iflag)
                t[] = tout
                istate[] = 2
                ILLIN[] = 0
                return
            end
            if itask == 2
                successreturn(y, t, itask, ihit, tcrit, istate)
                return
            end
            if itask == 3
                if (TN[] - tout) * h >= 0
                    successreturn(y, t, itask, ihit, tcrit, istate)
                    return
                end
                continue
            end
            if itask == 4
                if (TN[] - tout) * h >= 0
                    intdy(tout, 0, y, iflag)
                    t[] = tout
                    istate[] =2
                    ILLIN[] = 0
                    return
                else
                    hmx = abs(TN[]) + abs(H[])
                    ihit = abs(tn - tcrit) <= 100 * eps() *hmx
                    if ihit
                        successreturn(y, t, itask, ihit, tcrit, istate)
                        return
                    end
                    tnext = TN[] + H[] * (1 + 4 * eps())
                    if ((tnext - tcrit) * H[] <= 0)
                        continue
                    end
                    H[] = (tcrit - TN[]) * (1 - 4 * eps())
                    JSTART[] = -2
                    continue
                end
            end
            if itask == 5
                hmx = abs(TN[]) + abs(H[])
                ihit = abs(TN[] - tcrit) <= (100 * eps() * hmx)
                successreturn(y, t, itask, ihit, tcrit, istate)
                return
            end
        end
        if (KFLAG[] == -1 || KFLAG[] == -2)
            if KFLAG[] == -1
                @warn("""at t = $(TN[]), tentative step size h = $(H[]), step nst = $(NST[])\n
                 error test failed repeatedly or with fabs(h) = hmin""")
                istate[] = -4
            end
            if KFLAG[] == -2
                @warn("""         corrector convergence failed repeatedly or
                         with fabs(h) = hmin""")
                istate[] = -5
            end
            big = 0
            IMXER[] = 1
            for i in 1:N[]
                sizing = abs(ACOR[i]) * EWT[][i]
                if big < sizing
                    big = sizing
                    IMXER[] = 1
                end
            end
            terminate2(y, t)
            return
        end
    end
    return
end

function stoda(neq::Int, prob)
    KFLAG[] = 0
    told = Ref(TN[])
    corflag = Ref(0)
    ncf = Ref(0)
    delp = Ref(0.0)
    del = Ref(0.0)
    m = Ref(0)
    pdh = Ref(0.0)
    rh = Ref(0.0)    
    orderflag = Ref(0)
    IERPJ[] = 0
    IERSL[] = 0
    JCUR[] = 0
    if JSTART[] == 0
        LMAX[] = MAXORD[] + 1
        NQ[] = 1
        L[] = 2
        IALTH[] = 2
        RMAX[] = 10000.0
        RC[] = 0
        EL0[] = 1.0
        CRATE[] = 0.7
        HOLD[] = H[]
        NSLP[] = 0
        IPUP[] = MITER[]
        ICOUNT[] = 20
        IRFLAG[] = 0
        PDEST[] = 0.0
        PDLAST[] = 0.0
        RATIO[] = 5.0
        cfode(2)
        for i in 1:5
            CM2[i] = TESCO[i][2] * ELCO[i][i + 1]
        end
        cfode(1)
        for i in 1:12
            CM1[i] = TESCO[i][2] * ELCO[i][i + 1]
        end
        resetcoeff()
    end
    if JSTART[] == -1
        IPUP[] = MITER[]
        LMAX[] = MAXCOR[] + 1
        if IALTH[] == 1
            IALTH = 2
        end
        if METH[] != MUSED[]
            cfode(METH[])
            IALTH[] = 1
            resetcoeff()
        end
        if H[] != HOLD[]
            rh[] = H[] / HOLD[]
            H[] = HOLD[]
            scaleh(rh, pdh)
        end
    end
    if JSTART[] == -2
        if H[] != HOLD[]
            rh[] = H[] / HOLD[]
            H[] = HOLD[]
            scaleh(rh, pdh)
        end
    end
    while 1
        while 1
            if abs(RC[] - 1) > CCMAX[]
                IPUP[] = MITER[]
            end
            if NST[] >= NSLP[] + MSBP[]
                IPUP[] = MITER[]
            end
            TN[] += H[]
            for j in NQ : -1 : 1
                for i1 in j:NQ[]
                    YP1[] = YH[][i1]
                    YP2 = YH[][i1 + 1]
                    for i in 1:N[]
                        YP1[][i] += YP2[i]
                    end
                end
            end
            pnorm = vmnorm(N[], YH[][1,:], EWT[])
            #Ref??? y 
            correction(neq, y, prob.f, corflag, pnorm, del, delp, told, ncf, rh, m, prob.p)
            if corflag[] == 0
                break
            end
            if corflag[] == 1
                rh[] = max(rh[], HMIN[]/abs(H[]))
                scaleh(rh, pdh)
                continue
            end
            if corflag[] == 2
                KFLAG[] = -2
                HOLD[] = H[]
                JSTART[] = 1
                return
            end
        end
        JCUR[] = 0
        if m == 0
            dsm = del[] / TESCO[NQ[]][2]
        end
        if m[] > 0
            dsm = vmnorm(N[], ACOR[], EWT[]) / TESCO[NQ[]][2]
        end
        if dsm <= 1.0
            KFLAG[] = 0
            NST[] += 1
            HU[] = H[]
            NQU[] = NQ[]
            MUSED[] = METH[]
            for j = 1:L[]
                YP1[] = YH[][j]
                r = EL[j]
                for i = 1:N[]
                    YP1[][i] += r * ACOR[][i]
                end
            end
            ICOUNT[] -= 1
            if ICOUNT[] < 0
                methodswitch(dsm, pnorm, pdh, rh)
                if METH[] != MUSED[]
                    rh[] = max(rh[], HMIN[] / abs(H[]))
                    scaleh(rh, pdh)
                    RMAX[] = 10.0
                    endstoda()
                    break
                end
            end
            IALTH[] -= 1
            if IALTH[] == 0
                rhup = Ref(0.0)
                if L[] != LMAX[]
                    YP1[] = YH[][LMAX[]]
                    for i in 1:N[]
                        SAVF[][i] = ACOR[][i] - YP1[][i]
                    end
                    dup = vmnorm(N[], SAVF[], EWT[]) / TESCO[NQ[]][3]
                    exup = 1 / (L[] + 1)
                    rhup[] = 1 / (1.4 * dup ^ exup +0.0000014)
                end
                orderswitch(rhup, dsm, pdh, rh, orderflag)
                if orderflag[] == 0
                    endstoda()
                    break
                end
                if orderflag[] == 1
                    rh[] = max(rh, HMIN[] / abs(H[]))
                    scaleh(rh, pdh)
                    RMAX[] = 10.0
                    break
                end
                if orderflag[] == 2
                    resetcoeff()
                    rh[] = max(rh[], HMIN[] / abs(H[]))
                    scaleh(rh, pdh)
                    RMAX[] = 10
                    endstoda()
                    break
                end
            end
            if IALTH[] > 1 || L[] == LMAX[]
                endstoda()
                break
            end
            YP1[] = YH[][LMAX[]]
            for i in 1:N[]
                YP1[][i] = ACOR[][i]
            end
            endstoda()
            break
        else
            KFLAG[] -= 1
            TN[] = told[]
            for j in NQ[] : -1 : 1
                for i1 in j:NQ[]
                    YP1[] = YH[][i1]
                    YP2[] = YH[][i1+1]
                    for i in 1:N[]
                        YP1[][i] -= YP2[][i]
                    end
                end
            end
            RMAX[] = 2.0
            if abs(H[]) <= HMIN[] * 1.00001
                KFLAG[] = -1
                HOLD[] = H[]
                JSTART[] = 1
                break
            end
            if KFLAG[] > -3
                rhup[] = 0.0
                orderswitch(rhup, dsm, pdh, rh, orderflag)
                if orderflag[] == 1 || orderflag[] == 0
                    if orderflag[] == 0
                        rh[] = min(rh[], 0.2)
                    end
                    rh[] = max(rh[], HMIN[] / abs(H[]))
                    scaleh(rh, pdh)
                end
                if orderflag[] == 2
                    resetcoeff()
                    rh[] = max(rh[], HMIN[] / abs(H[]))
                    scaleh(rh, pdh)
                end
                continue
            else
                if KFLAG[] == -10
                    KFLAG[] = -1
                    HOLD[] = H[]
                    JSTART[] = 1
                    break
                else
                    rh[] = 0.1
                    rh[] = max(HMIN[] / abs(H[]), rh[])
                    H[] *= rh[]
                    YP1[] = YH[][1,:]
                    for i in 1:N[]
                        y[i] = YP1[][i]
                    end
                    prob.f(savf[], y, prob.p, TN[])
                    NFE[] += 1
                    YP1[] = YH[][2,:]
                    for i in 1:N[]
                        YP1[][i] = H[] * SAVF[][i]
                    end
                    IPUP[] = miter[]
                    IALTH[] = 5
                    if NQ[] == 1
                        continue
                    end
                    NQ[] = 1
                    L[] = 2
                    resetcoeff()
                    continue
                end
            end
        end
    end
end

function cfode(meth::Int)
    pc = zeros(13)
    if meth == 1
        ELCO[1][1] = 1.0
		ELCO[1][2] = 1.0
		TESCO[1][1] = 0.0
		TESCO[1][2] = 2.0
		TESCO[2][1] = 1.0
		TESCO[12][3] = 0.0
		pc[1] = 1.0
        rqfac = 1.0
        for NQ[] in 2:12
            rq1fac = rqfac
            rqfac = rqfac / NQ[]
            nqm1 = NQ[] - 1
            fnqm1 = Float64(nqm1)
            nqp1 = NQ[] + 1
            pc[NQ[]] = 0.0
            for i in NQ[] : -1 : 2
                pc[i] = pc[i - 1] + fnqm1 * pc[i]
            end
            pc[1] = fnqm1 * pc[1]
            pint = pc[1]
            xpin = pc[1] / 2.0
            tsign = 1.0
            for i in 2:NQ[]
                tsign = -tsign
                pint += tsign * pc[i] / i
                xpin += tsign * pc[i] / (i + 1)
            end
            ELCO[NQ[]][1] = pint * rq1fac
            ELCO[NQ[]][2] = 1.0
            for i in 2: NQ[]
                ELCO[NQ[]][i + 1] = rq1fac * pc[i] / i
            end
            agamq = rqfac * xpin
            ragq = 1.0 / agamq
            TESCO[NQ[]][2] = ragq
            if NQ[] < 12
                TESCO[nqp1][1] = ragq * rqfac / nqp1
            end
            TESCO[nqm1][3] = ragq
        end
        return
    pc[1] = 1.0
    rq1fac = 1.0
    for NQ[] in 1:5
        fnq = Float64(NQ[])
        nqp1 = NQ[] + 1
        pc[nqp1] = 0
        for i in NQ[] + 1 : -1 : 2
            pc[i] = pc[i - 1] + fnq * pc[i]
        end
        pc[i] *= fnq
        for i = 1:N[]qp1
            ELCO[NQ[]][i] = pc[i] / pc[2]
        end
        ELCO[NQ[]][2] = 1
        TESCO[NQ[]][1] = rq1fac
        TESCO[NQ[]][2] = Float64(nqp1) / ELCO[NQ[]][1]
        TESCO[NQ[]][3] = Float64(NQ[] + 2) / ELCO[NQ[]][1]
        rq1fac /= fnq
    end
    return
end

function scaleh(rh::Ref{Float64}, pdh::Ref{Float64})
    rh[] = min(rh[], RMAX[])
    rh[] = rh[] / max(1, abs(H[] * HMXI[] * rh[]))
    if METH[] ==1 
        IRFLAG[] = 0
        pdh[] = max(abs(H[]) * PDLAST[], 0.000001)
        if rh[] * pdh[] * 1.00001 >= SM1[NQ[]]
            rh[] = SM1[NQ[]] / pdh[]
            IRFLAG[] = 1
        end
    end
    r = 1.0
    for j in 2:L[]
        r *= rh[]
        YP1[] = YH[][j]
        for i = 1:N[]
            YP1[][i] *= r
        end
    end
    H[] *= rh[]
    RC[] *= rh[]
    IALTH[] = L[]
end

function resetcoeff()
    ep1 = ELCO[NQ[], :]
    for i in 1:L[]
        EL[i] = ep1[i]
    end
    RC[] = RC[] * EL[1] / EL0
    EL0 = EL[1]
    CONIT = 0.5 / (NQ[] + 2)
    return
end

function vmnorm(n::Int, v::Vector{Float64}, w::Vector{Float64})
    vm = 0.0
    for i in 1:N[]
        vm = max(vm, abs(v[][i]) * w[i])
    end
    return vm
end

function ewset!(rtol::Ref{Float64}, atol::Ref{Float64}, ycur::Ref{Float64})
    fill!(EWT[], rtol[] * abs(ycur[]) + atol[])
end

function intdy!(t::Float64, k::Int, dky::Vector{Float64}, iflag::Ref{Int})
    iflag[] = 0
    if (k < 0 || k > NQ)
        @warn("[intdy] k = $k illegal\n")
        iflag[] = -1
        return
    end
    tp = TN[] - HU[] - 100 * eps() * (TN[] + HU[])
    if (t - tp) * (t - TN[]) > 0
        @warn("intdy -- t = $t illegal. t not in interval tcur - hu to tcur\n")
        iflag[] = -2
        return
    end
    s = (t - TN[]) / H[]
    c = 1
    for jj in (L[] - k):NQ[]
        c *= jj
    end
    YP1[] = YH[][1,:]
    for i in 1:N[]
        dky[i] = c *YP1[][i]
    end
    for j in (NQ[] -1 : -1 : k)
        jp1 = j + 1
        c = 1
        for jj in jp1 - k : j
            c *= jj
        end
        YP1[] = YH[][jp1]
        for i = 1 : n
            dky[i] = c * YP1[][i] + s *dky[i]
        end
    end
    if k == 0
        return
    end
    r = h ^ (-k)
    for i in 1 : n
        dky[i] *= r
    end
    return
end

function prja(neq::Int, prob)
    NJE[] += 1
    IERPJ[] = 0
    JCUR[] = 1
    hl0 = H[] * EL0[]
    if MITER[] != 2
        @warn("[prja] miter != 2\n")
        return
    else
        fac = vmnorm(N[], SAVF[], EWT[])
        r0 = 1000 * abs(H[]) * eps() * N[] *fac
        if r0 == 0.0
            r0 = 1
        end
        for j in 1:N[]
            yj = y[j] ## y need to be changed
            r = max(sqrt(eps()) * abs(yj), r0/EWT[][j])
            y[j] += r
            fac = -hl0 / r
            prob.f(ACOR[], y, prob.p, TN[])
            for i in 1:N[]
                WM[][i][j] = ACOR[][i] - SAVF[][i] * fac
            end
            y[j] = yj
        end
        NFE[] += N[]
        PDNORM[] = fnorm(N[], WM[], EWT[]) / abs(hl0)
        for i in 1:N[]
            WM[][i][i] += 1.0
        end
        LUresult = lu!(WM[])
        WM[] = LUresult.L
        if 0 in diag(LUresult.U)
            IERPJ[] = 1
        end
        return
    end
end

function fnorm(n::Int, a::Matrix{Float64}, w::Vector{Float64})
    an = 0
    for i in 1:n
        sum = 0
        ap1 = a[i,:]
        for j in 1:n
            sum +=abs(ap1[j]) / w[j]
        end
        an = max(an, sum * w[i])
    end
    return an
end


end # module
