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
const MORD = [12, 5]
const SM1 = [0.5, 0.575, 0.55, 0.45, 0.35, 0.25, 0.2, 0.15, 0.1, 0.075, 0.05, 0.025]

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

const EL = zeros(13)
const ELCO = zeros(12, 13)
const TESCO = zeros(12, 3)
const CM1 = zeros(12)
const CM2 = zeros(5)
count1 = 0
@defconsts [YH, WM] Ref{Matrix{Float64}}(zeros(1,2))
@defconsts [EWT, SAVF, ACOR] Ref{Vector{Float64}}(zeros(2))

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
    YP1 = @view YH[][1,:]
    for i in 1:N[]
        y[i] = YP1[i]
    end
    t[] = TN[]
    ILLIN[] = 0
    return
end

function successreturn!(y::Vector{Float64}, t::Ref{Float64}, itask::Int, ihit::Int, icrit::Float64, istate::Ref{Int})
    YP1 = @view YH[][1,:]
    for i in 1:N[]
        y[i] = YP1[i]
    end
    t[] = TN[]
    if itask == 4 || itask == 5
        if bool(ihit)
            t = tcrit
        end
    end
    istate[] = 2
    ILLIN[] = 0
    return
end

function DiffEqBase.__solve(prob::ODEProblem{uType,tType,true}, ::LSODA;
                            itask::Int=1, istate::Ref{Int}=Ref(1), iopt::Bool=false,
                            tout=prob.tspan[end], rtol=Ref(1e-4), atol=Ref(1e-6),
                            tcrit#=tstop=#=nothing) where {uType,tType}
    mxstp0 = 500
    mxhnl0 = 10
    iflag = Ref(0)
    if istate[] < 1 || istate[] > 3
       @warn("[lsoda] illegal istate = $istate\n")
       terminate!(istate)
    end
    if itask < 1 || itask > 5
        @warn("[lsoda] illegal itask = $itask\n")
        terminate!(istate)
    end
    if (INIT[] == 0 && (istate[] == 2 || istate[] == 3))
        @warn("[lsoda] illegal itask = $itask\n")
        terminate!(istate)
    end
    if (INIT[] == 0 && (istate[] ==2 || istate[] == 3))
        @warn("[lsoda] istate > 1 but lsoda not initialized")
        terminate!(istate)
    end

    t = Ref(first(prob.tspan))
    neq = length(prob.u0)
    global YP1,YP2
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
    if istate[] == 1 || istate[] == 3
        ntrep = 0
        if neq <= 0
            @warn("[lsoda] neq = %d is less than 1\n", neq)
            terminate!(istate)
            return
        end
        if istate[] == 3 && neq > N[]
            @warn("[lsoda] istate = 3 and neq increased\n")
            terminate!(istate)
            return
        end
        N[] = neq
    end

    if iopt == false
        IXPR[] = 0
		MXSTEP[] = mxstp0
		MXHNIL[] = mxhnl0
		HMXI[] = 0.0
		HMIN[] = 0.0
		if (istate[] == 1)
            h0 = 0.0
			MXORDN[] = MORD[1]
            MXORDS[] = MORD[2]
        end
    #TODO iopt == true
    end
    (istate[] == 3) && (JSTART[] = -1)

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
        METH[] = 2
        g_nyh = N[]
        nyh = N[]
        g_lenyh = 1 + max(MXORDN[], MXORDS[])
        lenyh = 1 + max(MXORDN[], MXORDS[])
        YH[] = zeros(lenyh, nyh)
        WM[] = zeros(nyh, nyh)
        EWT[] = zeros(nyh)
        SAVF[] = zeros(nyh)
        ACOR[] = zeros(nyh)

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
        @views prob.f(YH[][2,:], y, prob.p, t[])
        NFE[] = 1
        YP1 = @view YH[][1,:]
        for i in 1:N[]
            YP1[i] = y[i]
        end
        NQ[] = 1
        H[] = 1.0
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
                return
            end
            tol = rtol[]
            if tol <= 0.0
                for i in 1:N[]
                    ayi = abs(y[i])
                    if ayi != 0
                        tol = max(tol[], atol[]/ayi)
                    end
                end
            end
            tol = max(tol, 100 * eps())
            tol = min(tol, 0.001)
            sum = vmnorm(N[], YH[][2,:], EWT[])
            sum = 1 / (tol * w0 * w0) + tol * sum * sum
            h0 = 1 / sqrt(sum)
            h0 = min(h0, tdist)
            # h0 = h0 * ((tout - t[] >= 0) ? 1 : -1)
            h0 = copysign(h0, tout - t[])
        end
        rh = abs(h0) * HMXI[]
        rh > 1 && (h0 /= rh)
        H[] = h0
        YP1 = @view YH[][2,:]
        for i in 1:N[]
            YP1[i] *= h0
        end
    end

    ###Block d###
    if (istate[] == 2 || istate[] == 3)
        NSLAST[] = NST[]
        if itask == 1
            if ((TN[] - tout) * H[] >= 0)
                intdy!(tout, 0, y, iflag)
                if iflag[] != 0
                    @warn("[lsoda] trouble from intdy, itask = $itask, tout = $tout\n")
                    terminate!(istate)
                    return
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
            hmx = abs(TN[]) + abs(H[])
            ihit = abs(TN[] - tcrit) <= (100 * eps() *hmx)
            if Bool(ihit)
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
    local count1 = 0
    local count2 = 0
    local count3 = 0
    while true
        if (istate[] != 1 || NST[] != 0)
            if ((NST[] - NSLAST[]) >= MXSTEP[])
                @warn("[lsoda] $(MXSTEP[]) steps taken before reaching tout\n")
                istate[] = -1
                terminate2!(y, t)
                return
            end
            ewset!(rtol, atol, YH[][1,:])
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
        count1+=1
        stoda(neq, prob)
        #Block f#
        if KFLAG[] == 0
            INIT[] = 1
            if METH[] != MUSED[]
                TSW[] = TN[]
                MAXORD[] = MXORDN[]
                if METH[] == 2
                    MAXORD[] = MXORDS[]
                end
                JSTART[] = -1
                if Bool(IXPR[])
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
                intdy!(tout, 0, y, iflag)
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
                if (TN[] - tout) * H[] >= 0
                    successreturn(y, t, itask, ihit, tcrit, istate)
                    return
                end
                continue
            end
            if itask == 4
                if (TN[] - tout) * H[] >= 0
                    intdy!(tout, 0, y, iflag)
                    t[] = tout
                    istate[] =2
                    ILLIN[] = 0
                    return
                else
                    hmx = abs(TN[]) + abs(H[])
                    ihit = abs(tn - tcrit) <= 100 * eps() *hmx
                    if Bool(ihit)
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
                sizing = abs(ACOR[][i]) * EWT[][i]
                if big < sizing
                    big = sizing
                    IMXER[] = 1
                end
            end
            terminate2!(y, t)
            return
        end
    end
    return
end

function stoda(neq::Int, prob)
    y = prob.u0
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
            CM2[i] = TESCO[i, 2] * ELCO[i, i + 1]
        end
        cfode(1)
        for i in 1:12
            CM1[i] = TESCO[i, 2] * ELCO[i, i + 1]
        end
        resetcoeff()
    end
    if JSTART[] == -1
        IPUP[] = MITER[]
        LMAX[] = MAXORD[] + 1
        if IALTH[] == 1
            IALTH[] = 2
        end
        if METH[] != MUSED[]
            cfode(METH[])
            IALTH[] = L[]
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
    while true
        local pnorm
        while true
            if abs(RC[] - 1) > CCMAX[]
                IPUP[] = MITER[]
            end
            if NST[] >= NSLP[] + MSBP[]
                IPUP[] = MITER[]
            end
            TN[] += H[]
            for j in NQ[] : -1 : 1
                for i1 in j : NQ[]
                    YP1 = @view YH[][i1, :]
                    YP2 = @view YH[][i1 + 1, :]
                    for i in 1:N[]
                        YP1[i] += YP2[i]
                    end
                end
            end
            pnorm = vmnorm(N[], YH[][1,:], EWT[])
            #Ref??? y
            println("correc before$(H[]) \n")
            correction(neq, prob, corflag, pnorm, del, delp, told, ncf, rh, m)
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
        if m[] == 0
            dsm = del[] / TESCO[NQ[], 2]
        end
        if m[] > 0
            dsm = vmnorm(N[], ACOR[], EWT[]) / TESCO[NQ[],2]
        end
        if dsm <= 1.0
            KFLAG[] = 0
            NST[] += 1
            HU[] = H[]
            NQU[] = NQ[]
            MUSED[] = METH[]
            for j = 1:L[]
                YP1 = @view YH[][j, :]
                r = EL[j]
                for i = 1:N[]
                    YP1[i] += r * ACOR[][i]
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
                    YP1 = @view YH[][LMAX[], :]
                    for i in 1:N[]
                        SAVF[][i] = ACOR[][i] - YP1[i]
                    end
                    dup = vmnorm(N[], SAVF[], EWT[]) / TESCO[NQ[], 3]
                    exup = 1 / (L[] + 1)
                    rhup[] = 1 / (1.4 * dup ^ exup +0.0000014)
                end
                orderswitch(rhup, dsm, pdh, rh, orderflag)
                if orderflag[] == 0
                    endstoda()
                    break
                end
                if orderflag[] == 1
                    rh[] = max(rh[], HMIN[] / abs(H[]))
                    scaleh(rh, pdh)
                    RMAX[] = 10.0
                    endstoda()
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
            YP1 = @view YH[][LMAX[], :]
            for i in 1:N[]
                YP1[i] = ACOR[][i]
            end
            endstoda()
            break
        else
            KFLAG[] -= 1
            TN[] = told[]
            for j in NQ[] : -1 : 1
                for i1 in j:NQ[]
                    YP1 = @view YH[][i1, :]
                    YP2 = @view YH[][i1+1, :]
                    for i in 1:N[]
                        YP1[i] -= YP2[i]
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
                rhup = Ref(0.0)
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
                    YP1 = @view YH[][1,:]
                    for i in 1:N[]
                        y[i] = YP1[i]
                    end
                    @views prob.f(SAVF[], y, prob.p, TN[])
                    NFE[] += 1
                    YP1 = @view YH[][2,:]
                    for i in 1:N[]
                        YP1[i] = H[] * SAVF[][i]
                    end
                    IPUP[] = MITER[]
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
    pc = zeros(12)
    if meth == 1
        ELCO[1, 1] = 1.0
		ELCO[1, 2] = 1.0
		TESCO[1, 1] = 0.0
		TESCO[1, 2] = 2.0
		TESCO[2, 1] = 1.0
		TESCO[12, 3] = 0.0
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
            ELCO[nq, 1] = pint * rq1fac
            ELCO[nq, 2] = 1.0
            for i in 2: nq
                ELCO[nq, i + 1] = rq1fac * pc[i] / i
            end
            agamq = rqfac * xpin
            ragq = 1.0 / agamq
            TESCO[nq, 2] = ragq
            if nq < 12
                TESCO[nqp1, 1] = ragq * rqfac / nqp1
            end
            TESCO[nqm1, 3] = ragq
        end
        return
    end
    pc[1] = 1.0
    rq1fac = 1.0
    for nq in 1:5
        fnq = Float64(nq)
        nqp1 = nq + 1
        pc[nqp1] = 0
        for i in nq  : -1 : 2
            pc[i] = pc[i - 1] + fnq * pc[i]
        end
        pc[1] *= fnq
        for i = 1:nqp1
            ELCO[nq, i] = pc[i] / pc[2]
        end
        ELCO[nq, 2] = 1.0
        TESCO[nq, 1] = rq1fac
        TESCO[nq, 2] = Float64(nqp1) / ELCO[nq, 1]
        TESCO[nq, 3] = Float64(nq + 2) / ELCO[nq, 1]
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
        YP1 = @view YH[][j, :]
        for i = 1:N[]
            YP1[i] *= r
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
    RC[] = RC[] * EL[1] / EL0[]
    EL0[] = EL[1]
    CONIT = 0.5 / (NQ[] + 2)
    return
end

function vmnorm(n::Int, v::Vector{Float64}, w::Vector{Float64})
    vm = 0.0
    for i in 1:n
        vm = max(vm, abs(v[i]) * w[i])
    end
    return vm
end

function ewset!(rtol::Ref{Float64}, atol::Ref{Float64}, ycur::Vector{Float64})
    for i in 1:N[]
        EWT[][i] = rtol[] * abs(ycur[i]) + atol[]
    end
end

function intdy!(t::Float64, k::Int, dky::Vector{Float64}, iflag::Ref{Int})
    iflag[] = 0
    if (k < 0 || k > NQ[])
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
    YP1 = @view YH[][L[],:]
    for i in 1:N[]
        dky[i] = c * YP1[i]
    end
    for j in (NQ[] -1 : -1 : k)
        jp1 = j + 1
        c = 1
        for jj in jp1 - k : j
            c *= jj
        end
        YP1 = @view YH[][jp1, :]
        for i = 1 : N[]
            dky[i] = c * YP1[i] + s *dky[i]
        end
    end
    if k == 0
        return
    end
    r = h ^ (-k)
    for i in 1 : N[]
        dky[i] *= r
    end
    return
end

function prja(neq::Int, prob)
    y = prob.u0
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
            r0 = 1.0
        end
        for j in 1:N[]
            yj = y[j] ## y need to be changed
            r = max(sqrt(eps()) * abs(yj), r0/EWT[][j])
            y[j] += r
            fac = -hl0 / r
            @views prob.f(ACOR[], y, prob.p, TN[])
            for i in 1:N[]
                WM[][i, j] = (ACOR[][i] - SAVF[][i]) * fac
            end
            y[j] = yj
        end
        NFE[] += N[]
        PDNORM[] = fnorm(N[], WM[], EWT[]) / abs(hl0)
        for i in 1:N[]
            WM[][i, i] += 1.0
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
            sum += abs(ap1[j]) / w[j]
        end
        an = max(an, sum * w[i])
    end
    return an
end

function correction(neq::Int, prob::ODEProblem, corflag::Ref{Int}, pnorm::Float64,
    del::Ref{Float64}, delp::Ref{Float64}, told::Ref{Float64}, ncf::Ref{Int}, rh::Ref{Float64},
    m::Ref{Int})
    y = prob.u0
    m[] = 0
    corflag[] = 0
    rate = 0.0
    del[] = 0
    YP1 = @view YH[][1, :]
    for i in 1:N[]
        y[i] = YP1[i]
    end
    @views prob.f(SAVF[], y, prob.p, TN[])
    NFE[] += 1
    #@show SAVF[]
    while true
        if m[] == 0
            if IPUP[] > 0
                prja(neq, prob)
                IPUP[] = 0
                RC[] = 1.0
                NSLP[] = NST[]
                CRATE[] = 0.7
                if IERPJ[] != 0
                    corfailure(told, rh, ncf, corflag)
                    println("corflag afterIERPJ corfailure %d\n",corflag)
                    return
                end
            end
            for i in 1:N[]
                ACOR[][i] = 0.0
            end
        end
        if MITER[] == 0
            YP1 = @view YH[][2, :]
            for i in 1:N[]
                SAVF[][i] = H[] * SAVF[][i] - YP1[i]
                y[i] = SAVF[][i] - ACOR[][i]
            end
            del[] = vmnorm(N[], y, EWT[])
            #@show y
            YP1 = @view YH[][1, :]
            for i = 1:N[]
                y[i] = YP1[i] + EL[1] * SAVF[][i]
                ACOR[][i] = SAVF[][i]
            end
        else
            YP1 = @view YH[][2, :]
            for i in 1:N[]
                y[i] = H[] * SAVF[][i] - (YP1[i] + ACOR[][i])
            end
            y[:] = solsy(y)
            del[] = vmnorm(N[], y, EWT[])
            YP1 = @view YH[][1, :]
            for i in 1:N[]
                ACOR[][i] += y[i]
                y[i] = YP1[i] + EL[1] *ACOR[][i]
            end
        end
        if del[] <= 100 *pnorm *eps()
            break
        end
        if m[] != 0 || METH[] != 1
            if m[] != 0
                rm = 1024
                if del[] <= (1024 * delp[])
                    rm = del[] / delp[]
                end
                rate = max(rate, rm)
                CRATE[] = max(0.2 * CRATE[], rm)
            end
            dcon = del[] * min(1.0, 1.5 * CRATE[]) / (TESCO[NQ[], 2] * CONIT[])
            if dcon <= 1.0
                PDEST[] = max(PDEST[], rate / abs(H[] * EL[1]))
                if PDEST[] != 0
                    PDLAST[] = PDEST[]
                end
                break
            end
        end
        m[] += 1
        #@show m[]
        #@show delp[]
        if m[] == MAXCOR[] || (m[] >= 2 && del[] > 2 * delp[])
            #@show JCUR[]
            #@show MITER[]
            if MITER[] == 0 || JCUR[] == 1
                corfailure(told, rh, ncf, corflag)
                return
            end
            IPUP[] = MITER[]
            m[] = 0
            rate = 0
            del[] = 0
            YP1 = @view YH[][1, :]
            for i in 1:N[]
                y[i] = YP1[i]
            end
            @views prob.f(SAVF[], y, prob.p, TN[])
            NFE[] += 1
        else
            delp[] = del[]
            @views prob.f(SAVF[], y, prob.p, TN[])
            NFE[] += 1
        end
    end
end

function corfailure(told::Ref{Float64}, rh::Ref{Float64}, ncf::Ref{Int},
    corflag::Ref{Int})
    ncf[] += 1
    RMAX[] = 2.0
    TN[] = told[]
    for j in NQ[] : -1 : 1
        for i1 in j : NQ[]
            YP1 = @view YH[][i1, :]
            YP2 = @view YH[][i1 + 1, :]
            for i in 1 : N[]
                YP1[i] -= YP2[i]
            end
        end
    end
    if (abs(H[]) <= HMIN[] * 1.00001) || (ncf[] == MXNCF[])
        corflag[] = 2
        return
    end
    corflag[] = 1
    rh[] = 0.25
    IPUP[] = MITER[]
end

function solsy(y::Vector{Float64})
    IERSL[] = 0
    if MITER[] != 2
        print("solsy -- miter != 2\n")
        return
    end
    if MITER[] == 2
        return WM[] \ y
    end
end

function methodswitch(dsm::Float64, pnorm::Float64, pdh::Ref{Float64}, rh::Ref{Float64})
    if (METH[] == 1)
		if (NQ[] > 5)
            return
        end
		if (dsm <= (100 * pnorm * eps()) || PDEST[] == 0)
			if (IRFLAG[] == 0)
                return
            end
			rh2 = 2.0
            nqm2 = min(NQ[], MXORDS[])
		else
			exsm = 1 / L[]
			rh1 = 1 / (1.2 * (dsm ^ exsm) + 0.0000012)
			rh1it = 2 * rh1
			pdh[] = PDLAST[] * abs(H[])
			if (pdh[] * rh1) > 0.00001
                rh1it = SM1[NQ[]] / pdh[]
            end
			rh1 = min(rh1, rh1it)
			if (NQ[] > MXORDS[])
				nqm2 = MXORDS[]
				lm2 = MXORDS[] + 1
				exm2 = 1 / lm2
				lm2p1 = lm2 + 1
				dm2 = vmnorm(N[], YH[][lm2p1,:], EWT[]) / CM2[MXORDS[]]
				rh2 = 1 / (1.2 * (dm2 ^ exm2) + 0.0000012)
			else
				dm2 = dsm * (CM1[NQ[]] / CM2[NQ[]])
				rh2 = 1 / (1.2 * (dm2 ^ exsm) + 0.0000012)
				nqm2 = NQ[]
            end
			if (rh2 < RATIO[] * rh1)
				return
            end
        end

        rh[] = rh2
        ICOUNT[] = 20
        METH[] = 2
        MITER[] = JTYP[]
        PDLAST[] = 0.0
        NQ[] = nqm2
        L[] = NQ[] + 1
        return
    end

    exsm = 1 / L[]
    if MXORDN[] < NQ[]
        nqm1 = MXORDN[]
        lm1 = MXORDN[] + 1
        exm1 = 1 / lm1
        lm1p1 = lm1 + 1
        dm1 = vmnorm(N[], YH[][lm1p1,:], EWT[]) / CM1[MXORDN]
        rh1 = 1 / (1.2 * (dm1 ^ exm1) + 0.0000012)
    else
        dm1 = dsm * ((CM2[NQ[]] / CM1[NQ[]]))
        rh1 = 1 / (1.2 * (dm1 ^ exsm) + 0.0000012)
        nqm1 = NQ[]
        exm1 = exsm
    end
    rh1it = 2 * rh1
    pdh[] = PDNORM[] * abs(H[])
    if (pdh[] * rh1) > 0.00001
        rh1it = SM1[nqm1] / pdh[]
    end
    rh1 = min(rh1, rh1it)
    rh2 = 1 / (1.2 * (dsm ^ exsm) + 0.0000012)
    if (rh1 * RATIO[]) < (5 * rh2)
        return
    end
    alpha = max(0.001, rh1)
    dm1 *= alpha ^ exm1
    if dm1 <= 1000 * eps() * pnorm
        return
    end
    rh[] = rh1
    ICOUNT[] = 20
    METH[] = 1
    MITER[] = 0
    PDLAST[] = 0.0
    NQ[] = nqm1
    L[] = NQ[] + 1
end

function endstoda()
    r = 1 / TESCO[NQU[], 2]
    for i in 1:N[]
        ACOR[][i] *= r
    end
    HOLD[] = H[]
    JSTART[] = 1
    return
end


function orderswitch(rhup::Ref{Float64}, dsm::Float64, pdh::Ref{Float64}, rh::Ref{Float64}, orderflag::Ref{Int})
    orderflag[] = 0
    exsm = 1 / L[]
    rhsm = 1 / (1.2 * (dsm ^ exsm) + 0.0000012)
    rhdn = 0
    if NQ[] != 1
        ddn = vmnorm(N[], YH[][L[], :], EWT[]) / TESCO[NQ[], 1]
        exdn = 1 / NQ[]
        rhdn = 1 / (1.3 * (ddn ^ exdn) + 0.0000013)
    end

    if METH[] == 1
        pdh[] = max(abs(H[]) * PDLAST[], 0.000001)
        if L[] < LMAX[]
            rhup[] = min(rhup[], SM1[L[]] / pdh[])
        end
        rhsm = min(rhsm, SM1[NQ[]] / pdh[])
        if NQ[] > 1
            rhdn = min(rhdn, SM1[NQ[] - 1] / pdh[])
        end
        PDEST[] = 0
    end
    if rhsm >= rhup[]
        if rhsm >= rhdn
            newq = NQ[]
            rh[] = rhsm
        else
            newq = NQ[] - 1
            rh[] = rhdn
            if KFLAG[] < 0 && rh[] > 1
                rh[] = 1
            end
        end
    else
        if rhup[] <= rhdn
            newq = NQ[] - 1
            rh[] = rhdn
            if KFLAG[] < 0 && rh[] > 1
                rh[] = 1
            end
        else
            rh[] = rhup[]
            if rh[] >= 1.1
                r = EL[L[]] / L[]
                NQ[] = L[]
                L[] = NQ[] + 1
                YP1 = @view YH[][L[],:]
                for i in 1:N[]
                    YP1[i] = ACOR[][i] * r
                end
                orderflag[] = 2
                return
            else
                IALTH[] = 3
                return
            end
        end
    end
    if METH[] == 1
        if rh[] * pdh[] * 1.00001 < SM1[newq]
            if KFLAG[] == 0 && rh[] < 1.1
                IALTH[] = 3
                return
            end
        else
            if KFLAG[] == 0 && rh[] < 1.1
                IALTH[] = 3
                return
            end
        end
    end
    if KFLAG[] <= -2
        rh[] = min(rh[], 0.2)
    end
    if newq == NQ[]
        orderflag[] = 1
        return
    end
    NQ[] = newq
    L[] = NQ[] + 1
    orderflag[] = 2
    return
end

function fex(du, u, p, t)
    du[1] = 1e4 * u[2] * u[3] - 0.04e0 * u[1]
    du[3] = 3e7 * u[2] * u[2]
    du[2] = - (du[1] + du[3])
end

prob = ODEProblem(fex, [1.0, 0, 0], (0.0,0.4E0))
sol2 = solve(prob, LSODA())
end  #module