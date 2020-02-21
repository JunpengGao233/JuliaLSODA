module LSODA

using LinearAlgebra
# idamax-> LinearAlgebra.BLAS.iamax
# dscal -> LinearAlgebra.rmul!
# ddot  -> LinearAlgebra.BLAS.dot
# daxpy -> LinearAlgebra.axpy!
# dgesl ->
# dgefa -> LU
using Reexport: @reexport
using Parameters: @unpack, @with_kw
@reexport using DiffEqBase

mutable struct JLsoda_f{::Float64, ::Float64, ::Float64, ::Any}

mutable struct JVIntegrator{Alg,uType,tType,uEltype,solType,Rtol,Atol,F,P} <: DiffEqBase.AbstractODEIntegrator{Alg,true,uType,tType}
    sol::solType
 #   opts::JVOptions{Atol,Rtol}
    zn::NTuple{L_MAX,uType} # Nordsieck history array

    # vectors with length `length(u0)`
    ewt::uType # error weight vector
    u::uType
    acor::uType
    tempv::uType
    ftemp::uType

    # step data
    q::Int                        # current order
    qprime::Int                   # order to be used on the next step {q-1, q, q+1}
    qwait::Int                    # number of steps to wait before order change
    #L::Int                        # L = q+1
    dt::tType                     # current step size
    dtprime::tType                # next step size
    eta::tType                    # eta = dtprime / dt
    dtscale::tType                # the step size information in `zn`
    t::tType                      # current time
    tau::NTuple{L_MAX,tType}      # tuple of previous `q+1` successful step sizes
    tq::NTuple{NUM_TESTS,tType}   # tuple of test quantities
    coeff::NTuple{L_MAX,uEltype}  # coefficients of l(x)
    rl2::uEltype                  # 1/l[2]
    gamma::uEltype                # gamma = h * rl2
    gammap::uEltype               # `gamma` at the last setup call
    gamrat::uEltype               # gamma/gammap
    crate::uEltype                # estimated corrector convergence rate
    acnrm::uEltype                # | acor | wrms
    mnewt::Int                    # Newton iteration counter

    # Limits
    qmax::Int                     # q <= qmax
    mxstep::Int                   # maximum number of internal steps for one user call
    maxcor::Int                   # maximum number of `nlsolve`
    mxhnil::Int                   # maximum number of warning messages issued to the
                                  # user that `t + h == t` for the next internal step

    dtmin::tType                  # |h| >= hmin
    dtmax_inv::tType              # |h| <= 1/hmax_inv
    etamax::tType                 # eta <= etamax

    # counters
    nst::Int                      # number of internal steps taken
    nfe::Int                      # number of f calls
    ncfn::Int                     # number of corrector convergence failures
    netf::Int                     # number of error test failures
    nni::Int                      # number of Newton iterations performed
    nsetups::Int                  # number of setup calls
    nhnil::Int                    # number of messages issued to the user that
                                  # `t + h == t` for the next iternal step
    lrw::Int                      # number of real words in CVODE work vectors
    liw::Int                      # no. of integer words in CVODE work vectors

    # saved vales
    qu::Int                       # last successful q value used
    nstlp::Int                    # step number of last setup call
    dtu::tType                    # last successful h value used
    saved_tq5::tType              # saved value of tq[5]
    jcur::Bool                    # Is the Jacobian info used by
                                  # linear solver current?
    tolsf::Float64                # tolerance scale factor
    setupNonNull::Bool            # Does setup do something?

    # Arrays for Optional Input and Optional Output

    #long int *cv_iopt::Int  # long int optional input, output */
    #real     *cv_ropt::Int  # real optional input, output     */
    function JVIntegrator(prob::ODEProblem, ::solType, ::Alg, ::Rtol, ::Atol) where {solType,Alg,Rtol,Atol}
        @unpack f, u0, tspan, p = prob
        obj = new{Alg,typeof(u0),eltype(tspan),eltype(u0),solType,Rtol,Atol,typeof(f),typeof(p)}()
        return obj
    end
end

'
@with_kw mutable struct JLsodaInit{F,K}
    JLsoda_f::F
    neq::int
    y::Float64
    t::Float64
    tout::Float64
    itol::Int
    rtol::Float64
    atol::Float64
    itask::Int
    istate::Int
    iopt::Int
    jt::Int
    #iwork
    #rwork
    _data::K
    illin = 0
'
function Terminate!(JLsoda::JLsodaInit)
    if JLsoda.illin == 5
        error("[lsoda] repeated occurrence of illegal input. run aborted.. apparent infinite loop\n")
    else
        illin += 1
        Integrator.istate = -3
    end
end

function Terminate2!(JLsoda::JLsodaInit, y::Float64, t::Float64)
    yp1 = yh[1]
    i = 1
    for i <= n
        y[i] = yp1[i]
        i += 1
    end
    JLsoda.t = tn
    illin = 0
    #TO DO t? tn? freevectors()
end

function successreturn(JLsoda::JLsodaInit, ihit::Int, tcrit::Float64)
    yp = yh[1]
    i = 1
    for i <= n
        y[i] = yp[i]
    end
    JLsoda.t = tn
    if (JLsoda.itask == 4 || JLsoda.itask == 5)
        if ihit
            JLsoda.t = tcrit
        end
    end
    JLsoda.istate = 2
    JLsoda.illin = 0
end

function 


function prja!(neq::Int, y::Float64, f::JLsoda, data)
#TO DO
end
