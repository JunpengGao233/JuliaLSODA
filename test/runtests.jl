using JuliaLSODA, Test

@testset "LSODA Example" begin
    function fex(du, u, p, t)
        du[1] = 1e4 * u[2] * u[3] - 0.04e0 * u[1]
        du[3] = 3e7 * u[2] * u[2]
        du[2] = - (du[1] + du[3])
    end

    prob = ODEProblem(fex, [1.0, 0, 0], (0.0,0.4e0))
    sol2 = solve(prob, LSODA())
    reference = [9.8517230200179651e-01, 3.3863992446169744e-05, 1.4793834005757239e-02]
#    @show sol2
    #@test prob.u0 ≈ reference rtol = 1e-15
end

@testset "Rober" begin
    function rober(du,u,p,t)
      y₁,y₂,y₃ = u
      k₁,k₂,k₃ = p
      du[1] = -k₁*y₁+k₃*y₂*y₃
      du[2] =  k₁*y₁-k₂*y₂^2-k₃*y₂*y₃
      du[3] =  k₂*y₂^2
      nothing
    end

    prob = ODEProblem(rober,[1.0,0.0,0.0],(0.0,1e5),(0.04,3e7,1e4))
    sol2 = solve(prob, LSODA())
end

@testset "Linear ODE" begin
    function f(du,u,p,t)
        du[1] = 1.01*u[1]
    end
    u0=[1/2]
    tspan = (0.0,1.0)
    prob = ODEProblem(f,u0,tspan)
end

@testset "Pendulum" begin
    l = 1.0                             # length [m]
    m = 1.0                             # mass[m]
    g = 9.81                            # gravitational acceleration [m/s²]

    function pendulum!(du,u,p,t)
        du[1] = u[2]                    # θ'(t) = ω(t)
        du[2] = -3g/(2l)*sin(u[1]) + 3/(m*l^2)*p(t) # ω'(t) = -3g/(2l) sin θ(t) + 3/(ml^2)M(t)
    end

    θ₀ = 0.01                           # initial angular deflection [rad]
    ω₀ = 0.0                            # initial angular velocity [rad/s]
    u₀ = [θ₀, ω₀]                       # initial state vector
    tspan = (0.0,10.0)                  # time interval

    M = t->0.1sin(t)                    # external torque [Nm]

    prob = ODEProblem(pendulum!,u₀,tspan,M)
    sol = solve(prob, LSODA())
end
