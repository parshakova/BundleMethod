using Convex, ECOS, OSQP


function sc_g_domain(var_x, params)
    # var_x is a public variable for SC problem
    idx = 0
    constraints = Any[]
    M = size(params)[1] - 2 # first and last entry are sell and retail functions
    for i in 2 : M + 1
        m = params[i]["m"]
        n = params[i]["n"]
        # @show i, m, n, idx
        # flow conservation
        push!(constraints, sum(var_x[idx + 1 : idx + m]) == sum(var_x[idx + m + 1: idx + m + n]))
        # upper / lower bounds
        append!(constraints, [var_x[idx + 1 : idx + m + n] <= params[i]["upper_bound"],
                              var_x[idx + 1 : idx + m + n] >= params[i]["lower_bound"]])
        idx += m 
    end
    return constraints 
end


function project_onto_domain(v, size_x, params)
    var_x = Variable(size_x)
    constraints = Constraint[var_x >= 0]
    append!(constraints, sc_g_domain(var_x, params))
    prob = minimize(sumsquares(var_x - v), constraints)
    try
        solve!(prob, ECOS.Optimizer; silent_solver = true)
    catch
        solve!(prob, OSQP.Optimizer; silent_solver = true)
    end
    return evaluate(var_x)
end


function init_feasible_point(size_x, params)
    mid_point = (lb + ub) / 2
    return project_onto_domain(mid_point, size_x, params)
end


function sc_coupling_query(params, v)
    # get subgradient of g
    m1 = params[1]["m"]
    nM = params[end]["n"]
    var_x = Variable(dim)
    obj = dot(params[1]["sale"], var_x[begin : m1]) - dot(params[end]["retail"], var_x[end - nM + 1:end])
    constraints = Constraint[var_x == v]
    append!(constraints, sc_g_domain(var_x, params))
    subgrad_g = constraints[1].dual
    return subgrad_g, evaluate(obj)
end


function sc_query(params, v, idx, norm, mu)
    n = params[idx]["n"]
    m = params[idx]["m"]
    lwb = params[idx]["lower_bound"]
    upb = params[idx]["upper_bound"]
    cap = params[idx]["cap"]
    lin  = params[idx]["lin"]
    quad = params[idx]["quad"]
    dim = params[idx]["dimension"]
    X = Variable(n, m) # nonneg=True)
    u = Variable(dim) # nonneg=True)
    r = Variable(dim)

    ones_m = ones(m, 1)
    ones_n = ones(n, 1)
    constraints = Constraint[ u - r == v, 
                    (X') * ones_n == u[1 : m],
                    X * ones_m == u[m + 1 : end],
                    u <= upb, 
                    u >= lwb, 
                    sum(u[1 : m]) == sum(u[m + 1 : end]),
                    X >= 0, 
                    u >= 0]
    if sum(cap) < Inf
        push!(constraints, X <= cap)
    end

    # @assert size(constraiånts)[0] == 9
    if norm == "l2"
        penalty = 0.5 * mu * sumsquares(r)
    elseif norm == "l1"
        penalty = 0.5 * mu * sum(abs(r))
    end
    F = dot(vec(X), vec(lin)) + 0.5 * dot(vec(square(X)), vec(quad))
    f = F + penalty

    prob = minimize(f, constraints)
    try
        solve!(prob, ECOS.Optimizer; silent_solver = true)
    catch
        solve!(prob, OSQP.Optimizer; silent_solver = true)
    end
    f = evaluate(f)
    q = constraints[1].dual
    # @show idx, prob.status, f, typeof(f)
    return q, f
end

function sc_fval_subgrad_f(params, size_x, v, norm, mu)
    f_val = 0
    subgrad_fis = Any[]
    M = size(params)[1] - 2 # number of SC agents
    idx = 0
    vis = Any[]
    for i in 2 : M + 1
        n = params[i]["n"]
        m = params[i]["m"]
        vi = v[idx + 1 : idx + params[i]["dimension"]]
        qi, fi = sc_query(params, vi, i, norm, mu)
        push!(subgrad_fis, vec(qi))
        push!(vis, vec(vi))
        f_val += fi
        idx += m + n
    end
    
    @assert idx == size_x
    subgrad_f = reduce(append!, subgrad_fis, init=Any[])
    @assert size(subgrad_f)[1] == size_x
    # @assert all(vec(v) .≈ reduce(append!, vis, init=Any[]))

    return subgrad_f, f_val
end

function sc_hval_subgrad(params, size_x, v, norm, mu, grad_g_val)
    subgrad_f, f_val = sc_fval_subgrad_f(params, size_x, v, norm, mu)

    g_val = dot(grad_g_val, v)
    @assert size(subgrad_f)[1] == size_x
    subgrad = subgrad_f + grad_g_val
    h_val = f_val + g_val

    return subgrad, h_val
end 


function test_f_subgrad(params, size_x, norm, mu)
    for i in 1:20
        v = rand(Float64, (size_x, 1))
        v = project_onto_domain(v, size_x, params)
        g_v, f_v = sc_fval_subgrad_f(params, size_x, v, norm, mu)
        for j in 1:5
            a = rand(Float64, (size_x, 1))
            a = project_onto_domain(a, size_x, params)
            g_a, f_a = sc_fval_subgrad_f(params, size_x, a, norm, mu)

            # @show f_v + 1e-7 >= f_a + dot(vec(g_a), vec(v) - vec(a)), f_v + 1e-7, f_a + dot(vec(g_a), vec(v) - vec(a))
            # @show f_a + 1e-7 >= f_v + dot(vec(g_v), vec(a) - vec(v)), f_a + 1e-7, f_v + dot(vec(g_v), vec(a) - vec(v))
            @assert f_v + 1e-7 >= f_a + dot(vec(g_a), vec(v) - vec(a))
            @assert f_a + 1e-7 >= f_v + dot(vec(g_v), vec(a) - vec(v))

        end
    end
end


function test_h_subgrad(params, size_x, norm, mu, grad_g_val)
    for i in 1:20
        v = rand(Float64, (size_x, 1))
        v = project_onto_domain(v, size_x, params)
        g_v, h_v = sc_hval_subgrad(params, size_x, v, norm, mu, grad_g_val)
        for j in 1:5
            a = rand(Float64, (size_x, 1))
            a = project_onto_domain(a, size_x, params)
            g_a, h_a = sc_hval_subgrad(params, size_x, a, norm, mu, grad_g_val)

            # @show h_v + 1e-7 >= h_a + dot(vec(g_a), vec(v) - vec(a)), h_v + 1e-7, h_a + dot(vec(g_a), vec(v) - vec(a))
            # @show h_a + 1e-7 >= h_v + dot(vec(g_v), vec(a) - vec(v)), h_a + 1e-7, h_v + dot(vec(g_v), vec(a) - vec(v))
            @assert h_v + 1e-7 >= h_a + dot(vec(g_a), vec(v) - vec(a))
            @assert h_a + 1e-7 >= h_v + dot(vec(g_v), vec(a) - vec(v))

        end
    end
end

