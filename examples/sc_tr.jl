# Supply chain example taken from OSBDO

using BundleMethod
using Convex, ECOS, OSQP
using Pandas 

using BundleMethod
using SparseArrays
using JuMP
using Ipopt
using Random
using CSV, Tables

const BM = BundleMethod

include("osbdo_funcs.jl")


data = read_pickle("/Users/parshakova.tanya/Documents/Distributed_Optimization/OSBDO_private/examples/supply_chain/sc_data.pickle")
params = data["params"]
lb = data["lower_bound"]
ub = data["upper_bound"]
A = data["A"]
b = data["b"] # linear constraints contain conservation of flow 
grad_g_val = data["grad_g_val"]
h_cvx = data["h_cvx"]
x_cvx = data["x_cvx"]

n = size_x = size(lb)[1] # size of x
N = 1 # number of separable functions in the objective
mu = 50.
norm = "l1"


jl_h_cvx = sc_hval_subgrad(params, size_x, x_cvx, norm, mu, grad_g_val)[2]
@show h_cvx, jl_h_cvx
@assert h_cvx ≈ jl_h_cvx

# feasible starting value
x_init = init_feasible_point(size_x, params)


function evaluate_f(x)
    subgrad, h_val = sc_hval_subgrad(params, size_x, x, norm, mu, grad_g_val)
    subgrads = Dict{Int, SparseVector{Float64}}()
    subgrads[1] = SparseVector(vec(subgrad))
    return [h_val], subgrads
end

# This initializes the trust region bundle method with required arguments.
pm = BM.TrustRegionMethod(n, N, evaluate_f) #, init=x_init)

# Set optimization solver to the internal JuMP.Model
model = BM.get_jump_model(pm)
set_optimizer(model, Ipopt.Optimizer)
set_optimizer_attribute(model, "print_level", 0)
set_optimizer_attribute(model, "max_iter", 10000)

@show size(A), size(b)

# We overwrite the function to have column bounds.
function BM.add_variables!(method::BM.TrustRegionMethod)
    bundle = BM.get_model(method)
    model = BM.get_model(bundle)
    @variable(model, vec(lb)[i] <= x[i=1:bundle.n] <= vec(ub)[i])# x[i=1:bundle.n])
    @constraint(model, c1, A * x .== vec(b))
    # @constraint(model, c2, vec(lb) .<= x .<= vec(ub))
    # @constraint(model, c2, vec(lb)[i] <= x[i=1:bundle.n] <= vec(ub)[i=1:bundle.n])
    @variable(model, θ[j=1:bundle.N])
end

# This builds the bundle model.
BM.build_bundle_model!(pm)

# This runs the bundle method.
BM.run!(pm)

@show BM.get_objective_value(pm)
@show h_cvx

CSV.write("/Users/parshakova.tanya/Documents/Distributed_Optimization/BundleMethod/examples/jl_tr_fx0.csv",  
    Tables.table(vec(pm.all_fx0)), writeheader=false)
