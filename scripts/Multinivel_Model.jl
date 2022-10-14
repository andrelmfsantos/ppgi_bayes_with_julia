# Pacotes necessários
import Pkg
#Pkg.add("StatsPlots")
#Pkg.add("Distributions")
#Pkg.add("LaTeXStrings")
#Pkg.add("DataFrames")
#Pkg.add("CSV")
#Pkg.add("Plots")

# Carregar pacotes
using StatsPlots, Distributions, LaTeXStrings, DataFrames, CSV

using DataFrames, CSV, HTTP

url = "https://raw.githubusercontent.com/storopoli/Bayesian-Julia/master/datasets/cheese.csv"
cheese = CSV.read(HTTP.get(url).body, DataFrame)
println(cheese)
describe(cheese)

for c in unique(cheese[:, :cheese])
    cheese[:, "cheese_$c"] = ifelse.(cheese[:, :cheese] .== c, 1, 0)
end

cheese[:, :background_int] = map(cheese[:, :background]) do b
    b == "rural" ? 1 :
    b == "urban" ? 2 : missing
end

first(cheese, 5)

X = Matrix(select(cheese, Between(:cheese_A, :cheese_D)));
y = cheese[:, :y];
idx = cheese[:, :background_int];

# Instalar pacotes necessários
#Pkg.add("Turing")
#Pkg.add("StatsBase")

using Turing
using LinearAlgebra: I
using Statistics: mean, std
using Random: seed!
seed!(123)

# varying_intercept
@model function varying_intercept(X, idx, y; n_gr=length(unique(idx)), predictors=size(X, 2))
    #priors
    α ~ Normal(mean(y), 2.5 * std(y))       # population-level intercept
    β ~ filldist(Normal(0, 2), predictors)  # population-level coefficients
    σ ~ Exponential(1 / std(y))             # residual SD
    #prior for variance of random intercepts
    #usually requires thoughtful specification
    τ ~ truncated(Cauchy(0, 2); lower=0)    # group-level SDs intercepts
    αⱼ ~ filldist(Normal(0, τ), n_gr)       # group-level intercepts

    #likelihood
    ŷ = α .+ X * β .+ αⱼ[idx]
    y ~ MvNormal(ŷ, σ^2 * I)
end;

# varying_intercept
model_intercept = varying_intercept(X, idx, y)
chain_intercept = sample(model_intercept, NUTS(), MCMCThreads(), 1_000, 4)
summarystats(chain_intercept) |> DataFrame |> println

# varying_slope
@model function varying_slope(X, idx, y; n_gr=length(unique(idx)), predictors=size(X, 2))
    #priors
    α ~ Normal(mean(y), 2.5 * std(y))                    # population-level intercept
    σ ~ Exponential(1 / std(y))                          # residual SD
    #prior for variance of random slopes
    #usually requires thoughtful specification
    τ ~ filldist(truncated(Cauchy(0, 2); lower=0), n_gr) # group-level slopes SDs
    βⱼ ~ filldist(Normal(0, 1), predictors, n_gr)        # group-level standard normal slopes

    #likelihood
    ŷ = α .+ X * βⱼ * τ
    y ~ MvNormal(ŷ, σ^2 * I)
end;

model_slope = varying_slope(X, idx, y)
chain_slope = sample(model_slope, NUTS(), MCMCThreads(), 1_000, 4)
summarystats(chain_slope) |> DataFrame |> println

# varying_intercept_slope
@model function varying_intercept_slope(X, idx, y; n_gr=length(unique(idx)), predictors=size(X, 2))
    #priors
    α ~ Normal(mean(y), 2.5 * std(y))                     # population-level intercept
    σ ~ Exponential(1 / std(y))                           # residual SD
    #prior for variance of random intercepts and slopes
    #usually requires thoughtful specification
    τₐ ~ truncated(Cauchy(0, 2); lower=0)                 # group-level SDs intercepts
    τᵦ ~ filldist(truncated(Cauchy(0, 2); lower=0), n_gr) # group-level slopes SDs
    αⱼ ~ filldist(Normal(0, τₐ), n_gr)                    # group-level intercepts
    βⱼ ~ filldist(Normal(0, 1), predictors, n_gr)         # group-level standard normal slopes

    #likelihood
    ŷ = α .+ αⱼ[idx] .+ X * βⱼ * τᵦ
    y ~ MvNormal(ŷ, σ^2 * I)
end;

model_intercept_slope = varying_intercept_slope(X, idx, y)
chain_intercept_slope = sample(model_intercept_slope, NUTS(), MCMCThreads(), 1_000, 4)
summarystats(chain_intercept_slope) |> DataFrame |> println


