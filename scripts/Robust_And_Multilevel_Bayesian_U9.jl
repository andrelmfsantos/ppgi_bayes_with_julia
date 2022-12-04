# Pacotes necessários
import Pkg
#Pkg.add("StatsPlots")
#Pkg.add("Distributions")
#Pkg.add("LaTeXStrings")
#Pkg.add("DataFrames")

# Carregar pacotes
using StatsPlots, Distributions, LaTeXStrings, DataFrames

# Instalar pacotes necessários
#Pkg.add("Turing")
#Pkg.add("StatsBase")

# Carregar pacotes
using Turing
using Statistics: mean, std
using StatsBase: mad
using Random: seed!

# Travar seed
seed!(123)

# Modelo
@model function robustreg(X, y; predictors=size(X, 2))
        #priors
        α ~ LocationScale(median(y), 2.5 * mad(y), TDist(3))
        β ~ filldist(TDist(3), predictors)
        σ ~ Exponential(1)
        ν ~ LogNormal(2, 1)

        #likelihood
        y ~ arraydist(LocationScale.(α .+ X * β, σ, TDist.(ν)))
end;

using DataFrames, CSV, HTTP

url = "https://raw.githubusercontent.com/andrelmfsantos/df/main/dataset_brands_quarter.csv"
data = CSV.read(HTTP.get(url).body, DataFrame)
println(first(data,5))
describe(data)

# Substituir nas casas decimais, vírgula por pontos
df = replace.(data, r"," => ".")
println(first(df,5))

# Converter colunas para Float
df.Spends = parse.(Float64, df.Spends)
df.Maco = parse.(Float64, df.Maco)
df.Causals = parse.(Float64, df.Causals)
df.Volume = parse.(Float64, df.Volume)
df.CPM = parse.(Float64, df.CPM)
df.EFF = parse.(Float64, df.EFF)
df.ROI = parse.(Float64, df.ROI)
df.P1 = parse.(Float64, df.P1)
println(first(df,5))

# Subset
df = df[!, names(data,[:Brand,:P1,:CPM, :EFF, :ROI])]
show(df, allcols=true)

# Densidade
@df df density(:CPM, label=false)

# Instanciar os dados
X = Matrix(select(df, [:P1]))
y = df[:, :CPM]
model = robustreg(X, y);

# Cadeias de Markov
chain = sample(model, NUTS(), MCMCThreads(), 1_000, 4)
#summarystats(chain)
#show(summarystats(chain), allcols=true)
DataFrame(summarystats(chain))

# Base
# Substituir nas casas decimais, vírgula por pontos
df = replace.(data, r"," => ".")
# Converter colunas para Float
df.Spends = parse.(Float64, df.Spends)
df.Maco = parse.(Float64, df.Maco)
df.Causals = parse.(Float64, df.Causals)
df.Volume = parse.(Float64, df.Volume)
df.CPM = parse.(Float64, df.CPM)
df.EFF = parse.(Float64, df.EFF)
df.ROI = parse.(Float64, df.ROI)
df.P1 = parse.(Float64, df.P1)
# Subset
df = df[!, names(data,[:Brand,:Date,:P1,:CPM, :EFF, :ROI])]

show(df, allcols=true)

# Nova coluna com quarter
df = hcat(df, DataFrame(reduce(vcat, permutedims.(split.(df.Date, 'Q'))), [:Year, :Quarter]))
df

for c in unique(df[:, :Brand])
    df[:, "Brand_$c"] = ifelse.(df[:, :Brand] .== c, 1, 0)
end

df[:, :Quarter_int] = map(df[:, :Quarter]) do b
    b == "1" ? 1 :
    b == "2" ? 2 :
    b == "3" ? 3 :
    b == "4" ? 4 : missing
end

show(df, allcols=true)

X = Matrix(select(df, Between(:Brand_Brand1, :Brand_Brand6)));
y = df[:, :CPM];
idx = df[:, :Quarter_int];

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


