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

url = "https://raw.githubusercontent.com/andrelmfsantos/df/main/ppgi_bayes_base_atividade_final.csv"
df = CSV.read(HTTP.get(url).body, DataFrame)
println(first(df,5))
describe(df)

# Densidade
@df df density(:ROI, label=false)

# Instanciar os dados
X = Matrix(select(df, [:LL,:P1,:P2,:P3]))
y = df[:, :ROI]
model = robustreg(X, y);

# Cadeias de Markov
chain = sample(model, NUTS(), MCMCThreads(), 1_000, 4)
#summarystats(chain)
#show(summarystats(chain), allcols=true)
DataFrame(summarystats(chain))
