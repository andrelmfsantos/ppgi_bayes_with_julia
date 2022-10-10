# Pacotes necessários
import Pkg
#Pkg.add("StatsPlots")
#Pkg.add("Distributions")
#Pkg.add("LaTeXStrings")
#Pkg.add("DataFrames")

# Carregar pacotes
using StatsPlots, Distributions, LaTeXStrings, DataFrames

# Plots Densidade
#plot(Normal(0, 1), lw=5, label=false, xlabel=L"x", ylabel="Density")
#plot(TDist(2), lw=5, label=false, xlabel=L"x", ylabel="Density", xlims=(-4, 4))

# Normal vs Student-t
#plot(Normal(0, 1), lw=5, label="Normal", xlabel=L"x", ylabel="Density", xlims=(-4, 4))
#plot!(TDist(2), lw=5, label="Student")

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

# Carregar base
using DataFrames, CSV, HTTP

url = "https://raw.githubusercontent.com/andrelmfsantos/df/main/dataset_roi.csv"
data = CSV.read(HTTP.get(url).body, DataFrame)
describe(data)

# Subset
df = data[!, names(data,[:MARCA,:LL,:P1,:P2,:P3,:INVESTIMENTO,:LUCRO, :ROI])]

# Replace missing data
for col in eachcol(df)
    replace!(col,missing => 0)
end

# Converter colunas para inteiro
df[!,:LL] = map(Int64, df[!,:LL])
df[!,:P1] = map(Int64, df[!,:P1])
df[!,:P2] = map(Int64, df[!,:P2])
df[!,:P3] = map(Int64, df[!,:P3])
first(df,5)

# Densidade
@df df density(:ROI, label=false)

# Agrupar
# https://stackoverflow.com/questions/64226866/groupby-with-sum-on-julia-dataframe
numcols = names(df, findall(x -> eltype(x) <: Number, eachcol(df)))
gdf = combine(groupby(df, ["MARCA"]), numcols .=> sum .=> numcols)

# Coluna calculada (ROI)
roi = gdf[!,:LUCRO]./gdf[!,:INVESTIMENTO]
gdf.ROI = roi
show(gdf, allcols=true)

#Pkg.add("Plots")
using Plots
Xs = gdf[!,:MARCA]
Ys = gdf[!,:ROI]
p = bar(Xs,Ys, label=false)
ylabel!("ROI")
xlabel!("Marcas")
title!("ROI = Lucro/Investimento (em todo período)")
#savefig(p,"barplot.png")

# Instanciar os dados
X = Matrix(select(df, [:LL, :P1, :P2, :P3]))
y = df[:, :ROI]
model = robustreg(X, y);

# Cadeias de Markov
chain = sample(model, NUTS(), MCMCThreads(), 1_000, 4)
#summarystats(chain)
#show(summarystats(chain), allcols=true)
DataFrame(summarystats(chain))

# Quartiles
quantile(chain)
