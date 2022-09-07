# Importar pacotes necessários
import Pkg; Pkg.add("HTTP"); Pkg.add("DataFrames"); Pkg.add("CSV")

# Carregar pacotes
using HTTP, DataFrames, Dates, CSV

# Ativo de interesse
acao = "GOLL4"

# Carregar dados
r = HTTP.get("https://query1.finance.yahoo.com/v7/finance/download/"*acao*".SA?period1=1640995200&period2=1662508800&interval=1d&events=history&includeAdjustedClose=true");
data = String(r.body);

# Dados
data

# Preparação da base
csv_data = replace(data,"Date,Open,High,Low,Close,Adj Close,Volume\n"=>"") # remove cabeçalho
csv_data = split(csv_data, "\n")                                           # separa strings por "\n" 

# Criar dataframe
df = DataFrame([csv_data],[:x1])
first(df, 5)

# Separar coluna em múltiplas colunas
transform!(df, :x1 => ByRow(x -> split(x, ",")) => [:Y1, :Y2, :Y3, :Y4, :Y5, :Y6, :Y7])
show(df, allcols=true)

# Renomear colunas
select!(df,
    "Y1" => "Date",
    "Y2" => "Open",
    "Y3" => "High",
    "Y4" => "Low",
    "Y5" => "Close",
    "Y6" => "Adj_Close",
    "Y7" => "Volume")

# Mudar os tipos das variáveis
#using Dates
df.Date = Date.(df.Date, "yyyy-mm-dd")
df.Open = parse.(Float64, df.Open)
df.High = parse.(Float64, df.High)
df.Low = parse.(Float64, df.Low)
df.Close = parse.(Float64, df.Close)
df.Adj_Close = parse.(Float64, df.Adj_Close)
df.Volume = parse.(Float64, df.Volume)
first(df,5)

# Salvar CSV
#using CSV
CSV.write("C:\\Users\\User\\Documents\\goll4.csv", df)
