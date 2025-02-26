using Turing:@model, sigmoid, NUTS, Bernoulli, Normal, sample, Exponential, filldist
import CSV
using DataFrames

readdir()
data = CSV.read("data/interim/sshsph_mlr/LFMalaysiaKmeans_clean.csv", DataFrame)

X_names = ["houseID", "gender", "age"]
y_name = ["lf"]

df = select(data, vcat(X_names, y_name))
df[:,"gender_cat"] = ifelse.(df[:,"gender"] .== "m",1, 0)
select!(df, ["houseID","gender_cat","age","lf"])

size(df.age)
length(unique(df.houseID))

@model function multilevel_logistic(Xₐ,Xₛ,y,hid)
    """
    Xa : Age (Independendent variable)
    Xs : Sex (Independent Binary variable)
    y : Filarilias
    """ 
    N = size(Xₐ) #7106
    num_clus = length(unique(hid))

    # Priors for the fixed effects : age and sex
    α ~ Normal(0,1) # intercept
    βₛ ~ Normal(0,1) # gender effect
    βₐ ~ Normal(0,1) # age effect

    # Random intercepts for houses
    σₕ ~ Exponential(1)
    house_effect ~ filldist(Normal(0, σₕ), num_clus)

    # likelihood
    for i in 1:N
        house_idx = hid[i]
        η = α + βₛ * Xₛ[i] + Βₐ * Xₐ[i] + house_effects[house_idx]
        p = sigmoid(η)
        y[i] ~ Bernoulli(p)
    end
end

res = sample(multilevel_logistic(df.gender_cat, df.age, df.lf, df.houseID), NUTS(), 1000)

for i in 1:10
    @show i
end