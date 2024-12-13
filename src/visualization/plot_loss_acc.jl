using CSV:read
using DataFrames
using ArgParse
using CairoMakie

s = ArgParseSettings()
@add_arg_table s begin
    "--csv_p"
        arg_type = String
        help = "Path to losses csv"
end
args = parse_args(s)

# ------------------------------ Read DataFrame ------------------------------ #
df = read(args["csv_p"], DataFrame)
folder_p = dirname(args["csv_p"])
f_name = "loss.png"

with_theme(theme_light()) do 
    fig,ax,l = lines(
        df[:,"epoch"], 
        df[:,"training loss"],
        color = :black
    )
    ax.xlabel ="epochs" ; ax.ylabel ="loss"  ; ax.title ="loss" 
    save(joinpath(folder_p,f_name), fig; pt_per_unit = 2)
end

