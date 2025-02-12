using CSV:read 
using DataFrames 
using ArgParse 
using CairoMakie 

s = ArgParseSettings()
@add_arg_table s begin
    "--csv_p"
    arg_type = String 
    help = "Path to downstream model performance metrics"
end

args = parse_args(s)

function plot_acc(df::DataFrame, plot_title::String; save_p::String)
    with_theme(theme_light()) do 
        
        fig = Figure(background = RGBf(0.98,0.98,0.98), size = (900,1200))
        # Subplot for AUC
        ax_auc = Axis(fig[1,1], ylabel = "auc", title = plot_title)
        l_trn_auc = lines!(ax_auc, df.epoch_weight, df.train_auc, color = "black")
        l_val_auc = lines!(ax_auc, df.epoch_weight, df.val_auc, color = "blue")
        # Subplot for F1
        ax_f1 = Axis(fig[2,1], ylabel = "f1")
        l_trn_f1 = lines!(ax_f1, df.epoch_weight, df.train_f1, color = "black")
        l_val_f1 = lines!(ax_f1, df.epoch_weight, df.val_f1, color = "blue")
        # SUbplot for acc
        ax_acc = Axis(fig[3,1], ylabel = "acc", xlabel = "representation at epoch")
        l_trn_acc = lines!(ax_acc, df.epoch_weight, df.train_acc, color = "black")
        l_val_acc = lines!(ax_acc, df.epoch_weight, df.val_acc, color = "blue")
        # Legend added in 2nd row 2nd column
        Legend(fig[2,2],[[l_trn_auc,l_trn_f1, l_trn_acc],[l_val_auc,l_val_f1,l_val_acc]], ["train", "val"])
        # Add X,Y labels
        save(save_p, fig ; pt_per_unit = 2)
    end
end

# ------------------------------ Read DataFrame ------------------------------ #
df = read(args["csv_p"], DataFrame)
save_root = dirname(args["csv_p"])
fname = "acc_plot.png"
title = "Malaria prediction using LR for Represnetations produced by Dino with Resnet backbone"

plot_acc(df, title, save_p = joinpath(save_root,fname))