# ==============================================================================
# Helps plots losses and accuracy for pytorch lightning logged values
# ==============================================================================

import os
import pandas as pd
import plotly.express as px
import plotly.io as pio
import argparse 

pio.templates.defaults = "plotly_white"

parser = argparse.ArgumentParser(description="Plots losses or accuracy")
parser.add_argument("-csv_p", type = str, help = "Path to csv containing losses or accuracy")

def plot_losses(csv_p:str):
    """Returns a Plot of the lossed"""
    df = pd.read_csv(csv_p)
    plots = []
    if "collapse_level" in df.columns.values:
        p1 = px.line(x = df.epoch, y = df.collapse_level, title = "collapse_level")
        plots.append(p1)
    if "training loss" in df.columns.values:
        p2 = px.line(x = df.epoch, y = df["training loss"], title = "Training Loss")
        plots.append(p2)
    return plots, df

if __name__ == "__main__": 
    args = parser.parse_args()
    plots, df = plot_losses(args.csv_p)
    save_dir = os.path.dirname(args.csv_p)

    if len(plots) == 1:
        p_loss = plots[0]
        p_loss.write_image(os.path.join(save_dir, "loss.png"))
        
    if len(plots) == 2:
        p_collapse = plots[0]
        p_loss = plots[1]
        p_loss.write_image(os.path.join(save_dir, "loss.png"))
        p_collapse.write_image(os.path.join(save_dir, "collapse.png"))

    print(df)


#todo : take a title as input and save them in a folder called "Visualization"