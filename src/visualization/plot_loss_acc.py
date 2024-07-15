# ==============================================================================
# Helps plots losses and accuracy for pytorch lightning logged values
# ==============================================================================

import pandas as pd
import plotly.express as px
import argparse 

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
    for p in plots:
        p.show()

    print(df)

#todo : take a title as input and save them in a folder called "Visualization"