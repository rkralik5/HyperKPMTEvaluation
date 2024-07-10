import matplotlib.pyplot as plt
import pandas as pd
from sys import argv
from pathlib import Path

brand_dict = {"ham": "Hamamatsu", "nnvt" : "NNVT", "NNVT":"NNVT"}
markers = ['o', 's', 'v', '^']
colors = ("b", "g", "r", "c", "m", "y", "k")
fname = argv[1]

fdt = fname.split("_")[0]

df = pd.read_csv(fname)

# Assumes standarde file naming format
def get_v_from_fname(row):
    base = Path(row["fname"]).stem

    # Get brand model and voltage from filename
    brand, model, voltage = base.split("_")
    brand = brand_dict[brand]
    # Get rid of "v" in string if there
    if voltage[-1] == "v":
        voltage = voltage[:-1]
    return brand, model, float(voltage)

# Expand here means when apply returns multiple values, they get assigned to
# each given new column
df[["brand","model","v"]] = df.apply(get_v_from_fname, axis="columns",
    result_type="expand")

# Sort so V is in order
df = df.sort_values(["v","fname"], ascending=True)
df = df.set_index("v")

# Group by the model and plot each value
df_group = df.groupby("model")

# Get the NNVT measurements if given
if len(argv) > 2:
    nnvt_fname = argv[2]
    nnvt_df = pd.read_csv(nnvt_fname)

    # Only get the same models for direct comparison
    nnvt_df = pd.merge(nnvt_df, df["model"].drop_duplicates())
    # Add clarification in label
    nnvt_df["model"] = nnvt_df["model"].apply(lambda x: x + " (NNVT)")
    nnvt_df = nnvt_df.sort_values(["v","model"], ascending=True)
    nnvt_df = nnvt_df.set_index("v")

    # Given in units of 1e7
    nnvt_df["gain"] *= 1e7
    nnvt_df_group = nnvt_df.groupby("model")

plot_cols = {
    "gain" : "Gain", 
    "chisqr" : r"$\chi^2$", 
    "pv_r" : "Peak-Valley Ratio", 
    "sigma" : r"$\sigma$", 
    "pe_res" : "PE Resolution"
}

# List of values to be log-plotted
logs = [
    "chisqr"
]

# Marker size is a bit small by default
msize = 15

fontsize = 15

for key,value in plot_cols.items():
    fig, ax = plt.subplots()
    

    
    # Cycle through grouped df plotting each PMT
    idx=0
    for pmt,df_pmt in df_group:
        
        if 'KM' in pmt:
            marker = markers[0]
        elif 'PN' in pmt:
            marker = markers[1]
            if 'PN23' in pmt:
                marker = markers[2]
            if '(NNVT)' in pmt:
                marker = markers[3]
        color=colors[idx]
        format_str = "{color}{marker}-".format(color=color, marker=marker)
        ax.plot(df_pmt[key], format_str, label=pmt)
        idx+=1

    if len(argv) > 2:
        try:
            # Same for NNVT df, if it has the info
            #idx = 0
            for pmt,df_pmt in nnvt_df_group:
                
                if 'KM' in pmt:
                    marker = markers[0]
                elif 'PN' in pmt:
                    marker = markers[1]
                    if 'PN23' in pmt:
                        marker = markers[2]
                if '(NNVT)' in pmt:
                    marker = markers[3]
                color=colors[idx]
                format_str = "{color}{marker}-".format(color=color, marker=marker)
                ax.plot(df_pmt[key], format_str, label=pmt)
                idx +=1
        except:
            print(f"{key} not in NNVT data. Won't plot")

    # Log it if set to
    if key in logs:
        ax.set_yscale("log")

    ax.legend(fontsize=fontsize)
    ax.set_xlabel("Voltage [V]",fontsize=fontsize)
    ax.set_ylabel(value, fontsize=fontsize)

    fig.set_size_inches(14,8)

    fig.savefig(f"{fdt}_{key}.png")

plt.show()
