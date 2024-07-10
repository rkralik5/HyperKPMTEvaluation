"""
Plots the dark rate for a given threshold, against voltage.
Only works for files where the last term before "_dr.csv" is the voltage.
e.g. 20230830_1000v_dr.csv
"""
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from sys import argv

def v_from_fname(fname):
    terms = fname.split("_")
    # Last term should be "dr.csv" if from waveform.py
    # Second to last is the voltage
    v = terms[-2]

    # Remove the "v"
    return float(v[:-1])

def main():
    if "-h" in argv or "--help" in argv:
        print("--threshold : Threshold (mV) for peak triggering.")
        return

    threshold = -8
    if "--threshold" in argv:
        # Set threshold to value after flag
        thresh_i = argv.index("--threshold")
        threshold = float(argv[(thresh_i+1)])
    else:
        print("No threshold given using --threshold. "
            "Using default threshold of -8 mV")

    if threshold > 0:
        print("PMT threshold given is positive, converting to negative.")
        threshold *= -1

    # Use glob to expand wildcards
    fnames = []
    for arg in argv[1:]:
        fnames += glob(arg)

    dfs = [pd.read_csv(x) for x in fnames]
    vs = [v_from_fname(x) for x in fnames]

    dr_v_rows = []
    for df in dfs:
        # Threshold isn't necessarily in the df
        # Insert a new row with only the thresh info
        df.at[len(df), "threshold"] = threshold
        # Sort and interpolate
        df = df.sort_values("threshold")
        # df[df.columns != "threshold"] = df[df.columns !=
        # "threshold"].interpolate()
        df = df.interpolate()

        dr_v_rows.append(df.loc[df["threshold"] == threshold])

    dr_v_df = pd.concat(dr_v_rows)
    dr_v_df["v"] = vs

    dr_v_df = dr_v_df.sort_values("v")

    # Columns are the channel labels from waveform.py with _dr suffix
    dr_cols = [x for x in dr_v_df.columns if x.endswith("_dr")]
    labels = [x[:-3] for x in dr_cols]

    dr_fig, dr_ax = plt.subplots()
    dr_fig.set_size_inches(14,8)

    dr_ax.plot(dr_v_df["v"], dr_v_df[dr_cols], label=labels, marker=".")
    dr_ax.legend()
    dr_ax.set_yscale("log")

    dr_ax.set_xlabel("Supply Voltage [V]")
    dr_ax.set_ylabel("Dark Rate [/s]")
    plt.show()

    return

if __name__ == "__main__":
    main()