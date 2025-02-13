#!/usr/bin/env python3
import matplotlib.pyplot as plt
import pandas as pd

# Read the data from the results file.
df = pd.read_csv("average_speedup_simd.txt", sep='\s+', comment='#',
                 names=["mode", "steps", "threads", "avg_speedup"])

# Convert 'steps' and 'avg_speedup' to numeric values.
df["steps"] = pd.to_numeric(df["steps"], errors='coerce')
df["avg_speedup"] = pd.to_numeric(df["avg_speedup"], errors='coerce')
df = df.dropna(subset=["steps", "avg_speedup"])

print("Dataframe shape:", df.shape)

plt.figure(figsize=(10, 6))
plt.xscale("log")
plt.xlabel("Steps")
plt.ylabel("Average Speedup")
plt.title("Average Speedup Comparison: OpenMP, C++ Threads, and SIMD")

mode_styles = {
    "o": {"label": "OpenMP (12 threads)", "marker": "o", "color": "blue"},
    "p": {"label": "C++ (12 Threads)", "marker": "s", "color": "green"},
    "simd": {"label": "SIMD", "marker": "d", "color": "red"}
}

for mode, style in mode_styles.items():
    mode_df = df[df["mode"] == mode]
    if not mode_df.empty:
        plt.plot(mode_df["steps"].to_numpy(), mode_df["avg_speedup"].to_numpy(),
                 marker=style["marker"],
                 color=style["color"],
                 linestyle='-',
                 label=style["label"])

plt.legend(title="Parallelization Mode")
plt.grid(True, which="both", ls="--", lw=0.5)
plt.savefig("speedup_comparison.png")
plt.show()
