#!/usr/bin/env python3

import matplotlib.pyplot as plt
from collections import defaultdict

def load_data(filename):
    """
    Returns a dict of structure:
    {
      ('o', 1000): [(2, 0.762353), (4, 0.965411), ...],
      ('o', 3000): [...],
      ('p', 1000): [...],
      ...
    }
    where 'o' = openmp, 'p' = c++ threads
    """
    data = defaultdict(list)
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            # Ignore comments or empty lines
            if not line or line.startswith('#'):
                continue
            
            # Format: mode steps threads avg_speedup
            parts = line.split()
            if len(parts) != 4:
                continue
            
            mode_str = parts[0]     # 'o' or 'p'
            steps = int(parts[1])
            threads = int(parts[2])
            speedup = float(parts[3])
            
            # Append to dict under key (mode, steps)
            data[(mode_str, steps)].append((threads, speedup))
    
    # For consistency, sort each list by 'threads'
    for key in data:
        data[key].sort(key=lambda x: x[0])  # sort by threads
    return data


def plot_speedup(data, output_prefix='speedup_plot'):
    """
    Generates line plots of speedup vs. threads for each step count.
    We plot both 'o' (OpenMP) and 'p' (C++ threads) on the same axes,
    for easy comparison.
    """
    
    # 1) Extract the unique step counts
    # They appear in keys as (mode, steps)
    step_set = sorted({key[1] for key in data.keys()})  # unique steps
    
    # We'll create one figure per step count
    for steps in step_set:
        # Gather (threads, speedup) for openmp (o) and c++ (p)
        openmp_key = ('o', steps)
        cpp_key    = ('p', steps)
        
        if openmp_key not in data and cpp_key not in data:
            # No data for this step?
            continue
        
        fig, ax = plt.subplots(figsize=(6,4))
        
        # Plot OpenMP if present
        if openmp_key in data:
            threads_openmp = [item[0] for item in data[openmp_key]]
            speedups_openmp = [item[1] for item in data[openmp_key]]
            ax.plot(threads_openmp, speedups_openmp,
                    marker='o', label=f"OpenMP (o), steps={steps}")
        
        # Plot C++ Threads if present
        if cpp_key in data:
            threads_cpp = [item[0] for item in data[cpp_key]]
            speedups_cpp = [item[1] for item in data[cpp_key]]
            ax.plot(threads_cpp, speedups_cpp,
                    marker='s', label=f"C++ Threads (p), steps={steps}")
        
        ax.set_title(f"Speedup vs Threads (steps={steps})")
        ax.set_xlabel("Threads")
        ax.set_ylabel("Average Speedup")
        ax.grid(True)
        ax.legend()
        
        # Save figure, e.g. speedup_plot_1000.png
        fig_name = f"{output_prefix}_{steps}.png"
        plt.savefig(fig_name, dpi=150)
        plt.close(fig)
        print(f"Saved {fig_name}")


def main():
    # 1) Load data from file
    filename = "average_speedup.txt"
    data = load_data(filename)
    
    # 2) Plot
    plot_speedup(data, output_prefix="speedup_plot")


if __name__ == "__main__":
    main()
