import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV files.
seq_data = pd.read_csv("heatmap_seq_timing_avg.csv")
gpu_data = pd.read_csv("heatmap_timing_avg.csv")

# Create a DataFrame for comparison.
comparison = pd.DataFrame({
    'Sequential (ms)': seq_data.iloc[0],
    'GPU (ms)': gpu_data.iloc[0]
})

# Remove the 'Total' row if you want individual comparisons, or include it.
# Here, we plot all four metrics.
comparison = comparison.transpose()

# Create a grouped bar chart.
labels = comparison.columns
x = np.arange(len(labels))  # label locations
width = 0.35  # width of the bars

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, comparison.loc['Sequential (ms)'], width, label='Sequential')
rects2 = ax.bar(x + width/2, comparison.loc['GPU (ms)'], width, label='GPU')

# Add labels, title, and custom x-axis tick labels.
ax.set_ylabel("Time (ms)")
ax.set_title("Average Heatmap Update Timings (200 ticks)")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# Optionally, add text labels showing the value on top of each bar.
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width()/2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
plt.show()
