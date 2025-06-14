# Re-import necessary libraries after kernel reset
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# Reload the Excel file
file_path = "./vs Lin.xlsx"
df = pd.read_excel(file_path)


# Reshape to long format
df_long = df.melt(id_vars=['Layout', 'Instance'], value_vars=['Lin', 'Ours'],
                  var_name='Method', value_name='WT')

# Preprocess
df_long['Scale'] = df_long['Layout']
df_long['Side'] = df_long['Method'].map({'Lin': -0.15, 'Ours': 0.15})
df_long['Color'] = df_long['Method'].map({'Lin': 'tab:blue', 'Ours': 'tab:orange'})
df_long['Marker'] = df_long['Method'].map({'Lin': 'o', 'Ours': 'o'})

# Set up plot
plt.figure(figsize=(6, 6))
unique_scales = df_long['Scale'].unique()
xticks = np.arange(len(unique_scales))

# Plot
for i, scale in enumerate(unique_scales):
    subset = df_long[df_long['Scale'] == scale]
    for _, row in subset.iterrows():
        x = i + row['Side']
        plt.scatter(x, row['WT'], color=row['Color'], s=60, edgecolors='black', linewidths=0.8, zorder=2, marker=row['Marker'])

# Legend
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Lin', markerfacecolor='tab:blue', markersize=8),
    Line2D([0], [0], marker='o', color='w', label='Ours', markerfacecolor='tab:orange', markersize=8),
]
plt.legend(handles=legend_elements, loc='upper left', fontsize=10)

# Labels and layout
plt.xticks(xticks, unique_scales, fontsize=10)
plt.xlabel("Layout", fontsize=12)
plt.ylabel("Gap (%)", fontsize=12)
plt.title("Comparison between Lin and Ours per Instance", fontsize=13)
plt.xticks(rotation=45)

plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("./vs_Lin.pdf", format="pdf", bbox_inches="tight")
plt.show()
