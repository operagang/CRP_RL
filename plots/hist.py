import pandas as pd
import matplotlib.pyplot as plt

# Load the Excel file
file_path = "./log_lin.xlsx"
df = pd.read_excel(file_path)

# Filter for R instances only
df_r = df[df['Instance'].str.startswith('R')].reset_index(drop=True)

# Get all possible diff types
bay_columns = [col for col in df.columns if col.startswith('Bay')]
row_columns = [col for col in df.columns if col.startswith('Row')]

# Define the number of instances
n_instances = len(df_r)

# Create subplots: 4 columns (Bay_False, Bay_True, Row_False, Row_True), 1 row per instance
fig, axs = plt.subplots(n_instances, 4, figsize=(16, 2.5 * n_instances))
fig.subplots_adjust(hspace=0.6, wspace=0.3)

# Ensure axs is 2D
if n_instances == 1:
    axs = axs.reshape(1, -1)

for idx, (_, row) in enumerate(df_r.iterrows()):
    # Extract diff labels from False columns
    labels_bay = [col.split('_')[1] for col in bay_columns if col.endswith('_False')]
    labels_bay = [label for label in labels_bay if label != '0']
    x_bay = range(len(labels_bay))
    labels_row = [col.split('_')[1] for col in row_columns if col.endswith('_False')]
    labels_row = [label for label in labels_row if label != '0']
    x_row = range(len(labels_row))

    # Collect data
    bay_false = [row[f'BayDiff_{label}_False'] if f'BayDiff_{label}_False' in df.columns else 0 for label in labels_bay]
    bay_true = [row[f'BayDiff_{label}_True'] if f'BayDiff_{label}_True' in df.columns else 0 for label in labels_bay]
    row_false = [row[f'RowDiff_{label}_False'] if f'RowDiff_{label}_False' in df.columns else 0 for label in labels_row]
    row_true = [row[f'RowDiff_{label}_True'] if f'RowDiff_{label}_True' in df.columns else 0 for label in labels_row]

    for col_idx, (data, title) in enumerate(zip(
        [bay_false, bay_true, row_false, row_true],
        ['Bay False', 'Bay True', 'Row False', 'Row True']
    )):
        if title.startswith('B'):
            x = x_bay
            labels = labels_bay
        else:
            x = x_row
            labels = labels_row
        ax = axs[idx, col_idx]
        ax.bar(x, data, width=0.6)
        ax.set_title(f'{row["Instance"]} - {title}', fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=90, fontsize=6)
        ax.tick_params(axis='y', labelsize=6)


# Save the figure
output_path = "./r_instance_histogram_split_lin.png"
plt.tight_layout()
plt.savefig(output_path)
plt.close()
