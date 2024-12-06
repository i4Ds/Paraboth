import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Configure seaborn aesthetics
sns.set_theme(style="whitegrid")

# Specify the folder containing your files
folder_path = "results"

# Initialize lists to store data
bleu_data = []
wer_data = []

# Iterate through each file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".tsv"):  # Adjust the extension if necessary
        file_path = os.path.join(folder_path, filename)

        # Read the file into a pandas DataFrame
        try:
            df = pd.read_csv(file_path, sep="\t")
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            continue  # Skip this file and continue with the next

        # Clean Filename
        filename = filename.split("_")[
            1:-2
        ]  # Remove the last two parts of the filename
        filename = "_".join(filename)  # Join the remaining parts back together
        # Extract BLEU and WER metrics
        for _, row in df.iterrows():
            metric = row["Metric"]
            score = row["Score"]

            if "BLEU" in metric:
                bleu_data.append({"File": filename, "Metric": metric, "Score": score})
            elif "WER" in metric:
                wer_data.append({"File": filename, "Metric": metric, "Score": score})

# Convert lists to DataFrames
bleu_df = pd.DataFrame(bleu_data).sort_values(by=["File"])

# Create names
scenario_names = bleu_df["File"].unique()
bleu_scores = (
    100 * bleu_df[bleu_df["Metric"] == "Original Corpus - BLEU"]["Score"].values
)
parableu_scores = (
    100
    * bleu_df[bleu_df["Metric"] == "Aligned and Paraphrased Corpus - ParaBLEU"][
        "Score"
    ].values
)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

print(scenario_names)
print(bleu_scores)
print(parableu_scores)

x = np.arange(len(scenario_names))
width = 0.35  # Width of the bars

# Create a figure and axis
fig, ax = plt.subplots(figsize=(12, 5))

# Plot BLEU and ParaBLEU bars
rects1 = ax.bar(
    x - width / 2, bleu_scores, width, label="BLEU", color="#1f77b4", edgecolor="black"
)
rects2 = ax.bar(
    x + width / 2,
    parableu_scores,
    width,
    label="ParaBLEU",
    color="#2ca02c",
    edgecolor="black",
)

# Add text labels, title, and custom x-axis tick labels, etc.
ax.set_xlabel("Scenarios", fontsize=14)
ax.set_ylabel("Score", fontsize=14)
ax.set_title("BLEU and ParaBLEU Scores Across Different Scenarios", fontsize=16, pad=15)
ax.set_xticks(x)
ax.set_xticklabels(scenario_names, rotation=45, ha="right")
ax.set_ylim(0, max(max(bleu_scores), max(parableu_scores)) + 0.1)
ax.legend(fontsize=12)
ax.grid(axis="y", linestyle="--", linewidth=0.5)

# Annotate bars with their heights
for rects in [rects1, rects2]:
    for rect in rects:
        height = rect.get_height()
        ax.annotate(
            f"{height:.2f}",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),  # Offset text by 3 points vertically
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
        )

plt.tight_layout()
plt.savefig(f"bleu_parableu_scores_{timestamp}.png")
