import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", dest="file", help="input data file")

args = parser.parse_args()
file = args.file

diabetes = pd.read_csv(filepath_or_buffer=file, sep=' ', header=0)



os.makedirs('viz-plots/seaborn_heatmap', exist_ok=True)

sns.set()

fig, ax = plt.subplots(figsize=(14,14))
sns.heatmap(diabetes.corr(), annot=True, ax=ax, cmap='icefire', fmt='.2f', annot_kws={"size": 15}, linewidths=.05)
ax.set_xticklabels(diabetes.columns, rotation=45)
ax.set_yticklabels(diabetes.columns, rotation=45)
fig.subplots_adjust(top=.75)
plt.savefig('viz-plots/seaborn_heatmap/diabetes_heatmap.png')
# plt.tight_layout()
plt.close()

diabetes.hist(bins=14, color='steelblue', edgecolor='black', linewidth=1.0,
           xlabelsize=6, ylabelsize=6, grid=False)
# plt.tight_layout(rect=(0, 0, 1.2, 1.2))
plt.savefig('viz-plots/seaborn_heatmap/diabetes_histogram.png')