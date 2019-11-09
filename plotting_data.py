import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os


parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", dest="file", help="input data file")

args = parser.parse_args()
file = args.file

sns.set(style="dark", palette="ocean")

diabetes = pd.read_csv(filepath_or_buffer=file, sep=' ', header=0)
print(diabetes)

os.makedirs('viz-plots/viz-homework', exist_ok=True)



pairs_plot = sns.pairplot(diabetes)
plt.savefig('viz-plots/viz-homework/pairs.png')

