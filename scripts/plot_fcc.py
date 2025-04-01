import numpy as np 
import pandas as pd
from tqdm import tqdm
import cloudpickle as pickle
import gzip
import os
import pymc as pm
import matplotlib.pyplot as plt
import seaborn as sns

import arviz as az

data_path = "/qfs/projects/agilebiofoundry/aspergillus_niger_round1_pytensor/AspergillusQ4Milestone/data/runs/round1/20250305_163347_aspergillus_niger_advi_100000_s1234_pytensor"

pymc_model_data_file = os.path.join(data_path, "pymc_model_data.pgz")
vi_file = os.path.join(data_path, "vi.pgz")

with gzip.open(pymc_model_data_file, "rb") as f:
    data = pickle.load(f)
    ll = data["ll"]

with gzip.open(vi_file, "rb") as f:
    results = pickle.load(f)
    hist = results["hist"]
    trace = results["trace"]

Ex_values = trace.posterior['Ex'].values  # return array of Ex values
print("Ex values:", Ex_values)
print("Ex values shape:", Ex_values.shape)

# Extract the shape details for checking
_, num_samples, dim_0, dim_1 = Ex_values.shape

# Create the Flux Control Coefficients (FCC) DataFrame
fcc = pd.DataFrame(
    np.array(
        [
            ll.flux_control_coefficient(Ex=Ex_values[0, sample, :, :])[data["r_labels"].index("EX_3hpp_e")]
            for sample in tqdm(range(num_samples))
        ]
    ),
    columns=data["r_labels"],
)

print(fcc)

fcc.to_csv(os.path.join(data_path, "fcc.csv"))

fcc_sort = fcc.reindex(columns=fcc.median().sort_values().index)
# fcc_prior_sort = fcc_prior.reindex(columns=fcc.median().sort_values().index)


# Calculate Highest Posterior Density (HPD) intervals using ArviZ
hpd = az.hdi(fcc.values)


fcc_consistent = np.sign(hpd[:, 0]) == np.sign(hpd[:, 1])
np.savetxt(os.path.join(data_path, "fcc_consistent.txt"), fcc_consistent, delimiter=',')


# fig = plt.figure(figsize=(7,3))
f, (ax, ax2) = plt.subplots(2, 1, sharex=True, figsize=(15, 7))

j = 0
for i, row in enumerate(hpd):

    if not fcc_consistent[i]:
        continue

    ax.plot([j, j], [row[0], row[1]], color="g")
    ax.plot(j, fcc_sort.iloc[:, i].median(), color="g", marker="o", ms=5)
    ax2.plot([j, j], [row[0], row[1]], color="g")
    ax2.plot(j, fcc_sort.iloc[:, i].median(), color="g", marker="o", ms=5)

    j += 1
ax2.hlines(y=[0], xmin=0, xmax=58, linestyles="dashed")
ax2.hlines(y=[-0.1, 0.1], xmin=0, xmax=58, linestyles="dotted")

ax.set_ylim([1, 2])
ax2.set_ylim([-0.5, 1])

sns.despine(right=False)

# hide the spines between ax and ax2
ax.spines["bottom"].set_visible(False)
ax2.spines["top"].set_visible(False)
ax.xaxis.set_visible(False)
# ax.tick_params(labeltop=False)  # don't put tick labels at the top
ax2.xaxis.tick_bottom()

d = 0.015  # how big to make the diagonal lines in axes coordinates
# arguments to pass to plot, just so we don't keep repeating them
kwargs = dict(transform=ax.transAxes, color="k", clip_on=False)
ax.plot((-d, +d), (-d, +d), **kwargs)  # top-left diagonal
ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

_ = ax2.set_xticks(np.arange(sum(fcc_consistent)))
_ = ax2.set_xticklabels(fcc_sort.columns[fcc_consistent], rotation=90, fontsize=13)

ax.set_ylabel("FCC")

plt.savefig(os.path.join(data_path,"fcc_consistent_credible_intervals.svg"))
