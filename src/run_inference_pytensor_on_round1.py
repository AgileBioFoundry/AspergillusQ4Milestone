# Code to run the ADVI inference with a near-genome scale model and relative
# omics data.

# So I've found that for certain hardware (the intel chips on the cluster here,
# for instance) the intel python and mkl-numpy are about 2x as fast as the
# openblas versions. You can delete a bunch of this stuff if it doesn't work
# for you. This example is a lot slower than some of the other ones though, but
# I guess that's expected

import os

os.environ["MKL_THREADING_LAYER"] = "GNU"
import sys

# sys.path.append('/qfs/projects/agilebiofoundry/emll')  # using PPI emll instead
import pandas as pd
import numpy as np
import pymc as pm
import pytensor.tensor as T
import argparse
import cobra
import emll, gzip
import cloudpickle as pickle
import time

from emll.util import initialize_elasticity
from datetime import datetime
import logging

# Note: need tellurium for emll

# run configuration
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
vi_method = "advi"  # use "advi" or "svgd" or "fullrank_advi"
n_inference_iterations = 40000  # 40000
n_posterior_predictive = 2000  # 2000
n_prior_predictive = 1000  # 1000
seed = 1

if vi_method == "advi":
    advi_lr = 0.005
    advi_grad_constaint = 100
    advi_inference_args = dict(
        obj_optimizer=pm.adagrad_window(learning_rate=advi_lr),
        total_grad_norm_constraint=advi_grad_constaint,
    )
run_directory = f"../data/runs/round1/pytensor/{timestamp}_aspergillus_niger_{vi_method}_{n_inference_iterations}_s{seed}_pytensor"
np.random.seed(seed)


# make run directory
if os.path.exists(run_directory):
    print(f"Error: The directory {run_directory} already exists.")
    sys.exit(1)
else:
    os.makedirs(run_directory)
    print(f"Created directory: {run_directory}")

# start logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f"{run_directory}/logfile.txt"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)
logger.info(f"Timestamp: {timestamp}")
logger.info(f"VI Method: {vi_method}")
logger.info(f"Number of Inference Iterations: {n_inference_iterations}")
logger.info(f"Number of Posterior Predictive Samples: {n_posterior_predictive}")
logger.info(f"Number of Prior Predictive Samples: {n_prior_predictive}")
logger.info(f"Random Seed: {seed}")

if vi_method == "advi":
    logger.info(f"ADVI Learning Rate: {advi_lr}")
    logger.info(f"ADVI Gradient Constraint: {advi_grad_constaint}")


logger.info(f"Run Directory: {run_directory}")

# load model and data
model_file = "../models/iJB1325_HP.nonnative_genes.pubchem.flipped.nonzero.reduced.json"  # same as round 1
v_star_file = (
    "../data/round1/Eflux2_flux_rates.flipped.csv"  # --> notebooks/Eflux4A.niger.ipynb
)
x_file = "../data/round1/metabolite_concentrations.csv"  # --> notebooks/A.niger_MultiOmics.ipynb
e_file = "../data/round1/normalized_targeted_enzyme_activities.csv"  # --> notebooks/A.niger_MultiOmics.ipynb
v_file = (
    "../data/round1/Eflux2_flux_rates.flipped.csv"  # --> notebooks/Eflux4A.niger.ipynb
)
y_file = "../data/round1/normalized_external_metabolites.csv"  # --> notebooks/A.niger_MultiOmics.ipynb
ref_state = "SF ABF93_7-R3"  # change this to highest producing strain in round 2

# set output file paths
vi_file = f"{run_directory}/vi.pgz"
pymc_model_file = f"{run_directory}/pymc_model.pgz"
pymc_model_data_file = f"{run_directory}/pymc_model_data.pgz"

# log the file paths and reference state
logger.info(f"Model file: {model_file}")
logger.info(f"V* file: {v_star_file}")
logger.info(f"X file: {x_file}")
logger.info(f"E file: {e_file}")
logger.info(f"V file: {v_file}")
logger.info(f"Y file: {y_file}")
logger.info(f"Reference state: {ref_state}")
logger.info(f"VI file: {vi_file}")
logger.info(f"PyMC model file: {pymc_model_file}")
logger.info(f"PyMC model data file: {pymc_model_data_file}")


model = cobra.io.load_json_model(model_file)
r_labels = [r.id for r in model.reactions]
r_compartments = [
    r.compartments if "e" not in r.compartments else "t" for r in model.reactions
]

# r_compartments[model.reactions.index('SUCCt2r')] = 'c'
# r_compartments[model.reactions.index('ACt2r')] = 'c'

for rxn in model.exchanges:
    r_compartments[model.reactions.index(rxn)] = "t"

m_compartments = [m.compartment for m in model.metabolites]

v_star = pd.read_csv(v_star_file, index_col=0)[ref_state]
v_star = v_star[[r.id for r in model.reactions if r.id in v_star.index]]
# print(v_star <= 0)
x = pd.read_csv(x_file, index_col=0)
x = x.loc[[m.id for m in model.metabolites if m.id in x.index]]
v = pd.read_csv(v_file, index_col=0)
v = v.loc[[r.id for r in model.reactions]]  # if 'e' in r.compartments]]
e = pd.read_csv(e_file, index_col=0)
e = e.loc[[r.id for r in model.reactions if r.id in e.index]]
y = pd.read_csv(y_file, index_col=0)
y = y.loc[[m.id for m in model.metabolites if m.id in y.index]]

# Drop wild-type
wild_type = "SF ABF93_1-R1,SF ABF93_1-R2,SF ABF93_1-R3".split(
    ","
)  # TODO: change to list of strings
# Reindex arrays to have the same column ordering
to_consider = [c for c in v.columns if c not in wild_type]
v = v.loc[:, to_consider]
x = x.loc[:, to_consider]
e = e.loc[:, to_consider]
y = y.loc[:, to_consider]

n_exp = len(to_consider) - 1


xn = (x.subtract(x[ref_state], 0) * np.log(2)).T
en = e.T  # (2 ** e.subtract(e[ref_state], 0)).T
yn = (y.subtract(y[ref_state], 0) * np.log(2)).T

# To calculate vn, we have to merge in the v_star series and do some
# calculations.
# v_star_df = pd.DataFrame(v_star).reset_index().rename(columns= {0: 'id', 1:'flux'})
# v_merge = v.merge(v_star_df, left_index=True, right_on='id').set_index('id')
# vn = v.divide(v_merge.flux, 0).drop('flux', 1).T
vn = v.T

# Drop reference state
vn = vn.drop(index=ref_state)
xn = xn.drop(index=ref_state)
en = en.drop(index=ref_state)
yn = yn.drop(index=ref_state)

# Get indexes for measured values
x_inds = np.array([model.metabolites.index(met) for met in xn.columns])
e_inds = np.array([model.reactions.index(rxn) for rxn in en.columns])
v_inds = np.array([model.reactions.index(rxn) for rxn in vn.columns])
y_inds = np.array([model.metabolites.index(met) for met in yn.columns])

e_laplace_inds = []
e_zero_inds = []

for i, rxn in enumerate(model.reactions):
    if rxn.id not in en.columns:
        if ("e" not in rxn.compartments) and (len(rxn.compartments) == 1):
            e_laplace_inds += [i]
        else:
            e_zero_inds += [i]

e_laplace_inds = np.array(e_laplace_inds)
e_zero_inds = np.array(e_zero_inds)
e_indexer = np.hstack([e_inds, e_laplace_inds, e_zero_inds]).argsort()

N = cobra.util.create_stoichiometric_matrix(model)
Ex = emll.util.create_elasticity_matrix(model)
Ey = np.zeros((N.shape[1], 2))
Ey[model.reactions.index("r1046"), 0] = 1
Ey[model.reactions.index("3HPPt"), 1] = -1

Ex *= 0.1 + 0.8 * np.random.rand(*Ex.shape)
print(
    "N: ",
    N.shape,
    "Ex: ",
    Ex.shape,
    "Ey: ",
    Ey.shape,
    "v_star: ",
    v_star.shape,
    "vn: ",
    vn.shape,
    "v: ",
    v.shape,
)
ll = emll.LinLogLeastNorm(N, Ex, Ey, v_star.values, driver="gelsy")


with pm.Model() as pymc_model:

    np.random.seed(seed)
    # Priors on elasticity values
    Ex_t = pm.Deterministic(
        "Ex",
        initialize_elasticity(
            ll.N,
            b=0.01,
            sigma=1,
            alpha=None,
            m_compartments=m_compartments,
            r_compartments=r_compartments,
        ),
    )

    Ey_t = pm.Deterministic(
        "Ey", initialize_elasticity(-Ey.T, "ey", b=0.05, sigma=1, alpha=None)
    )
    yn_t = T.as_tensor_variable(yn.values)

    e_measured = pm.Normal(
        "log_e_measured", mu=np.log(en), sigma=0.2, shape=(n_exp, len(e_inds))
    )
    e_unmeasured = pm.Laplace(
        "log_e_unmeasured", mu=0, b=0.1, shape=(n_exp, len(e_laplace_inds))
    )
    log_en_t = T.concatenate(
        [e_measured, e_unmeasured, T.zeros((n_exp, len(e_zero_inds)))], axis=1
    )[:, e_indexer]

    pm.Deterministic("log_en_t", log_en_t)

    # Priors on external concentrations
    # yn_t = pm.Normal('yn_t', mu=0, sd=10, shape=(n_exp, ll.ny),
    #                 testval=0.1 * np.random.randn(n_exp, ll.ny))

    chi_ss, vn_ss = ll.steady_state_pytensor(Ex_t, Ey_t, T.exp(log_en_t), yn_t)
    pm.Deterministic("chi_ss", chi_ss)
    pm.Deterministic("vn_ss", vn_ss)
    log_vn_ss = T.log(T.clip(vn_ss[:, v_inds], 1e-8, 1e8))
    log_vn_ss = T.clip(log_vn_ss, -1.5, 1.5)

    print("log(vn): ", T.shape(log_vn_ss), "vn: ", vn.shape)
    chi_clip = T.clip(chi_ss[:, x_inds], -1.5, 1.5)

    chi_obs = pm.Normal(
        "chi_obs", mu=chi_clip, sigma=0.2, observed=xn.clip(lower=-1.5, upper=1.5)
    )
    log_vn_obs = pm.Normal(
        "vn_obs",
        mu=log_vn_ss,
        sigma=0.1,
        observed=np.log(vn).clip(lower=-1.5, upper=1.5),
    )

# rename for round 2
with gzip.open(pymc_model_file, "wb") as f:
    pickle.dump(pymc_model, f)


with gzip.open(pymc_model_data_file, "wb") as f:
    pickle.dump(
        {
            "model": model,
            "vn": vn,
            "en": en,
            "yn": yn,
            "xn": xn,
            "x_inds": x_inds,
            "e_inds": e_inds,
            "v_inds": v_inds,
            #'m_labels': m_labels,
            "r_labels": r_labels,
            "ll": ll,
            "v_star": v_star,
        },
        f,
    )


if __name__ == "__main__":

    start_time = time.time()
    with pymc_model:
        trace_prior = pm.sample_prior_predictive(
            samples=n_prior_predictive, random_seed=seed
        )
        # approx = pm.ADVI()

        if vi_method == "advi":
            inference_args = advi_inference_args
        else:
            inference_args = None

        hist = pm.fit(
            n=n_inference_iterations,
            method=vi_method,
            inf_kwargs=inference_args,
            random_seed=seed,
        )

        trace = hist.sample(n_posterior_predictive)
        ppc = pm.sample_posterior_predictive(
            trace, random_seed=seed
        )  # pm.sample_ppc(trace)

    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(
        f"Total wall clock runtime: {elapsed_time:.2f} seconds for {n_inference_iterations} inference iterations, {n_posterior_predictive} posterior samples, and {n_prior_predictive} prior samples"
    )
    # save approx, hist, trace, trace_prior

    # print(dir(approx))
    # print(dir(approx.approx))

    # Extract necessary attributes from approx
    # approx_params = {
    #     'approx': approx.approx,
    #     #'fit': approx.fit,
    #     'hist': approx.hist,
    #     'objective': approx.objective,
    #     #'refine': approx.refine,
    #     #'run_profiling': approx.run_profiling,
    #     #'state': approx.state,
    # }

    # Save approx_params, hist, trace, trace_prior separately
    with gzip.open(vi_file, "wb") as f:
        pickle.dump({"hist": hist, "trace": trace, "trace_prior": trace_prior}, f)

    # with gzip.open(vi_file, 'wb') as f:
    #     #pickle.dump({'approx': approx_params}, f)
    #     pickle.dump({'hist': hist}, f)
    #     pickle.dump({'trace': trace}, f)
    #     pickle.dump({'trace_prior': trace_prior}, f)

    # make ELBO plot
    import matplotlib.pyplot as plt

    with gzip.open(vi_file, "rb") as f:
        results = pickle.load(f)
        hist = results["hist"]
    plt.semilogy(hist.hist, ".", ms=2, alpha=0.8)
    plt.ylabel("Evidence Lower Bound\n(ELBO)")
    plt.xlabel("Iteration")
    plt.tight_layout()
    plt.savefig(f"{run_directory}/elbo.png")

    # TODO: make FCC plot

    logger.info(
        f"Run completed successfully. Output files saved in {run_directory}. Final timestmp: {datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
