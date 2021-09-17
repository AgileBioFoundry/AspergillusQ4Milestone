import argparse
import pickle
import gzip

import pymc3 as pm

parser = argparse.ArgumentParser(description='learning rate')
parser.add_argument('--lr', type=float)
args = parser.parse_args()

with gzip.open('model.pz', 'rb') as f:
    pymc_model = pickle.load(f)

with pymc_model:
    approx = pm.ADVI()
    hist = approx.fit(
        n=40000,
        obj_optimizer=pm.adagrad_window(learning_rate=args.lr),
        total_grad_norm_constraint=100)
    trace = hist.sample(500)

    
with gzip.open(f"results_small_xnclip_{args.lr}.pz", 'wb') as f:
    pickle.dump(
        {'approx': approx,
         'hist': hist,
         'trace': trace}, f)
