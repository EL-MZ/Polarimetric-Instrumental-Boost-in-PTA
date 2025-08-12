#!/bin/python

import numpy as np
from enterprise_warp import enterprise_warp
from enterprise_warp.enterprise_warp import get_noise_dict


import ppta_dr2_models 

# ─── Monkey-patch EntryPoint.default_kwargs ────────────────────────────────────
try:
    import entrypoints
    entrypoints.EntryPoint.default_kwargs = property(
        lambda self: getattr(self.load(), "__kwdefaults__", {}) or {}
    )
except ImportError:
    pass

try:
    # In case enterprise_warp is using importlib.metadata.EntryPoint instead
    import importlib.metadata as _im
    _im.EntryPoint.default_kwargs = property(
        lambda self: getattr(self.load(), "__kwdefaults__", {}) or {}
    )
except (ImportError, AttributeError):
    pass
# ────────────────────────────────────────────────────────────────────────────────

import os
import emcee
import multiprocessing 


opts = enterprise_warp.parse_commandline()

custom = ppta_dr2_models.PPTADR2Models

params = enterprise_warp.Params(opts.prfile,opts=opts,custom_models_obj=custom)
pta = enterprise_warp.init_pta(params)
print('Pulsar Timing Array: ', len(pta))
#super_model = hypermodel.HyperModel(pta)

x0 = np.hstack([p.sample() for p in pta[0].params])
ndim = len(x0)
print('ndim: ', ndim)
cov = np.diag(np.ones(ndim) *1**2)
print('Super model parameters: ', pta[0].params)

N = params.nsamp
print('Number of samples is :', N)
print('Number of dimensions is :', ndim)
print('initial parameters: ', x0)


def log_posterior(theta):

    return pta[0].get_lnlikelihood(theta)+pta[0].get_lnprior(theta)



output_folder = params.output_dir 

# Set up the MCMC sampler
nwalkers = ndim*3
nsteps = params.nsamp
os.makedirs(output_folder, exist_ok=True)
filename = os.path.join(output_folder, "mcmc_run.h5")
backend = emcee.backends.HDFBackend(filename)
backend.reset(nwalkers, ndim)
print(f"Chains will be saved to: {filename}")

# --- 4. Set up and Run the Sampler with Multiprocessing ---

# Get the number of CPUs allocated by SLURM.
# os.environ.get will return None if the variable is not set.
# We default to 1 core if not running on SLURM.
n_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
print(f"Running with {n_cpus} CPUs.")

# A 'with' statement is the safest way to manage the pool.
# It ensures the pool is properly closed even if errors occur.
with multiprocessing.Pool(n_cpus) as pool:
    
    # Instantiate the sampler, passing the pool object to the 'pool' argument.
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, 
                                    args=[], 
                                    backend=backend, 
                                    pool=pool)

    print("Running MCMC with multiprocessing...")
    # Run the MCMC. emcee will now use the pool to parallelize the likelihood calls.
    x0 = np.hstack([p.sample() for p in pta[0].params])
    ll =pta[0].get_lnlikelihood(x0)
    while np.isnan(ll) or np.isinf(ll):
        x0 = np.hstack([p.sample() for p in pta[0].params])
        ll =pta[0].get_lnlikelihood(x0)
    print("Found a non-infinite starting point for the MCMC run.")
    initial_pos = x0 + 1e-3 * np.random.randn(nwalkers, ndim)
    state = sampler.run_mcmc(initial_pos, nsteps, progress=True)
    print("MCMC run complete.")

