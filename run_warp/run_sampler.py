#!/bin/python

import numpy as np
from enterprise_warp import enterprise_warp
from enterprise_warp.enterprise_warp import get_noise_dict

from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc

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
sampler = ptmcmc(ndim, pta[0].get_lnlikelihood, pta[0].get_lnprior, cov, 
                outDir=params.output_dir, resume=False)
N = params.nsamp
print('Number of samples is :', N)
print('Number of dimensions is :', ndim)
print('initial parameters: ', x0)

try:
      noisedict = get_noise_dict(psrlist=[pp.name for pp in params.psrs],
                                 noisefiles=params.noisefiles)
      x0 = sampler.informed_sample(noisedict)
      print('Informed sample')
except:
      print('Informed sample is not possible')

sampler.sample(x0, N, SCAMweight=30, AMweight=15, DEweight=50,thin=30)

