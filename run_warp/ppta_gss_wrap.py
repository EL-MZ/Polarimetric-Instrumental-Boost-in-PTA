
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

import time
from mpi4py import MPI
import gss_mpi as gss_mpi

if __name__ == "__main__":
    Comm = MPI.COMM_WORLD
    Rank = Comm.Get_rank()
    Size = Comm.Get_size()

    opts = enterprise_warp.parse_commandline()

    custom = ppta_dr2_models.PPTADR2Models

    params = enterprise_warp.Params(opts.prfile,opts=opts,custom_models_obj=custom)
    pta = enterprise_warp.init_pta(params)
    print('Pulsar Timing Array: ', len(pta))
    #super_model = hypermodel.HyperModel(pta)

 
    print('Super model parameters: ', len(pta[0].params))


    
    mu0 = []
    sig0 = []
    folder_path = '/fred/oz103/ezahraoui/PPTA/ppta_posteriors/noeccor_7_psr_pol_cal_by_band_allpsr/0/'
    if Rank == 0 :
        posterior = {}
        params = np.genfromtxt(folder_path+'pars.txt',dtype=str,unpack=True)
        posterior = gss_mpi.read_post( folder_path, params, 13000)
        mu0 , sig0 = gss_mpi.ref_calibrate(posterior)
        del posterior
    mu0 = Comm.bcast(mu0,root = 0)
    sig0 = Comm.bcast(sig0,root = 0)

    start_time = time.time()
    n =  10
    nchain = Size #number of chain from N-core
    log_z = np.zeros(n)
    g_samp = 10000
    thin_fac = 100
    #new_path = folder_path.replace(model_name+'/','paper_results/')
    new_path = folder_path
    for i in range(n):
        
        GSS = gss_mpi.Gss_sampler(N_chain = nchain)

        log_z[i]= GSS.sample_is(n_samp = g_samp,mu0s = mu0 , sig0s= sig0, lnlikefn = pta[0].get_lnlikelihood, lnpriorfn = pta[0].get_lnprior, thin= thin_fac,parallel= True)
        print("the GSS_IS istimated log(z):", log_z[i])
        z_path = new_path+'log_z_'+str(g_samp)+'s_tf'+str(thin_fac)+'_'+str(n)+'_'+str(nchain)+'T.txt'
        if Rank == 0 :
            np.savetxt(z_path, log_z, fmt='%1.14e')
    stop_time = time.time()
    if Rank == 0 :
        #ss_z = gss_mpi.SS(q_b)
        #np.savetxt(z_path+'final', log_z, fmt='%1.14e')
        print("the GSS_IS istimated log(z):", log_z)
        #print("the SS_IS istimated log(z):", ss_z)
        
        print ("Time for GSS estimation ---", int((time.time()-start_time)), " seconds .---")