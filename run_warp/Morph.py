import numpy as np

from scipy.stats import  gaussian_kde
from scipy.stats import norm
from scipy.special import logsumexp
from scipy.signal import correlate
from statsmodels.tsa.stattools import acf


class KDE_approx(object):
    def __init__(self,data, bw='silverman', param_names=None):

        #coup = GaussianMultivariate(distribution=GaussianKDE)
        #coup.fit(data)

        data = np.asarray(data)
        if data.ndim != 2:
            raise ValueError("data must be a 2D array-like object with shape (n_samples, n_params)")
        
        n_samples, n_params = data.shape

        if param_names is None:
            param_names = [f"{i}" for i in range(n_params)]
        elif len(param_names) != n_params:
            raise ValueError("Length of param_names must equal the number of parameters (columns) in data")

        self.kde_dict = {}
        for i, name in enumerate(param_names):
            # Extract the i-th parameter (all samples for that parameter)
            param_data = data[:, i]
            # Fit a Gaussian KDE
            kde_obj = gaussian_kde(param_data,bw_method= bw)#'silverman'
            # Save the callable KDE object in the dictionary
            self.kde_dict[name] = kde_obj

    def logpdf_kde(self,point):
        
        
        logpdf = 0
        for name, kde_obj in self.kde_dict.items():
            # Evaluate the logpdf of the i-th parameter at the i-th coordinate of the point
            logpdf += kde_obj.logpdf(point[int(name)])
        return logpdf#- coup.log_probability_density(point)

    def resample(self,size=1):
        if size == 1:
            resampled = np.zeros(len(self.kde_dict))
            for i, kde_obj in self.kde_dict.items():
                resampled[int(i)] = kde_obj.resample(1)
        else:
            resampled = np.zeros((len(self.kde_dict),size))
            for i, kde_obj in self.kde_dict.items():
                resampled[int(i),:] = kde_obj.resample(size)
        return resampled



def bridge_sampling_ln(f, g, samples_post,log_post, samples_prop, tol=1e-6, max_iter=40000):
    """
    Estimate log marginal likelihood log p(y) using bridge sampling in log-space.
    
    Args:
        f: function f(theta) returning the unnormalized density p(y|theta)*p(theta)
        g: function g(theta) returning the proposal density g(theta)
        samples_post: posterior samples (N1 x d numpy array)
        samples_prop: proposal samples (N2 x d numpy array)
        tol: convergence tolerance (absolute change in log p)
        max_iter: maximum iterations
        
    Returns:
        log_p: estimated log marginal likelihood
        n_iter: number of iterations taken to converge
    """
    post_iter= samples_prop.shape[1]
    # Compute logs of densities
    log_f_post = log_post
    log_g_post = g(samples_post.T)
    #print('kde',log_g_post)
    log_f_prop = []
    for idx, theta in enumerate(samples_prop.T):
        print(f"\r Iteration {idx+1} out of {post_iter}", end="")
        result = f(theta)
        log_f_prop.append(result)
    log_f_prop = np.array(log_f_prop)

    finite_mask = np.isfinite(log_f_prop)
    log_f_prop = log_f_prop[finite_mask]
    
    log_g_prop = g(samples_prop)
    log_g_prop = log_g_prop[finite_mask]
    print(f"\n{len(log_f_prop)} surviving proposal samples out of {len(samples_prop.T)}")

    N1 = len(log_f_post)
    #print(N1)
    N2 = len(log_f_prop)
   
    s1 = N1 / (N1 + N2)
    s2 = N2 / (N1 + N2)

    # ---------- initial guess via IS on proposal draws ------------------------
    log_p_old = logsumexp(log_f_prop - log_g_prop) - np.log(len(log_f_prop))
    print(f'\n Computing initial log p(y) using proposal samples initial guess:{log_p_old}')
    #print(log_p_old)
    term1 = np.log(s1) + log_f_prop
    term1_post = np.log(s1) + log_f_post
    
    log_z = log_p_old
    print('Bridging')
    for t in range(max_iter):

        # For proposal samples:
        # Compute log numerator terms:
        # log_term_i = log f_prop[i] - log( s1 * f_prop[i] + s2 * exp(log_p_old) * g_prop[i] )
        # Use log-sum-exp trick for the denominator:
         # log(s1 * f_prop)
        term2 = np.log(s2) + log_p_old + log_g_prop
        #log_den_prop = np.array([log_plus(term1[jj],term2[jj]) for jj in range(len(term1))])
        #log_den_prop = np.array([log_plus(term1[jj],term2[jj]) for jj in range(len(term1))])
        log_den_prop = logsumexp(np.vstack((term1, term2)), axis=0)
        log_terms_prop = log_f_prop - log_den_prop  # log of individual terms
        
        log_num = -np.log(N2) + log_sum(log_terms_prop)
        
        # For posterior samples:
        term2_post = np.log(s2) + log_p_old + log_g_post

        #log_den_post = np.array([log_plus(term1_post[jj],term2_post[jj]) for jj in range(len(term1_post))])
        log_den_post = logsumexp(np.vstack((term1_post, term2_post)), axis=0)

        log_terms_post = log_g_post - log_den_post
        
        log_den = -np.log(N1) + log_sum(log_terms_post)
        
        log_p_new = log_num - log_den
        log_z= np.append(log_z,log_p_new)
        print(f"\r iteration: {t+1} log(z) old: {log_p_old} log(z) New: {log_p_new}", end="")
        
        # Check convergence:
        if np.abs(log_p_new - log_p_old) < tol:
            log_p_old = log_p_new
            rmse_est = compute_bridge_rmse(log_p_new, f, g,
        samples_prop, samples_post,
        log_f_prop, log_g_prop,log_f_post, log_g_post,s1, s2)
            print(f"\r iteration: {t+1} log(z): {log_p_new} +/-: {rmse_est}", end="")
            
            return [log_p_new, rmse_est],log_z
        
        log_p_old = log_p_new
    rmse_est = compute_bridge_rmse(
    log_p_new, f, g,
    samples_prop, samples_post,
    log_f_prop, log_g_prop,
    log_f_post, log_g_post,
    s1, s2)
    return [log_p_new, rmse_est],log_z

    #raise RuntimeError("Bridge sampling (log version) did not converge within max_iter iterations")
def compute_rho_f2_0_via_statsmodels(f2_values, nlags=None):
    """
    Estimate the integrated autocorrelation time for a 1D array of f2 values using statsmodels.acf.
    
    The integrated autocorrelation time is defined as:
        tau = 1 + 2 * sum_{lag>=1} acf(lag)
    where the sum stops at the first negative value.
    
    Args:
        f2_values: 1D numpy array of f2(theta) values computed from posterior samples.
        nlags: Maximum number of lags to compute (if None, defaults to len(f2_values) - 1).
    
    Returns:
        tau: An estimate of the integrated autocorrelation time (rho_f2(0)).
    """
    if nlags is None:
        nlags = len(f2_values) - 1
    # Compute the autocorrelation function using fft for speed.
    acf_values = acf(f2_values, nlags=nlags, fft=True)
    # Start with lag 0 which is 1; then sum positive autocorrelations until the first negative value.
    tau = 1.0
    for lag in range(1, len(acf_values)):
        if acf_values[lag] > 0:
            tau += 2 * acf_values[lag]
        else:
            break
    return tau

def compute_rho_f2_0_via_correlate(f2_values):

    x = f2_values - np.mean(f2_values)
    n = x.size
    # Full correlation has length 2*n - 1
    corr = correlate(x, x, mode='full')
    # The 'center' index for lag=0:
    mid = n - 1
    # Keep only nonnegative lags: corr[mid:] => lags 0,1,...,n-1
    corr = corr[mid:]
    # Normalize so corr[0] = 1
    corr /= corr[0]

    # Sum the positive portion of corr
    # (some strategies sum all lags until correlation first becomes negative).
    sum_pos = 0.0
    for val in corr[1:]:  # skip lag0 because it's 1
        if val > 0:
            sum_pos += val
        else:
            break

    # Integrated autocorr time approx: tau = 1 + 2 * sum_{k>0} corr(k)
    tau = 1.0 + 2.0 * sum_pos
    return tau

def compute_bridge_rmse(
    log_p_final,
    f, g,
    samples_prop, samples_post,
    log_f_prop, log_g_prop,
    log_f_post, log_g_post,
    s1, s2,
    posterior_acf_func=compute_rho_f2_0_via_correlate
):

    # Use epsilon safeguard to avoid division by 0 (or extremely small numbers)
    eps = 1e-12

    N1 = len(log_f_post)
    #print(N1)
    N2 = len(log_f_prop)
    
    # --- For PROPOSAL SAMPLES ---
    # Compute log p(theta|y) = log_f_prop - log_p_final
    lp_py_prop = log_f_prop - log_p_final
    left_part  = np.log(s1) + lp_py_prop
    right_part = np.log(s2) + log_g_prop
    log_denom_prop = logsumexp(np.vstack((left_part, right_part)), axis=0)
    log_f1_prop = lp_py_prop - log_denom_prop

    # Safe exponentiation: shift by maximum to avoid overflow
    max_log_f1 = np.max(log_f1_prop)
    f1_prop_shifted = np.exp(log_f1_prop - max_log_f1)
    
    mean_f1_prop = log_f1_prop.mean()
    #print('mean_f1_prop:',mean_f1_prop)
    var_f1_prop = log_f1_prop.var(ddof=1)
    #print('var_f1_prop:',var_f1_prop)
    # --- For POSTERIOR SAMPLES ---
    lp_py_post = log_f_post - log_p_final
    left_part_post  = np.log(s1) + lp_py_post
    right_part_post = np.log(s2) + log_g_post
    log_denom_post = logsumexp(np.vstack((left_part_post, right_part_post)), axis=0)
    log_f2_post = log_g_post - log_denom_post
    #plt.plot(log_f2_post)
    
    max_log_f2 = np.max(log_f2_post)
    f2_post_shifted = np.exp(log_f2_post - max_log_f2)
    mean_f2_post = log_f2_post.mean()
    #print('mean_f2_post:',mean_f2_post)
    var_f2_post = log_f2_post.var(ddof=1)
    #print('var_f2_post:',var_f2_post)
    # --- Autocorrelation correction ---
    if posterior_acf_func is not None:
        # Convert to real scale
        rho_f2_0 = posterior_acf_func(f2_post_shifted)
        print('rho_f2_0:',rho_f2_0)

    else:
        rho_f2_0 = 1.0
    # --- Compute relative MSE using the formula ---
    term1 = (var_f1_prop / ((mean_f1_prop + eps)**2)) / N2
    #print('term1:',term1)
    term2 = (rho_f2_0 * var_f2_post / ((mean_f2_post + eps)**2)) / N1
    #print('term2:',term2)
    re2 = term1 + term2

    rmse = np.sqrt(re2)
    return rmse
def log_plus(x,y):
    
    if x > y:
      summ = x + np.log(1+np.exp(y-x))
    else:
        summ = y + np.log(1+np.exp(x-y))
    return summ

def log_sum(vec): 
    r = -np.Inf
    for i in range(len(vec)):
       #print('element:',vec[i])
       r =log_plus(r, vec[i])
       #print(r)
    return r
def error_bound_from_oscillation(x):
    """
    Given a sequence x of iterates (assumed to be oscillatory about the fixed point)
    this function returns an error bound computed as half the distance between the min
    and max of the last two iterates.
    discard 20% of the iterates to avoid the initial transient.
    Parameters:
       x : list or np.array
           A sequence of iterates.
    
    Returns:
       err_bound : float
           An error bound estimate: (max(x) - min(x)) .
    """
    x = np.array(x, dtype=float)
    x = x[int(0.2*len(x)):] 
    if len(x) < 2:
        raise ValueError("Need at least two iterates to compute oscillation bounds.")
    lower = min(x)
    upper = max(x)
    return (upper - lower) 