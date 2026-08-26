import numpy as np
import os

def save_chain(filename, samples, log_likelihood, log_prior):
    """
    Saves the chain data (samples, log_likelihood, log_prior) to a text file.
    
    The output file will have the samples in the first D columns,
    followed by log_likelihood and log_prior in the last two columns.
    
    Parameters:
    -----------
    filename : str
        Path to the output file.
    samples : array_like
        The samples from the chain. Shape (N, D).
    log_likelihood : array_like
        The log likelihood values. Shape (N,).
    log_prior : array_like
        The log prior values. Shape (N,).
    """
    samples = np.asarray(samples)
    log_likelihood = np.asarray(log_likelihood)
    log_prior = np.asarray(log_prior)
    
    # Ensure they have consistent lengths
    if not (len(samples) == len(log_likelihood) == len(log_prior)):
        raise ValueError("samples, log_likelihood, and log_prior must have the same length.")
    
    # Stack them horizontally: [samples, logl, logp]
    # Reshape 1D arrays to (N, 1) to stack
    data = np.hstack([
        samples, 
        log_likelihood.reshape(-1, 1), 
        log_prior.reshape(-1, 1)
    ])
    
    # Save to text file
    # Using a header to describe the columns is good practice
    header = "samples (cols 0:-2) | log_likelihood (col -2) | log_prior (col -1)"
    np.savetxt(filename, data, header=header)
    print(f"Saved chain to {filename}")

def read_chain(filename):
    """
    Reads chain data from a text file.
    
    Parameters:
    -----------
    filename : str
        Path to the file to read.
        
    Returns:
    --------
    dict:
        {'samples': np.ndarray, 'log_likelihood': np.ndarray, 'log_prior': np.ndarray}
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File {filename} not found.")
        
    data = np.loadtxt(filename)
    
    if data.ndim == 1:
        # Handle case with single sample or flattened? 
        # Usually chains have many samples. If only 1 sample, shape is (D+2,)
        # We probably want to return shape (1, D)
        data = data.reshape(1, -1)
        
    # The last two columns are log_likelihood and log_prior
    samples = data[:, :-2]
    log_likelihood = data[:, -2]
    log_prior = data[:, -1]
    
    return {
        "samples": samples,
        "log_likelihood": log_likelihood,
        "log_prior": log_prior
    }



# samples = post["samples"]
# log_prob_samples = post["log_likelihood"]+ post["log_prior"]
