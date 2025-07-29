import numpy as np
import os
import glob


def read_post_pt(folder_path,param_name,burn_in=0):

   results = {}
   
   chain_list = []
   for filename in os.listdir(folder_path):
     if filename.startswith("chain_"):
        chain_list.append(filename)
       #print(chain_list)
   chain_list = [name.replace('chain_','') for name in chain_list]
   chain_list = [name.replace('.txt','') for name in chain_list]
   chain_list = sorted(chain_list,key=float)
   chain_list = [str('chain_')+name for name in chain_list]
   chain_list = [name+str('.txt') for name in chain_list]
   print('Calibrating using chains from file',chain_list[0])
   data = []
   filename = chain_list[0]
   full_path = os.path.join(folder_path, filename)
   temp = np.genfromtxt(full_path,dtype=float,unpack=True)
   ndim = len(temp)-4
   print('Number of dimensions :',ndim)
   for i in range(ndim):
    post = temp[i,burn_in:]
    data.append(post)
    results[param_name[i]] = post
   print('Posterior samples imported')
   return results


def read_pt_post_eval(folder_path,burn_in=0):

   results = {}
   
   chain_list = []
   for filename in os.listdir(folder_path):
     if filename.startswith("chain_"):
        chain_list.append(filename)
       #print(chain_list)
   chain_list = [name.replace('chain_','') for name in chain_list]
   chain_list = [name.replace('.txt','') for name in chain_list]
   chain_list = sorted(chain_list,key=float)
   chain_list = [str('chain_')+name for name in chain_list]
   chain_list = [name+str('.txt') for name in chain_list]
   print('Calibrating using chains from file',chain_list[0])
   data = []
   filename = chain_list[0]
   full_path = os.path.join(folder_path, filename)
   temp = np.genfromtxt(full_path,dtype=float,unpack=True)
   post = temp[-4,burn_in:]
   print('len post:',len(post))
   return post


def read_pt_lnlike_eval(folder_path,burn_in=0):

   results = {}
   
   chain_list = []
   for filename in os.listdir(folder_path):
     if filename.startswith("chain_"):
        chain_list.append(filename)
       #print(chain_list)
   chain_list = [name.replace('chain_','') for name in chain_list]
   chain_list = [name.replace('.txt','') for name in chain_list]
   chain_list = sorted(chain_list,key=float)
   chain_list = [str('chain_')+name for name in chain_list]
   chain_list = [name+str('.txt') for name in chain_list]
   print('Calibrating using chains from file',chain_list[0])
   data = []
   filename = chain_list[0]
   full_path = os.path.join(folder_path, filename)
   temp = np.genfromtxt(full_path,dtype=float,unpack=True)
   post = temp[-3,burn_in:]
   print('len post:',len(post))
   return post

def dens_evaluate(post_samples):
        
        li = list(post_samples.keys())
        ran_samples = np.zeros((len(li),len(post_samples.get(li[0]))))
        
        for j in range(len(li)):

              ran_samples[j,:]=post_samples.get(li[j])
              
        return ran_samples

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