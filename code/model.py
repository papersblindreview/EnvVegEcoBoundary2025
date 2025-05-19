import scipy
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import geopandas as gpd 
import pymc as pm
from pymc import model_to_graphviz
import pytensor.tensor as pt
import pytensor
from scipy import optimize, stats
from scipy.stats import gamma, lognorm, norm, halfnorm, dirichlet
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from geopandas import GeoDataFrame
from shapely.geometry import Point
from shapely.ops import unary_union
import netCDF4 as nc
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pyproj import Transformer
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pyreadr
import os
import time
from functions import *
current_directory = os.getcwd()
pd.options.mode.chained_assignment = None
seed = 123

# some temporal info on pymc here
# https://juanitorduz.github.io/short_time_series_pymc/

## HYPER PARAMETERS
train_size = 220
val_size = -1 #"-1" if all sites for validation
n_vi = 15000
n_vi_draws = 1000
lr = 0.005 
other = True
opt = pm.adam(learning_rate=lr)

#################
# LOAD AND PREPROCESS DATA

# These functions are all saved in import_functions.py

# load MI, MN, WI state borders
umw = UMW(filepath='/data/US_MAP/cb_2018_us_cd116_500k.shp').get_map()

# load PLS data
veg_pls, taxa_all = PLS(filepath='/data/8km.RData', truncate_up=800, umw=umw).get_data()

# load pollen sites coordinates
coordinates = SITES(filepath='/data/df_pollen.pkl').get_coords()

# load environmental variables
env_types = ['soil','climate','flood','topo']
env_vars = ['tmean_mean','ppt_mean','sand','clay','elevation','Floodplain']
env_names = ['Temperature','Precipitation','Sand %','Clay %','Elevation','Floodplain']

env_pls = ENVIRONMENT(filepath='/data/env').get_data(veg_pls, env_types, env_vars)

data_loader = DATA_LOADER(coordinates, veg_pls, taxa_all, env_pls, env_vars)
taxa_train = data_loader.get_taxa()

s_env_veg_train, density_train, s_env_veg_val, density_val = data_loader.load_train(train_size=train_size, val_size=val_size, seed=seed)

scaler = MinMaxScaler(feature_range=(0,1))
n_env = len(env_vars)
taxa_names = s_env_veg_train.columns[-(s_env_veg_train.shape[1]-n_env-2):]
n_taxa = len(taxa_names)
idx = s_env_veg_train.shape[0]

scaler.fit(s_env_veg_train.iloc[:,:(n_env+2)])
s_env_veg_train_scaled = scaler.transform(s_env_veg_train.iloc[:,:(n_env+2)])


s_env_veg_train = s_env_veg_train.astype('float32')
s_env_veg_train_scaled = s_env_veg_train_scaled.astype('float32')

# MODEL SET UP

hyper_gamma = 1
hyper_beta = 1 
hyper_ell = 0.5
hyper_sigma = 0.25
hyper_chol = 0.5

coords = {'lon_lat': range(2),
          'tree_taxa': range(n_taxa),
          'env_vars': range(n_env),
          'states': range(2),
          'n': range(idx)}

with pm.Model(coords=coords) as model:

  # DATA
  s = pm.Data('sites', s_env_veg_train_scaled[:,:2], dims=('n','lon_lat'))
  z = pm.Data('environment', s_env_veg_train_scaled[:,2:], dims=('n','env_vars'))
  x = pm.Data('trees', s_env_veg_train[taxa_names], dims=('n','tree_taxa'))
  y = pm.Data('density', density_train, dims='n')
  
  # ENVIRONMENT -> VEGETATION
  ## Coefficients
  gammas = pm.Normal(f'gammas', mu=0, sigma=hyper_gamma, dims=('env_vars','tree_taxa'))
  omegas = pm.Normal('omegas', mu=0, sigma=hyper_gamma, dims='tree_taxa')
  sigmas_zero = pm.HalfCauchy(name='sigmas_zero', beta=hyper_sigma, dims='tree_taxa')
  
  ## Spatial dynamics
  ell_env_mat = pm.HalfCauchy(name='ell_env_mat', beta=hyper_ell, dims='tree_taxa')
  ell_env_exp = pm.HalfCauchy(name='ell_env_exp', beta=hyper_ell, dims='tree_taxa')
  thetas_env = pm.Normal('thetas_env', mu=0, sigma=hyper_gamma, dims=('lon_lat', 'tree_taxa'))
  sigma_env = pm.HalfCauchy(name='sigma_env', beta=hyper_sigma, dims='tree_taxa')
  
  ### On mean abundance
  covs_env = [sigma_env[i]*pm.gp.cov.Matern32(input_dim=2,ls=ell_env_mat[i])*pm.gp.cov.Exponential(input_dim=2,ls=ell_env_exp[i]) for i in range(len(taxa_names))]
  means_env = [pm.gp.mean.Linear(coeffs=thetas_env[:,i]) for i in range(len(taxa_names))]
  gps_env = [pm.gp.Latent(mean_func=means_env[i], cov_func=covs_env[i]) for i in range(len(taxa_names))]
  psi_env = [gps_env[i].prior(f"psi_env_{taxa_names[i]}", X=s, dims='n') for i in range(len(taxa_names))]
  
  ### On presence vs absence
  covs_zeros = [sigma_env[i]*pm.gp.cov.Matern32(input_dim=2,ls=ell_env_mat[i])*pm.gp.cov.Exponential(input_dim=2,ls=ell_env_exp[i]) for i in range(len(taxa_names))]
  means_zeros = [pm.gp.mean.Linear(coeffs=thetas_env[:,i]) for i in range(len(taxa_names))]
  gps_zeros = [pm.gp.Latent(mean_func=means_env[i], cov_func=covs_env[i]) for i in range(len(taxa_names))]
  psi_zeros = [gps_zeros[i].prior(f"psi_zeros_{taxa_names[i]}", X=s, dims='n') for i in range(len(taxa_names))]
  
  ## Taxa correlation
  chol, corr, precs = pm.LKJCholeskyCov(f'chol', n=n_taxa, eta=1.0, sd_dist=pm.HalfCauchy.dist(hyper_chol, shape=n_taxa), compute_corr=True)
  
  p_nonzeros = pm.LogitNormal('p_nonzeros', mu=omegas + pm.math.stack(psi_zeros).T, sigma=sigmas_zero, dims=('n', 'tree_taxa'))
  env_to_veg_mu = pm.Deterministic('env_to_veg_mu', pt.math.sigmoid(pt.dot(z, gammas) + pm.math.stack(psi_env).T), dims=('n', 'tree_taxa'))
  
  ## Abundance
  veg = HurdleBeta('veg', psi=p_nonzeros, alpha=env_to_veg_mu*precs, beta=(1-env_to_veg_mu)*precs, observed=x, dims=('n','tree_taxa'))
  
  ## Scaling for better identifiability
  veg_std = pt.math.sqrt((1-p_nonzeros)*p_nonzeros*(env_to_veg_mu**2) + p_nonzeros*env_to_veg_mu*(1-env_to_veg_mu)/(1+precs))
  veg_min = pt.math.min(veg, axis=0, keepdims=True)
  veg_max = pt.math.max(veg, axis=0, keepdims=True)
  #veg_scaled = pm.Deterministic('veg_scaled', (veg-veg_min) / (veg_max-veg_min), dims=('n','tree_taxa'))
  veg_scaled = pm.Deterministic('veg_scaled', env_to_veg_mu / veg_std, dims=('n','tree_taxa'))
  
  # VEGETATION -> ECOSYSTEM 
  
  ## Coefficients   
  betas = pm.Normal('betas', mu=0, sigma=hyper_beta, dims='tree_taxa')
  sigma_w = pm.HalfCauchy(name='sigma_w', beta=hyper_sigma)
  
  ## Spatial dynamics
  ell_veg_mat = pm.HalfCauchy(name=f'ell_veg_mat', beta=hyper_ell)
  ell_veg_exp = pm.HalfCauchy(name=f'ell_veg_exp', beta=hyper_ell)
  thetas_veg = pm.Normal(name='thetas_veg',  mu=0, sigma=hyper_beta, dims='lon_lat')
  sigma_veg = pm.HalfCauchy(name=f'sigma_veg', beta=hyper_sigma)

  cov_veg = sigma_veg*pm.gp.cov.Matern32(input_dim=2, ls=ell_veg_mat)*pm.gp.cov.Exponential(input_dim=2, ls=ell_veg_exp)
  gp_s = pm.gp.Latent(mean_func=pm.gp.mean.Linear(coeffs=thetas_veg), cov_func=cov_veg)
  psi_s = gp_s.prior("psi_s", X=s, dims='n')

  ## Weight of the prairie ecosystem
  w_s = pm.LogitNormal('w_s', mu=pt.dot(veg_scaled, betas) + psi_s, sigma=sigma_w, dims='n')
  mu_mix = pm.Uniform('mu', 100, 500)
  sigma_mix = pm.Gamma('sigma', alpha=4, beta=1/50)
  scale_mix = pm.Uniform('lambda', 0, 100)
  
  ## Mixture of end distributions
  components = [pm.Exponential.dist(scale=scale_mix), pm.Gamma.dist(mu=mu_mix, sigma=sigma_mix)]
  likelihood = pm.Mixture('y', w=pm.math.stack([1-w_s, w_s]).T, comp_dists=components, observed=y, dims='n')            
  
# TRAIN MODEL

start_time = time.time()

with model:
    advi_fit = pm.fit(n=n_vi, method='advi', random_seed=seed, obj_optimizer=opt, progressbar=True)
    trace = advi_fit.sample(n_vi_draws)
    trace_pred = pm.sample_posterior_predictive(trace, random_seed=seed, progressbar=False)

# SAVE RESULTS
taxa_names = [s.split(',')[0] for s in taxa_names]

data_train = {'s_env_veg': s_env_veg_train,
              'taxa_names': taxa_names,
              'env_names': env_names, 
              'density': density_train}

with open(os.getcwd() + f'/model_output/dataTrain.pkl', 'wb') as f:
    pickle.dump(data_train, f)

with open(os.getcwd() + f'/model_output/trace.pkl', 'wb') as f:
    pickle.dump(trace, f)
    
with open(os.getcwd() + f'/model_output/tracePred.pkl', 'wb') as f:
    pickle.dump(trace_pred, f)

