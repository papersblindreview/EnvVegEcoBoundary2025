import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import geopandas as gpd 
from scipy import optimize, stats
from scipy.stats import gamma, lognorm, norm, halfnorm, dirichlet
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from geopandas import GeoDataFrame
from shapely.geometry import Point, Polygon, box, LineString
from shapely.ops import unary_union
import seaborn as sns
import netCDF4 as nc
import cmasher as cmr
from pyproj import Transformer
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pyreadr
import os
import pymc as pm
import pytensor.tensor as pt
import pytensor
pd.options.mode.chained_assignment = None
pd.set_option('future.no_silent_downcasting', True)

global current_directory
current_directory = os.getcwd()

# constructing the dataframe containing the state borders, for plotting purposes
class UMW:
  def __init__(self, filepath):
    self.filepath = filepath
      
  def get_map(self):
    districts = gpd.read_file(current_directory + self.filepath)
    districts = districts.sort_values(by='STATEFP')
    districts['STATEFP'] = districts['STATEFP'].astype('int')
    districts = districts[districts['STATEFP'].isin([26,27,55])]
    
    michigan = gpd.GeoSeries(unary_union(districts[districts['STATEFP']==26].geometry))
    minnesota = gpd.GeoSeries(unary_union(districts[districts['STATEFP']==27].geometry))
    wisconsin = gpd.GeoSeries(unary_union(districts[districts['STATEFP']==55].geometry))
    
    umw = pd.concat([michigan, minnesota, wisconsin], ignore_index=True)
    umw = gpd.GeoDataFrame(umw, geometry=umw, crs='EPSG:4326')
    umw[0] = ['MI','MN','WI']
    umw.columns = ['state', 'geometry']
    return umw
 
        
# loading and cleaning the PLS vegetation data
class PLS:
  def __init__(self, filepath, truncate_up, umw):
    self.filepath = filepath
    self.truncate_up = truncate_up
    self.umw = umw
      
  def get_data(self):
    # (2) loading and cleaning the PLS vegetation data
    result = pyreadr.read_r(current_directory + self.filepath)
    k = list(result.keys())
    pls_raw_data = result[k[0]]
    pls_raw_data.drop(columns='id', inplace=True)
    
    transformer = Transformer.from_crs("EPSG:3175", "EPSG:4326")
    pls_xy = transformer.transform(pls_raw_data.iloc[:,0], pls_raw_data.iloc[:,1])
    pls_xy = np.array(pls_xy).T
    pls_raw_data.drop(columns=['x','y'], inplace=True)

    geometry_pls = [Point(xy) for xy in zip(pls_xy[:,1], pls_xy[:,0])]
    pls_data = GeoDataFrame(pls_raw_data, crs="EPSG:4326", geometry=geometry_pls)
    pls_data = pls_data.to_crs(epsg=4326)     

    pls_data = gpd.overlay(pls_data, self.umw)  
    pls_data.reset_index(inplace=True, drop='True')
    pls_data.drop(columns='state', inplace=True)
    
    tree_taxa_all = [c for c in pls_data.columns.to_list() if c not in ['density','geometry']]

    pls_data.density = np.where(pls_data.density > self.truncate_up, self.truncate_up, pls_data.density)
    pls_data[tree_taxa_all] = np.where(pls_data[tree_taxa_all] < 1/pls_data.density.max(), 0, pls_data[tree_taxa_all])
    
    return pls_data, tree_taxa_all
  
        
# loading pollen data to get pollen site coordinates
class SITES:
  def __init__(self, filepath):
    self.filepath = filepath
      
  def get_coords(self):
    with open(current_directory + self.filepath, 'rb') as f:
        pollen_data = pickle.load(f)
    
    pollen_only = pollen_data.drop(columns=['lon','lat','RC_yrs','region'])
    coordinates = pollen_data.groupby(['lon', 'lat']).size().reset_index(name='Freq')
    coordinates = coordinates.sort_values(by=['Freq'], ascending=False).reset_index(drop=True)
    geometry_sites = [Point(xy) for xy in zip(coordinates.lon, coordinates.lat)]
    coordinates = coordinates.drop(['lon', 'lat'], axis=1)
    return GeoDataFrame(coordinates, crs="EPSG:4326", geometry=geometry_sites)
 
        
# load and cleanup environmental data     
class ENVIRONMENT:
  def __init__(self, filepath):
    self.filepath = filepath
      
  def load_dataset(self, state, var_type):
    result = pyreadr.read_r(current_directory + f'/data/env/processed_{var_type}_{state}.RData')
    k = list(result.keys())
    return result[k[0]]
      
  def get_env_var(self, env_type):
    df = self.load_dataset('mi', env_type)
    df = pd.concat([df, self.load_dataset('mn', env_type)], axis=0, ignore_index=True)
    df = pd.concat([df, self.load_dataset('wi', env_type)], axis=0, ignore_index=True)
    if env_type == 'flood':
      df.replace('No', 0, inplace=True)
      df.replace('Yes', 1, inplace=True)

    df.reset_index(drop=True, inplace=True)
    geometry_env = [Point(xy) for xy in zip(df['pls_x'], df['pls_y'])]
    df = df[[c for c in df.columns if '_x' not in c]]
    df = df[[c for c in df.columns if '_y' not in c]]
    
    return df, geometry_env
      
      
  def get_data(self, veg_pls, env_types, env_vars):

    env_data, geometry_env = self.get_env_var(env_types[0])
    for v in env_types[1:]:
      temp_df, _ = self.get_env_var(v)
      env_data = pd.concat([env_data, temp_df], axis=1)

    env_data = env_data[env_vars]
    env_data = GeoDataFrame(env_data, crs="EPSG:4326", geometry=geometry_env)
    
    return env_data
        

class DATA_LOADER:
  def __init__(self, coordinates, veg_pls, taxa_all, env_data, env_vars):
    self.coordinates = coordinates
    self.veg_pls = veg_pls.to_crs(epsg=4326)
    self.taxa_all = taxa_all
    self.env_data = env_data.to_crs(epsg=4326)
    self.env_vars = env_vars
    
  def get_modern_env_var(self):
    with open(current_directory + f'/data/env/modern_temp.pkl', 'rb') as f:
      df = pickle.load(f)
      df.columns = ['tmean_mean', 'geometry']
      
    with open(current_directory + f'/data/env/modern_ppt.pkl', 'rb') as f:
      df_temp = pickle.load(f)
      df['ppt_mean'] = df_temp.iloc[:,0]
      
    return df
      
  def get_taxa(self):
    taxa_train = self.veg_pls.describe().T.sort_values(by='std', ascending=False).iloc[1:,:].index[:10].to_list()
    return list(taxa_train)
    
  def match_env_pls(self):
    coords_env = np.stack([self.env_data.geometry.x, self.env_data.geometry.y]).T
    coords_pls = np.stack([self.veg_pls.geometry.x, self.veg_pls.geometry.y]).T
    
    k = coords_env.shape[0] // coords_pls.shape[0]

    tree = cKDTree(coords_env)
    distances, indices = tree.query(list(coords_pls), k=k)

    closest_points_to_pls = coords_env[indices.flatten()]
    
    env_data_on_pls = np.zeros((self.veg_pls.shape[0], len(self.env_vars)))
    for row in self.veg_pls.index:
      nearest_values = self.env_data.loc[indices[row],self.env_vars]
      local_averages = nearest_values.mean(axis=0)
      env_data_on_pls[row] = local_averages
      
    env_pls_data = self.veg_pls.copy()
    env_pls_data[self.env_vars] = env_data_on_pls
    return env_pls_data
      
  def load_train(self, train_size, val_size, seed):
    np.random.seed(seed)
    taxa = self.get_taxa()
    env_pls_data = self.match_env_pls()
    density_data = env_pls_data[['density']]
  
    veg_data = env_pls_data.loc[:, taxa] 
    veg_data['other'] = 1-env_pls_data[taxa].sum(axis=1)

    coords_pls = np.stack([env_pls_data.geometry.x, env_pls_data.geometry.y]).T
    
    if val_size < 0:
      train_val_ids = np.random.permutation(np.arange(0,env_pls_data.shape[0]))
    else:
      train_val_ids = np.random.permutation(np.arange(0,env_pls_data.shape[0]))[:(train_size+val_size)]
      
    coords_train_val = coords_pls[train_val_ids,:]
    veg_data_train_val = veg_data.loc[train_val_ids,:].reset_index(drop=True)
    env_data_train_val = env_pls_data.loc[train_val_ids, self.env_vars].reset_index(drop=True)
    density_train_val = env_pls_data.loc[train_val_ids, 'density'].reset_index(drop=True)
      
    s_env_veg_train_val = pd.concat([pd.DataFrame(coords_train_val), env_data_train_val, veg_data_train_val], axis=1)
    s_env_veg_train_val.columns = ['lon','lat'] + self.env_vars + taxa + ['other']
    
    X_train = s_env_veg_train_val.iloc[:train_size,:]
    y_train = density_train_val.iloc[:train_size]
    if val_size > 0:
      X_val = s_env_veg_train_val.iloc[train_size:,:]
      y_val = density_train_val.iloc[train_size:]
    else:
      X_val = s_env_veg_train_val
      y_val = density_train_val
        
    return X_train, y_train, X_val, y_val 


def _hurdle_mixture(*, name, nonzero_p, nonzero_dist, dtype, **kwargs):
  if dtype == "float":
      zero = 0.0
      lower = np.finfo(pytensor.config.floatX).eps
  elif dtype == "int":
      zero = 0
      lower = 1
  else:
      raise ValueError("dtype must be 'float' or 'int'")

  nonzero_p = pt.as_tensor_variable(nonzero_p)
  weights = pt.stack([1 - nonzero_p, nonzero_p], axis=-1)
  comp_dists = [pm.DiracDelta.dist(zero), pm.Truncated.dist(nonzero_dist, lower=lower)]

  if name is not None:
    return pm.Mixture(name, weights, comp_dists, **kwargs)
  else:
    return pm.Mixture.dist(weights, comp_dists, **kwargs)


class HurdleBeta:
  def __new__(cls, name, psi, alpha=None, beta=None, mu=None, sigma=None, **kwargs):
      return _hurdle_mixture(name=name, nonzero_p=psi,
          nonzero_dist=pm.Beta.dist(alpha=alpha, beta=beta, mu=mu, sigma=sigma), dtype="float", **kwargs)

  @classmethod
  def dist(cls, psi, alpha=None, beta=None, mu=None, sigma=None, **kwargs):
    return _hurdle_mixture(name=None, nonzero_p=psi,
      nonzero_dist=pm.Beta.dist(alpha=alpha, beta=beta, mu=mu, sigma=sigma), dtype="float", **kwargs)

def load_validation_results(suffix, taxa_names, naive):
  results_vals = []  
  print(f'\nScenario {suffix[-1]}')
  by = 400 
  naive_str = 'Naive' if naive else ''
  for i in np.arange(0,8000,by): 
    print(f'Loading {i} to {i+by}...')
    with open(current_directory + f'/model_output/trace{naive_str}Post{suffix}_{i}to{i+by}.pkl', 'rb') as f:
      trace_post_temp = pickle.load(f) 
    with open(current_directory + f'/model_output/dataVal{suffix}_{i}to{i+by}.pkl', 'rb') as f:
        data_val_temp = pickle.load(f)  
    
    try:
      results_val_temp = data_val_temp.iloc[:,:2] 
    except:
      results_val_temp = data_val_temp['s_env_veg'].iloc[:,:2] 
    results_val_temp['w_s_pred'] = np.array(trace_post_temp.predictions['w_s_val']).mean(axis=(0,1))
    results_val_temp['w_s_uq'] = np.array(trace_post_temp.predictions['w_s_val']).std(axis=(0,1))
    results_val_temp[taxa_names] = np.array(trace_post_temp.predictions['veg_val']).mean(axis=(0,1))[:,:-1]
    results_vals.append(results_val_temp)
  
  results_val = pd.concat(results_vals, axis=0, ignore_index=True)
  results_val = GeoDataFrame(results_val, crs="EPSG:4326", geometry=[Point(xy) for xy in zip(results_val.iloc[:,0], results_val.iloc[:,1])])
  return results_val

def get_boundary(data, umw):
  boundary = data[(data['w_s_pred'] < 0.55) & (data['w_s_pred'] > 0.45)]
  poly = np.poly1d(np.polyfit(boundary.geometry.x, boundary.geometry.y, 7))
  grid = np.linspace(boundary.geometry.x.min(), boundary.geometry.x.max(), 500)
  curve = poly(grid)
  curve_gpd = GeoDataFrame(crs="EPSG:4326", geometry=[Point(xy) for xy in zip(grid, curve)])
  curve_gpd = GeoDataFrame(crs="EPSG:4326", geometry=[LineString(curve_gpd.geometry.values)]).intersection(umw.unary_union)
  return curve_gpd



        
