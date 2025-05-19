## Supplemental codes and data for "Modeling Midwestern Prairie-Forest Ecotone and Environment-Vegetation Feedback with a Hierarchical Bayesian Model".

The `data` folder contains the data used to train the model:
1) `8km.r` is the PLS data
2) `env` contains the environmental variables
3) `US_MAP` is an auxilliary dataset (containing a map of the continental US) to preprocess the data

The code folder contains the scripts to run and validate the model:
1) `functions.py` contains helper functions to load and preprocess the data
2) `model.py` contains code to run the hierarchical Bayesian model
3) `model_val.py` contains code to run validation and interpolate in space

To reproduce results, one should first download the files "data" folder. To reproduce the results from the manuscript, a user should first run the `model.py` script and then the `model_val.py` script.
