# Packages
## Data wrangling and math
import numpy as np

import pandas as pd

import scipy

import xarray as xr

## Modelling
import sklearn

## Visualisation
import matplotlib
import matplotlib.pyplot as plt

import seaborn as sns

## Versioning
import sys

print(sys.version) # 3.12
print(np.__version__) # 1.26
print(pd.__version__) # 2.2
print(scipy.__version__) # 1.13
print(xr.__version__) # 2023.6
print(sklearn.__version__) # 1.5
print(matplotlib.__version__) # 3.9
print(sns.__version__) # 0.13

# Data Processing
## Reading in the data



# The general model
"""
HFI = (1-HQ) + PD + NTL_DS + R

HFI - Human Footprint Index

HQ - Habitat quality

PD - Population Density

NTL_DS - Nighttime light

R - Road Impact


"""

## Comparing Regions

### SPSS1
f, ax = plt.subplots(1, 3, sharey=True)
g = sns.scatterplot()

### SPSS4.5
f, ax = plt.subplots(1, 3, sharey=True)
g = sns.scatterplot()


### SPSS8
f, ax = plt.subplots(1, 3, sharey=True)
g = sns.scatterplot()


# Evaluation

## Model Metrics

### RMSE

### AIC

### BIC

## Feature Importance

# The extended model
"""
HFI = (1-HQ) + PD + NTL_DS + R + ...

The extended model has the purpose of investigating the possibility of 
expanding on the original HFI model to be further representative. 
However the model as is is already including the most prominent aspects.
"""
