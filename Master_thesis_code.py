# Packages
## Data wrangling and math
import numpy as np

import pandas as pd

import scipy

import xarray as xr

import netCDF4 as nc
from netCDF4 import num2date

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

### EQI

#### LAI

#### GPP

#### FPC


### Population Density
pop_hist = nc.Dataset(r"C:\Data\Population Density\Control\population_histsoc_0p5deg_annual_1861-2005.nc4")

print(pop_hist.dimensions)

print(pop_hist.variables)

total_pop_hist = pop_hist.variables["number_of_people"]

time_pop_hist = pop_hist.variables["time"]

lat_pop_hist = pop_hist.variables["lat"]

lon_pop_hist = pop_hist.variables["lon"]

new_unit_hist = "days since 1861-01-01 00:00:00"
times_pop_hist = num2date(time_pop_hist[:], units=new_unit_hist, 
                          calendar=getattr(time_pop_hist, 
                                           "calendar", 
                                           "standard"))


latitudes = lat_pop_hist[:]

longitudes = lon_pop_hist[:]

latitudes_2d, longitudes_2d = np.meshgrid(latitudes, 
                                          longitudes, 
                                          indexing="ij")

latitudes_flat = latitudes_2d.flatten()

longitudes_flat = longitudes_2d.flatten()


pop_hist_df = pd.DataFrame({
"date": np.repeat(times_pop_hist, len(latitudes_flat)),
"latitude": np.tile(latitudes_flat, len(times_pop_hist)),
"longitude": np.tile(longitudes_flat, len(times_pop_hist)),
"total population": total_pop_hist[:].flatten()
})

##### Limiting to Europe

pop_hist_df.loc[... < pop_hist_df["latitude"] < ...]

pop_hist_df.loc[... < pop_hist_df["longitude"] < ...]

##### Saving Data

csv_filename = "total_population_historical_europe.csv"
pop_hist_df.to_csv(csv_filename, index=False)

print(f"Historical Population (all rows) saved to {csv_filename}")


### Scenario

#### SSP1
pop_ssp1 = nc.Dataset(r"C:\Data\Population Density\SSP1\population_ssp1soc_0p5deg_annual_2006-2100.nc4")

print(pop_ssp1.dimensions)

print(pop_ssp1.variables)

total_pop_ssp1 = pop_ssp1.variables["number_of_people"]

time_pop_ssp1 = pop_ssp1.variables["time"]

lat_pop_ssp1 = pop_ssp1.variables["lat"]

lon_pop_ssp1 = pop_ssp1.variables["lon"]

new_unit = "days since 2006-01-01 00:00:00"
times_pop_ssp1 = num2date(time_pop_ssp1, units=new_unit, 
                 calendar=getattr(time_pop_ssp1, 
                                  "calendar", 
                                  "standard"))


latitudes_ssp1 = lat_pop_ssp1[:]

longitudes_ssp1 = lon_pop_ssp1[:]

latitudes_2d_ssp1, longitudes_2d_ssp1 = np.meshgrid(latitudes_ssp1, 
                                          longitudes_ssp1, 
                                          indexing="ij")

latitudes_flat_ssp1 = latitudes_2d_ssp1.flatten()

longitudes_flat_ssp1 = longitudes_2d_ssp1.flatten()

pop_ssp1_df = pd.DataFrame({
"date": np.repeat(times_pop_ssp1, len(latitudes_flat_ssp1)),
"latitude": np.tile(latitudes_flat_ssp1, len(times_pop_ssp1)),
"longitude": np.tile(longitudes_flat_ssp1, len(times_pop_ssp1)),
"total population": total_pop_ssp1[:].flatten()
})

##### Limiting to Europe

pop_ssp1_df.loc[... < pop_ssp1_df["latitude"] < ...]

pop_ssp1_df.loc[... < pop_ssp1_df["longitude"] < ...]

##### Saving Data

csv_filename = "total_population_ssp1_europe.csv"
pop_ssp1_df.to_csv(csv_filename, index=False)

print(f"Population for SSP1 (all rows) saved to {csv_filename}")


#### SSP2
pop_ssp2 = nc.Dataset(r"C:\Data\Population Density\SSP3\population_ssp2soc_0p5deg_annual_2006-2100.nc4")

print(pop_ssp2.dimensions)

print(pop_ssp2.variables)

total_pop_ssp2 = pop_ssp2.variables["number_of_people"]

time_pop_ssp2 = pop_ssp2.variables["time"]

lat_pop_ssp2 = pop_ssp2.variables["lat"]

lon_pop_ssp2 = pop_ssp2.variables["lon"]

new_unit = "days since 2006-01-01 00:00:00"
times_pop_ssp2 = num2date(time_pop_ssp2, units=new_unit, 
                 calendar=getattr(time_pop_ssp2, 
                                  "calendar", 
                                  "standard"))


latitudes_ssp2 = lat_pop_ssp2[:]

longitudes_ssp2 = lon_pop_ssp2[:]

latitudes_2d_ssp2, longitudes_2d_ssp2 = np.meshgrid(latitudes_ssp2, 
                                          longitudes_ssp2, 
                                          indexing="ij")

latitudes_flat_ssp2 = latitudes_2d_ssp2.flatten()

longitudes_flat_ssp2 = longitudes_2d_ssp2.flatten()

pop_ssp2_df = pd.DataFrame({
"date": np.repeat(times_pop_ssp2, len(latitudes_flat_ssp2)),
"latitude": np.tile(latitudes_flat_ssp2, len(times_pop_ssp2)),
"longitude": np.tile(longitudes_flat_ssp2, len(times_pop_ssp2)),
"total population": total_pop_ssp2[:].flatten()
})

##### Limiting to Europe

pop_ssp2_df.loc[... < pop_ssp2_df["latitude"] < ...]

pop_ssp2_df.loc[... < pop_ssp2_df["longitude"] < ...]

##### Saving Data

csv_filename = "population_ssp2_europe.csv"
pop_ssp2_df.to_csv(csv_filename, index=False)

print(f"Population for SSP2 (all rows) saved to {csv_filename}")



#### SSP5
pop_ssp5 = nc.Dataset(r"C:\Data\Population Density\SSP5\population_ssp5soc_0p5deg_annual_2006-2100.nc4")

print(pop_ssp5.dimensions)

print(pop_ssp5.variables)

total_pop_ssp5 = pop_ssp5.variables["number_of_people"]
print(total_pop_ssp5)

time_pop_ssp5 = pop_ssp5.variables["time"]

lat_pop_ssp5 = pop_ssp5.variables["lat"]

lon_pop_ssp5 = pop_ssp5.variables["lon"]

print(time_pop_ssp5)

new_unit = "days since 2006-01-01 00:00:00"
times_pop_ssp5 = num2date(time_pop_ssp5, units=new_unit, 
                 calendar=getattr(time_pop_ssp5, 
                                  "calendar", 
                                  "standard"))


latitudes_ssp5 = lat_pop_ssp5[:]

longitudes_ssp5 = lon_pop_ssp5[:]

latitudes_2d_ssp5, longitudes_2d_ssp5 = np.meshgrid(latitudes_ssp5, 
                                          longitudes_ssp5, 
                                          indexing="ij")

latitudes_flat_ssp5 = latitudes_2d_ssp5.flatten()

longitudes_flat_ssp5 = longitudes_2d_ssp5.flatten()

pop_ssp5_df = pd.DataFrame({
"date": np.repeat(times_pop_ssp5, len(latitudes_flat_ssp5)),
"latitude": np.tile(latitudes_flat_ssp5, len(times_pop_ssp5)),
"longitude": np.tile(longitudes_flat_ssp5, len(times_pop_ssp5)),
"total population": total_pop_ssp5[:].flatten()
})

##### Limiting to Europe

pop_ssp5_df.loc[... < pop_ssp5_df["latitude"] < ...]

pop_ssp5_df.loc[... < pop_ssp5_df["longitude"] < ...]

##### Saving Data

csv_filename = "population_ssp5:europe.csv"
pop_ssp5_df.to_csv(csv_filename, index=False)

print(f"Population for SSP5 (all rows) saved to {csv_filename}")


### Nighttime light


### Road Impact
road_impact_total = pd.read_csv(r"C:\Data\Road Impact\grip4_total_dens_m_km2.asc")
print(road_impact_total)


#### GDP
##### Historical

##### SSP1

##### SSP2

##### SSP5

#### Area
##### Historical

##### SSP1

##### SSP2

##### SSP5

#### OECD Membership
##### Historical

##### SSP1

##### SSP2

##### SSP5

#### Population Density
"""
See other population denstiy
"""

### LPJ-GUESS

#### Temperature
##### Historical

##### SSP1

##### SSP2

##### SSP5

#### Precipitation
##### Historical

##### SSP1

##### SSP2

##### SSP5


#### Radiation
##### Historical

##### SSP1

##### SSP2

##### SSP5

#### Land Use
##### Historical

##### SSP1

##### SSP2

##### SSP5

#### Soil
soil_hwsd = pd.read_excel(r"C:\Data\LPJ-GUESS\Soil Data\HWSD_DATA.xlsx")
print(soil_hwsd)


# Calculations
## EQI
def eqi(lai, gpp, fvc):
    eqi_result = []
    for i in range(1, len(lai)):
        eqi_result[i] = ((lai[i]+gpp[i]+fvc[i])/3)*100
        return eqi_result 
"""
eqi_hist = eqi(lai_hist, gpp_hist, fvc_hist)

eqi_ssp1 = eqi(lai_ssp1, gpp_ssp1, fvc_ssp1)

eqi_ssp2 = eqi(lai_ssp2, gpp_ssp2, fvc_ssp2)

eqi_ssp5 = eqi(lai_ssp5, gpp_ssp5, fvc_ssp5)
"""

# Model fitting Road Impact
road_model = sklearn.linear_model.RidgeCV()
road_model.fit()

road_results_ssp1 = road_model.predict()
road_results_ssp2 = road_model.predict()
road_results_ssp3 = road_model.predict()


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
