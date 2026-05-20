# Packages
## Array & DataFrame
import numpy as np
import pandas as pd
import geopandas
import duckdb
import re
import scipy
from scipy.interpolate import CubicSpline
import datetime


## Geospatial
from countrygroups import OECD
import reverse_geocoder as rg
import country_converter as coco

## Raster & Image Handeling
import rasterio
import rioxarray as rxr
import xarray as xr
import netCDF4 as nc

## Modelling & Statistics
import sklearn
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
import scipy.stats as stats
from statsmodels.graphics.gofplots import qqplot
import scikit_posthocs as sp

## Visualisation
import matplotlib
import matplotlib.pyplot as plt

## Versioning
import sys
import os
import glob

### Versions
print(sys.version) # 3.12
print(np.__version__) # 2.3
print(pd.__version__) # 2.3
print(scipy.__version__) # 1.17
print(xr.__version__) # 2026.2
print(sklearn.__version__) # 1.8
print(matplotlib.__version__) # 3.9
print(rasterio.__version__) #1.4
print(rxr.__version__) # 0.18
print(coco.__version__) # 1.3

# Data Processing
## Global Variables
### Resolution
"""
The spatial resolution used for the conversion of datasets.
"""
resolution = 0.5

### Offset
"""
The offset used for the snap-to-grid function which places cells in .25 or .75 degree bins?
"""
offset = -0.25

### Europe Boundaries
"""
a tuple of specified boundaries of the area investigated.
"""
europe_boundaries = (-9.75, 35.25, 31.75, 70.25) #lon, lat, lon, lat

### Countries in Europe
"""
a list of the names of all the european countries.
"""
european_countries = [
    "Albania", "Andorra", "Austria", "Belarus", "Belgium", 
    "Bosnia and Herzegovina", "Bulgaria", "Croatia", "Cyprus", 
    "Czech Republic", "Denmark", "Estonia", "Finland", "France", 
    "Germany", "Greece", "Hungary", "Iceland", "Ireland", "Italy", 
    "Kosovo", "Latvia", "Liechtenstein", "Lithuania", "Luxembourg", 
    "Malta", "Moldova", "Monaco", "Montenegro", "Netherlands", 
    "North Macedonia", "Norway", "Poland", "Portugal", "Romania", 
    "Russia", "San Marino", "Serbia", "Slovakia", "Slovenia", 
    "Spain", "Sweden", "Switzerland", "Ukraine", "United Kingdom", 
    "Vatican City", 'Türkiye', 'Kazakhstan', 'Georgia', 'Azerbaijan', 
    'Armenia'
]

### Environmental weights
weight = (0.35, 0.25, 0.18, 0.12, 0.10)

###  plotting
url = r"C:\Users\liamk\Downloads\ne_10m_admin_0_countries.shp"
world = geopandas.read_file(url)

europe = world.clip([europe_boundaries[0], europe_boundaries[2], 
                     europe_boundaries[1], europe_boundaries[3]])

## Functions
scaler=MinMaxScaler()

def get_year(filename, variable):
    """
    

    Parameters
    ----------
    filename : string
        the file path string for which you want to extract the year from.
    variable : string
        the structure of the filenames are different depending on the variable
        and so this is to distinguish between different filenames.

    Returns
    -------
    Year : String
        the year extracted from the filename.

    """
    basename = os.path.splitext(os.path.basename(filename))[0]
    
    if variable == 'ntl':
        parts = basename.split('_')
        idx = 3 if len(parts) > 3 else 2
        return int(parts[idx])
    elif variable == 'gdp' or variable == 'hfp':
        basename_list = re.split(r'(\d+)', basename)
        return int(basename_list[1])
    else:
        raise ValueError(f'unrecognised variable {variable}')


def filter_to_target_coords(df, target_df):
    """
    

    Parameters
    ----------
    df : DataFrame
        A Pandas DataFrame with columns named Lon and Lat.
    target_df : DataFrame
        A Pandas Dataframe containing the Lon and Lat of which you 
        want to limit original DataFrame.

    Returns
    -------
    DataFrame
        The original df merged with the coordinates of the target_df.

    """
    target_coords = target_df[['Lon', 'Lat']].drop_duplicates()
    return df.merge(target_coords, on=['Lon', 'Lat'], how='inner')

def snap_to_grid(df, resolution, offset=-0.25):
    """
    

    Parameters
    ----------
    df : DataFrame
        dataframe that contains 'Lon' and 'Lat' columns.
    resolution : float
        the degree of resolution to transform data into.
    offset : float or int
        a value to offset the coordinates by default to the centre,
        .25 and .75.

    Returns
    -------
    df : DataFrame
        the inputed dataframe with transformed 'Lon' and 'Lat'.

    """
    df['Lon'] = (np.floor((df['Lon'] - offset) / resolution + 0.5) * 
                 resolution + offset)
    df['Lat'] = (np.floor((df['Lat'] - offset) / resolution + 0.5) * 
                 resolution + offset)
    return df


def cubic_spline_interpolation(series, data_max):
    """
    

    Parameters
    ----------
    series : series
        the column of a dataframe with the variable that is to be interpolated 
        as a series.
    data_max : float or int
        the maximum value of the series.

    Returns
    -------
    series
        the column as a series with splined variables.

    """
    
    valid = series.dropna()
    cs = CubicSpline(valid.index.get_level_values('Year'),
                     valid.values,
                     bc_type='not-a-knot')

    all_years = series.index.get_level_values('Year')
    interpolated = np.clip(cs(all_years), a_min=0, a_max=data_max)
    return pd.Series(interpolated, index=series.index, dtype='float64')

def annual_interpolation(df, variable, variable_scaled):
    """
    

    Parameters
    ----------
    df : DataFrame
        A Pandas DataFrame containing Lon, Lat, Years and variable
        for which you want to interpolate e.g. 5 year interval to annual.
    variable : String
        A string with the name of the variable that is to be interpolated.
    variable_scaled : String
        A string with the name of the scaled version of 
        the variable that is to be interpolated.

    Returns
    -------
    DataFrame
        A Pandas DataFrame with interpolated years.

    """
    df = df.set_index(['Lon', 'Lat', 'Year'])
    
    years = range(df.index.get_level_values('Year').min(),
                  df.index.get_level_values('Year').max() + 1)
    
    lonlat_pairs = df.index.droplevel('Year').unique()
    
    full_index = pd.MultiIndex.from_arrays(
        [
            np.repeat(lonlat_pairs.get_level_values('Lon'), len(years)),
            np.repeat(lonlat_pairs.get_level_values('Lat'), len(years)),
            np.tile(list(years), len(lonlat_pairs))
        ],
        names=['Lon', 'Lat', 'Year']
    )
    
    def apply_interpolation(g):
        """
        

        Parameters
        ----------
        g : pandas DataFrame
            a subset of df containing all years for a single location.

        Returns
        -------
        Dataframe
            a dataframe containing the interpolated variable, scaled variable,
            variances and variances of the scaled variable.

        """
        interpolated = cubic_spline_interpolation(g[variable], 
                                                  g[variable].max())
        
        interpolated_scaled = cubic_spline_interpolation(g[variable_scaled], 
                                                         g[variable_scaled].max())
        return pd.DataFrame({variable: interpolated, 
                             variable_scaled:interpolated_scaled, 
                             'var': g['var'].fillna(0).values, 
                             'var_scaled':g['var_scaled'].fillna(0).values})

    df_annual = (df
             .reindex(full_index)
             .groupby(level=['Lon', 'Lat'])
             .apply(apply_interpolation)
             .droplevel([0, 1])
             .reset_index()
             )

    return pd.DataFrame(df_annual).sort_values(by=['Year', 
                                                   'Lon', 
                                                   'Lat'], 
                                               ignore_index=True)


def resample_tif_to_csv(directories, output_directory, boundaries,
                            time_period, resolution, variable, target_df,
                            training_year=None):
    """
    

    Parameters
    ----------
    directories : List of strings
        A list of strings for the folder(s) containing the
        data in TIFF format.
    output_directory : String
        The folder where the output should be stored.
    boundaries : Tuple
        A tuple containing Lon_min, Lat_min, Lon_max and 
        Lat_max of the study region.
    time_period : String
        Name of the time_period, primarily for naming the output file, 
        e.g. "historical" or "ssp1".
    resolution : Float
        What resolution the snap_to_grid should snap on.
    variable : String
        The variable of the data for which is to be resampled.
    target_df : DataFrame
        The Pandas DataFrame that the Lon, Lat of the data should be limited to..
    training_year : Int, optional
        For the variables that scale for the regression model there is an 
        option to have it scale using a training year. 
        For the variables where this is not necessary the default is None.

    Returns
    -------
    A output file saved on computer in the output folder.

    """
    
    os.makedirs(output_directory, exist_ok=True)
    output_file = os.path.join(output_directory, f'df_{time_period}_05deg.csv')
    
    log_variables={'ntl', 'gdp'}
    
    if isinstance(directories, str):
        directories = [directories]

    files = []

    for directory in directories:
        for file in glob.glob(os.path.join(directory, "*.tif")):
            year = get_year(file, variable)
            
            da = rxr.open_rasterio(file, lock=False)[0]
            
            if da.rio.crs is None:
                da = da.rio.write_crs("EPSG:4326")
            elif da.rio.crs.to_epsg() != 4326:
                da = da.rio.reproject("EPSG:4326")
            
            # clip to Europe
            da = da.rio.clip_box(minx=boundaries[0], miny=boundaries[1],
                                 maxx=boundaries[2], maxy=boundaries[3])
            
            vals = da.values.ravel()
            lons, lats = np.meshgrid(da.x.values, da.y.values)
            mask = ~np.isnan(vals)
            df = pd.DataFrame({
                'Lon': lons.ravel()[mask],
                'Lat': lats.ravel()[mask],
                variable: vals[mask]
                })
            
            df[variable] = df[variable].replace([np.inf, -np.inf], 0).clip(lower=0)
            
            df['Year'] = year
            
            df[variable] = df[variable].astype('float32')
            
            # Apply snap_to_grid and filter_to_target_coords
            df = snap_to_grid(df, resolution)
            df = filter_to_target_coords(df, target_df)

            files.append(df)
        
            del da, vals, lons, lats, mask
    
    if not files:
        raise FileNotFoundError(f'No .tif files found in {directories}')
    
    df_all = pd.concat(files, ignore_index=True)
    
    del files
    
    if variable in log_variables:
        df_all[variable] = np.log1p(df_all[variable])
    
    if training_year is not None:
        train_mask = df_all['Year'] == training_year
        train_data = df_all.loc[train_mask, variable]
    else:
        train_data = df_all[variable]
        
    mean_val = train_data.mean()
    std_val = train_data.std()
    
    scaled_variable = f'{variable}_scaled'
    df_all[scaled_variable] = (df_all[variable]-mean_val)/std_val
    
    # AVG for densities, SUM for count.
    agg_function = 'SUM' if variable == 'gdp' else 'AVG'
    
    duckdb.sql(f"""
               COPY (
                   SELECT Year,
                       ROUND(Lon::FLOAT, 2) AS Lon,
                       ROUND(Lat::FLOAT, 2) AS Lat,
                       {agg_function}({variable}::FLOAT) AS {variable},
                       {agg_function}({scaled_variable}::FLOAT) AS {scaled_variable},
                       VAR_POP({variable}::FLOAT) AS var,
                       VAR_POP({scaled_variable}::FLOAT) AS var_scaled
                       FROM df_all
                       GROUP BY Year, Lon, Lat
                ) TO '{output_file}' (FORMAT CSV,
                                             OVERWRITE_OR_IGNORE TRUE)
                                             """)
    
    print(f'df_{time_period}_05deg saved to {output_file}')

def weighted_road_densities(df, weight, column_name):
    """
    

    Parameters
    ----------
    df : pandas DataFrame
        the dataframe that contains road type columns.
    weight : tuple
        a tuple with the environmental weights, number of values
        equal to the number of road types.
    column_name : string
        the name of the road type column that the weight is applied to.

    Returns
    -------
    the weigted road type.

    """
    if column_name in df.columns:
        idx = int(re.search(r'\d+', column_name).group())-1 
        return(df[column_name]*weight[idx])
    raise ValueError(f"column '{column_name}' not found in DataFrame")


## Data Reading
### EQI
#### LAI
##### SSP1
lai_ssp1 = pd.read_csv(r"C:\Data\Habitat Quality\SPSS1\lai_sts_2015-2100_ssp126.txt", 
                       delimiter='\t')

lai_ssp1_df = pd.DataFrame(lai_ssp1)
lai_ssp1_df = lai_ssp1_df.rename(columns={'    Lon': 'Lon', 
                                          '    Lat': 'Lat', 
                                          '   Total':'Total'})

lai_ssp1_df = lai_ssp1_df.sort_values(by=['Year', 'Lon', 'Lat'], 
                                      ignore_index=True)

# Restrain to time period
lai_ssp1_df = lai_ssp1_df[lai_ssp1_df['Year']<=2050]
print(lai_ssp1_df)

##### SSP3
lai_ssp3 = pd.read_csv(r"C:\Data\Habitat Quality\SPSS3\lai_sts_2015-2100_ssp370.txt", 
                       delimiter='\t')
lai_ssp3_df = pd.DataFrame(lai_ssp3)
lai_ssp3_df = lai_ssp3_df.rename(columns={'    Lon': 'Lon', 
                                          '    Lat': 'Lat', 
                                          '   Total':'Total'})

lai_ssp3_df = lai_ssp3_df.sort_values(by=['Year', 'Lon', 'Lat'], 
                                      ignore_index=True)

# Restrain to time period
lai_ssp3_df = lai_ssp3_df[lai_ssp3_df['Year']<=2050]
print(lai_ssp3_df)

##### SSP5
lai_ssp5 = pd.read_csv(r"C:\Data\Habitat Quality\SPSS5\lai_sts_2015-2100_ssp585.txt", 
                       delimiter = '\t')

lai_ssp5_df = pd.DataFrame(lai_ssp5)

lai_ssp5_df = lai_ssp5_df.rename(columns={'    Lon': 'Lon', 
                                          '    Lat': 'Lat', 
                                          '   Total':'Total'})

lai_ssp5_df = lai_ssp5_df.sort_values(by=['Year', 'Lon', 'Lat'], 
                                      ignore_index=True)

# Restrain to time period
lai_ssp5_df = lai_ssp5_df[lai_ssp5_df['Year']<=2050]
print(lai_ssp5_df)

#### GPP
##### SSP1
gpp_ssp1 = pd.read_csv(r"C:\Data\Habitat Quality\SPSS1\agpp_sts_2015-2100_ssp126.txt", 
                       delimiter = '\t')

gpp_ssp1_df = pd.DataFrame(gpp_ssp1)
gpp_ssp1_df = gpp_ssp1_df.rename(columns={'    Lon': 'Lon', 
                                          '    Lat': 'Lat', 
                                          '   Total':'Total'})

gpp_ssp1_df = gpp_ssp1_df.sort_values(by=['Year', 'Lon', 'Lat'], 
                                      ignore_index=True)

# Restrain to time period
gpp_ssp1_df = gpp_ssp1_df[gpp_ssp1_df['Year']<=2050]
print(gpp_ssp1_df)

###### Plotting
gpp_ssp1_df_2050 = gpp_ssp1_df[gpp_ssp1_df['Year']==2050]
gpp_ssp1_df_2050 = gpp_ssp1_df_2050.drop(columns=['Year', 'Forest_sum'])

gpp_ssp1_df_2050 = geopandas.GeoDataFrame(
    gpp_ssp1_df_2050, 
    geometry=geopandas.points_from_xy(gpp_ssp1_df_2050.Lon, 
                                      gpp_ssp1_df_2050.Lat), 
    crs="EPSG:4326"
)

fig, ax = plt.subplots(figsize=(12,10))
europe.plot(color='lightgrey', edgecolor='black', ax=ax)

gpp_ssp1_df_2050.plot(
    ax=ax,
    column='Total',
    cmap = 'gist_earth',
    markersize = 150,
    marker='s',
    legend=True,
    legend_kwds = {
        'label': 'GPP  (kgC m-2 yr-1)',
        'orientation':'vertical'}
    )

ax.set_xlim(europe_boundaries[0], europe_boundaries[2])
ax.set_ylim(europe_boundaries[1], europe_boundaries[3])
ax.set_title("GPP SSP1 (2050)", fontsize=14)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

plt.tight_layout()
plt.show()

##### SSP3
gpp_ssp3 = pd.read_csv(r"C:\Data\Habitat Quality\SPSS3\agpp_sts_2015-2100_ssp370.txt", 
                       delimiter = '\t')

gpp_ssp3_df = pd.DataFrame(gpp_ssp3)

gpp_ssp3_df = gpp_ssp3_df.rename(columns={'    Lon': 'Lon', 
                                          '    Lat': 'Lat', 
                                          '   Total':'Total'})

gpp_ssp3_df = gpp_ssp3_df.sort_values(by=['Year', 'Lon', 'Lat'], 
                                      ignore_index=True)

gpp_ssp3_df = gpp_ssp3_df[gpp_ssp3_df['Year']<=2050]
print(gpp_ssp3_df)

gpp_ssp3_df_2050 = gpp_ssp3_df[gpp_ssp3_df['Year']==2050]
gpp_ssp3_df_2050 = gpp_ssp3_df_2050.drop(columns=['Year', 'Forest_sum'])

gpp_ssp3_df_2050 = geopandas.GeoDataFrame(
    gpp_ssp3_df_2050, 
    geometry=geopandas.points_from_xy(gpp_ssp3_df_2050.Lon, 
                                      gpp_ssp3_df_2050.Lat), 
    crs="EPSG:4326"
)

fig, ax = plt.subplots(figsize=(12,10))
europe.plot(color='lightgrey', edgecolor='black', ax=ax)

gpp_ssp3_df_2050.plot(
    ax=ax,
    column='Total',
    cmap = 'gist_earth',
    markersize = 150,
    marker='s',
    legend=True,
    legend_kwds = {
        'label': 'GPP (kgC m-2 yr-1)',
        'orientation':'vertical'}
    )

ax.set_xlim(europe_boundaries[0], europe_boundaries[2])
ax.set_ylim(europe_boundaries[1], europe_boundaries[3])
ax.set_title("GPP SSP3 (2050)", fontsize=14)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

plt.tight_layout()
plt.show()


##### SSP5
gpp_ssp5 = pd.read_csv(r"C:\Data\Habitat Quality\SPSS5\agpp_sts_2015-2100_ssp585.txt", delimiter = '\t')
gpp_ssp5_df = pd.DataFrame(gpp_ssp5)
gpp_ssp5_df = gpp_ssp5_df.rename(columns={'    Lon': 'Lon', '    Lat': 'Lat', '   Total':'Total'})
gpp_ssp5_df = gpp_ssp5_df.sort_values(by=['Year', 'Lon', 'Lat'], 
                                      ignore_index=True)

# Restrain to time period
gpp_ssp5_df = gpp_ssp5_df[gpp_ssp5_df['Year']<=2050]

# Plot 2050
gpp_ssp5_df_2050 = gpp_ssp5_df[gpp_ssp5_df['Year']==2050]
gpp_ssp5_df_2050 = gpp_ssp5_df_2050.drop(columns=['Year', 'Forest_sum'])

gpp_ssp5_df_2050 = geopandas.GeoDataFrame(
    gpp_ssp5_df_2050, 
    geometry=geopandas.points_from_xy(gpp_ssp5_df_2050.Lon, 
                                      gpp_ssp5_df_2050.Lat), 
    crs="EPSG:4326"
)

fig, ax = plt.subplots(figsize=(12,10))
europe.plot(color='lightgrey', edgecolor='black', ax=ax)

gpp_ssp5_df_2050.plot(
    ax=ax,
    column='Total',
    cmap = 'gist_earth',
    markersize = 150,
    marker='s',
    legend=True,
    legend_kwds = {
        'label': 'GPP (kgC m-2 yr-1)',
        'orientation':'vertical'}
    )

ax.set_xlim(europe_boundaries[0], europe_boundaries[2])
ax.set_ylim(europe_boundaries[1], europe_boundaries[3])
ax.set_title("GPP SSP5 (2050)", fontsize=14)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

plt.tight_layout()
plt.show()

#### FPC
##### SSP1
fpc_ssp1 = pd.read_csv(r"C:\Data\Habitat Quality\SPSS1\frac_2015-2100_ssp126.txt", 
                       delimiter='\t')
fpc_ssp1_df = pd.DataFrame(fpc_ssp1)
fpc_ssp1_df = fpc_ssp1_df.rename(columns={'    Lon': 'Lon', '    Lat': 'Lat'})
fpc_ssp1_df = fpc_ssp1_df.sort_values(by=['Year', 'Lon', 'Lat'], 
                                      ignore_index=True)
fpc_ssp1_df = fpc_ssp1_df.drop(columns=['Total', 
                                        'Barren_sum', 'Fodder_crops', 
                                        'Food_crops', 'Bioenergy_crops'])

print(fpc_ssp1_df)
total_veg_cover_ssp1 = []
for i in range(0, len(fpc_ssp1_df)):
    total_veg_ssp1 = (fpc_ssp1_df['Crop_sum'][i]+
                      fpc_ssp1_df['Pasture_sum'][i]+
                      fpc_ssp1_df['Forest_sum'][i]+
                      fpc_ssp1_df['Natural_sum'][i])
    
    total_veg_cover_ssp1.append(total_veg_ssp1)

total_veg_cover_ssp1 = pd.DataFrame(total_veg_cover_ssp1)
print(total_veg_cover_ssp1)

fpc_ssp1_df['total_veg_cover'] = total_veg_cover_ssp1[0].values

# Restrain to time period
fpc_ssp1_df = fpc_ssp1_df[fpc_ssp1_df['Year'] <= 2050]
fpc_ssp1_df = fpc_ssp1_df[fpc_ssp1_df['Year'] >= 2015]


fpc_ssp1_df = fpc_ssp1_df.sort_values(by=['Year','Lon', 'Lat'])

fpc_ssp1_df_2050 = fpc_ssp1_df[fpc_ssp1_df['Year']==2050]
fpc_ssp1_df_2050 = fpc_ssp1_df_2050.drop(columns=['Year', 
                                                  'Forest_sum', 
                                                  'Natural_sum', 
                                                  'Pasture_sum', 
                                                  'Crop_sum'])


fpc_ssp1_df_2050 = geopandas.GeoDataFrame(
    fpc_ssp1_df_2050, 
    geometry=geopandas.points_from_xy(fpc_ssp1_df_2050.Lon, 
                                      fpc_ssp1_df_2050.Lat), 
    crs="EPSG:4326"
)


fig, ax = plt.subplots(figsize=(12,10))
europe.plot(color='lightgrey', edgecolor='black', ax=ax)

fpc_ssp1_df_2050.plot(
    ax=ax,
    column='total_veg_cover',
    cmap = 'gist_earth',
    markersize = 150,
    marker='s',
    legend=True,
    legend_kwds = {
        'label': 'FVC (%)',
        'orientation':'vertical'}
    )

ax.set_xlim(europe_boundaries[0], europe_boundaries[2])
ax.set_ylim(europe_boundaries[1], europe_boundaries[3])
ax.set_title("FVC SSP1 (2050)", fontsize=14)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

plt.tight_layout()
plt.show()

##### SSP3
fpc_ssp3 = pd.read_csv(r"C:\Data\Habitat Quality\SPSS3\frac_2015-2100_ssp370.txt", 
                       delimiter='\t')

fpc_ssp3_df = pd.DataFrame(fpc_ssp3)
fpc_ssp3_df = fpc_ssp3_df.rename(columns={'    Lon': 'Lon', '    Lat': 'Lat'})

fpc_ssp3_df = fpc_ssp3_df.sort_values(by=['Year', 'Lon', 'Lat'], 
                                      ignore_index=True)

fpc_ssp3_df = fpc_ssp3_df.drop(columns=['Total', 
                                        'Barren_sum', 'Fodder_crops', 
                                        'Food_crops', 'Bioenergy_crops'])
fpc_ssp3_df = fpc_ssp3_df.dropna()
fpc_ssp3_df = fpc_ssp3_df.reset_index(drop=True)

total_veg_cover_ssp3 = []
for i in range(0, len(fpc_ssp3_df)):
    total_veg_ssp5 = (fpc_ssp3_df['Crop_sum'][i]+
                      fpc_ssp3_df['Pasture_sum'][i]+
                      fpc_ssp3_df['Forest_sum'][i]+
                      fpc_ssp3_df['Natural_sum'][i])
    total_veg_cover_ssp3.append(total_veg_ssp5)

total_veg_cover_ssp3 = pd.DataFrame(total_veg_cover_ssp3)

fpc_ssp3_df['total_veg_cover'] = total_veg_cover_ssp3[0].values

# Restrain to time period
fpc_ssp3_df = fpc_ssp3_df[fpc_ssp3_df['Year'] <= 2050]
fpc_ssp3_df = fpc_ssp3_df[fpc_ssp3_df['Year'] >= 2015]

fpc_ssp3_df_2050 = fpc_ssp3_df[fpc_ssp3_df['Year']==2050]
fpc_ssp3_df_2050 = fpc_ssp3_df_2050.drop(columns=['Year', 'Forest_sum', 
                                                  'Natural_sum', 
                                                  'Pasture_sum', 'Crop_sum'])

fpc_ssp3_df_2050 = geopandas.GeoDataFrame(
    fpc_ssp3_df_2050, 
    geometry=geopandas.points_from_xy(fpc_ssp3_df_2050.Lon, 
                                      fpc_ssp3_df_2050.Lat), 
    crs="EPSG:4326"
)
fpc_ssp3_df

fig, ax = plt.subplots(figsize=(12,10))
europe.plot(color='lightgrey', edgecolor='black', ax=ax)

fpc_ssp3_df_2050.plot(
    ax=ax,
    column='total_veg_cover',
    cmap = 'gist_earth',
    markersize = 150,
    marker='s',
    legend=True,
    legend_kwds = {
        'label': 'FVC (%)',
        'orientation':'vertical'}
    )

ax.set_xlim(europe_boundaries[0], europe_boundaries[2])
ax.set_ylim(europe_boundaries[1], europe_boundaries[3])
ax.set_title("FVC SSP3 (2050)", fontsize=14)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

plt.tight_layout()
plt.show()

##### SSP5
fpc_ssp5 = pd.read_csv(r"C:\Data\Habitat Quality\SPSS5\frac_2015-2100_ssp585.txt", 
                       delimiter='\t')

fpc_ssp5_df = pd.DataFrame(fpc_ssp5)

fpc_ssp5_df = fpc_ssp5_df.rename(columns={'    Lon': 'Lon', '    Lat': 'Lat'})

fpc_ssp5_df = fpc_ssp5_df.sort_values(by=['Year', 'Lon', 'Lat'], 
                                      ignore_index=True)

fpc_ssp5_df = fpc_ssp5_df.drop(columns=['Total', 
                                        'Barren_sum', 'Fodder_crops', 
                                        'Food_crops', 'Bioenergy_crops'])
print(fpc_ssp5_df)

total_veg_cover_ssp5 = []
for i in range(0, len(fpc_ssp5_df)):
    total_veg_ssp5 = (fpc_ssp5_df['Crop_sum'][i]+
                      fpc_ssp5_df['Pasture_sum'][i]+
                      fpc_ssp5_df['Forest_sum'][i]+
                      fpc_ssp5_df['Natural_sum'][i])
    
    total_veg_cover_ssp5.append(total_veg_ssp5)

total_veg_cover_ssp5 = pd.DataFrame(total_veg_cover_ssp5)
print(total_veg_cover_ssp5)

fpc_ssp5_df['total_veg_cover'] = total_veg_cover_ssp5[0].values

fpc_ssp5_df_2050 = fpc_ssp5_df[fpc_ssp5_df['Year']==2050]

fpc_ssp5_df_2050 = fpc_ssp5_df_2050.drop(columns=['Year', 
                                                  'Forest_sum', 
                                                  'Natural_sum', 
                                                  'Pasture_sum', 
                                                  'Crop_sum'])

fpc_ssp5_df_2050 = geopandas.GeoDataFrame(
    fpc_ssp5_df_2050, 
    geometry=geopandas.points_from_xy(fpc_ssp5_df_2050.Lon, 
                                      fpc_ssp5_df_2050.Lat), 
    crs="EPSG:4326"
)


fig, ax = plt.subplots(figsize=(12,10))
europe.plot(color='lightgrey', edgecolor='black', ax=ax)

fpc_ssp5_df_2050.plot(
    ax=ax,
    column='total_veg_cover',
    cmap = 'gist_earth',
    markersize = 150,
    marker='s',
    legend=True,
    legend_kwds = {
        'label': 'FVC (%)',
        'orientation':'vertical'}
    )

ax.set_xlim(europe_boundaries[0], europe_boundaries[2])
ax.set_ylim(europe_boundaries[1], europe_boundaries[3])
ax.set_title("FVC SSP5 (2050)", fontsize=14)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

plt.tight_layout()
plt.show()

### Population Density
pop_hist = nc.Dataset(r"C:\Data\Population Density\Control\population_histsoc_0p5deg_annual_1861-2005.nc4")

print(pop_hist.dimensions)

print(pop_hist.variables)

time_pop_hist = pop_hist.variables["time"]

lat_pop_hist = pop_hist.variables["lat"]

lon_pop_hist = pop_hist.variables["lon"]

latitudes = lat_pop_hist[:]

longitudes = lon_pop_hist[:]
longitudes

latitudes_2d, longitudes_2d = np.meshgrid(latitudes, 
                                          longitudes, 
                                          indexing="ij")

latitudes_flat = latitudes_2d.flatten()

longitudes_flat = longitudes_2d.flatten()

end_hist = datetime.date(2006, 1,1)
start_hist = datetime.datetime(1860, 1, 1)
times_pop_hist = [end_hist.replace(year=i) 
     for i in range(start_hist.year, end_hist.year)
     if end_hist.replace(year=i) > start_hist.date()]

total_pop_hist = pop_hist.variables["number_of_people"]
total_pop_hist_density = total_pop_hist[:].flatten()/((111.32 * 0.5) * 
                                                      (111.32 * 0.5 * 
                                                       np.cos(np.radians(np.tile(latitudes_flat, 
                                                                                 len(times_pop_hist))))))

pop_hist_df = pd.DataFrame({
"Year": np.repeat(times_pop_hist, len(latitudes_flat)),
"Lat": np.tile(latitudes_flat, len(times_pop_hist)),
"Lon": np.tile(longitudes_flat, len(times_pop_hist)),
"population": total_pop_hist[:].flatten(),
"population_density": total_pop_hist_density})

##### Limiting to Europe
pop_hist_df = pop_hist_df.loc[(pop_hist_df["Lat"] <= europe_boundaries[3]) &
                              (pop_hist_df["Lat"] >= europe_boundaries[1])&
                              (pop_hist_df["Lon"] <= europe_boundaries[2])&
                              (pop_hist_df["Lon"] >= europe_boundaries[0])]

pop_hist_df['Year'] = pd.to_datetime(pop_hist_df['Year']).dt.year

pop_hist_df['population_density'] = np.log1p(pop_hist_df['population_density'])

mean_population_density = pop_hist_df[pop_hist_df['Year']==2005]['population_density'].mean()
std_population_density = np.sqrt(pop_hist_df[pop_hist_df['Year']==2005]['population_density'].var())
pop_hist_df['population_density_scaled'] = (pop_hist_df['population_density']-mean_population_density)/std_population_density

mask = pop_hist_df.set_index(['Lon', 'Lat']).index.isin(
    fpc_ssp3_df.set_index([ 'Lon', 'Lat']).index
)

pop_hist_df = pop_hist_df[mask].reset_index(drop=True)

pop_hist_2005_df = pop_hist_df[pop_hist_df['Year'] == 2005]

#### Plotting
pop_hist_2005_df['population_density'] = pd.to_numeric(pop_hist_2005_df['population_density'])

pop_hist_2005_df = geopandas.GeoDataFrame(
    pop_hist_2005_df, 
    geometry=geopandas.points_from_xy(pop_hist_2005_df.Lon, 
                                      pop_hist_2005_df.Lat), 
    crs="EPSG:4326"
)

fig, ax = plt.subplots(figsize=(12,10))
europe.plot(color='lightgrey', edgecolor='black', ax=ax)

pop_hist_2005_df.plot(
    ax=ax,
    column='population_density',
    cmap = 'gist_earth',
    markersize = 150,
    marker='s',
    legend=True,
    legend_kwds = {
        'label': 'Population Density (N/km2)',
        'orientation':'vertical'}
    )

ax.set_xlim(europe_boundaries[0], europe_boundaries[2])
ax.set_ylim(europe_boundaries[1], europe_boundaries[3])
ax.set_title("Population Density (2005)", fontsize=14)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

plt.tight_layout()
plt.show()

##### Saving Data
csv_filename = "total_population_historical_europe.csv"
pop_hist_df.to_csv(csv_filename, index=False)

print(f"Historical Population (all rows) saved to {csv_filename}")

### Scenario
end_scenario = datetime.date(2101, 1,1)
start_scenario = datetime.datetime(2005, 1, 1)
times_pop_scenario = [end_scenario.replace(year=i) 
     for i in range(start_scenario.year, end_scenario.year)
     if end_scenario.replace(year=i) > start_scenario.date()]

#### SSP1
pop_ssp1 = nc.Dataset(r"C:\Data\Population Density\SSP1\population_ssp1soc_0p5deg_annual_2006-2100.nc4")

time_pop_ssp1 = pop_ssp1.variables["time"]

lat_pop_ssp1 = pop_ssp1.variables["lat"]

lon_pop_ssp1 = pop_ssp1.variables["lon"]

latitudes_ssp1 = lat_pop_ssp1[:]

longitudes_ssp1 = lon_pop_ssp1[:]

latitudes_2d_ssp1, longitudes_2d_ssp1 = np.meshgrid(latitudes_ssp1, 
                                          longitudes_ssp1, 
                                          indexing="ij")

latitudes_flat_ssp1 = latitudes_2d_ssp1.flatten()

longitudes_flat_ssp1 = longitudes_2d_ssp1.flatten()


total_pop_ssp1 = pop_ssp1.variables["number_of_people"]
total_pop_ssp1_density = total_pop_ssp1[:].flatten()/((111.32 * 0.5) * (111.32 *
                                                                        0.5 * 
                                                                        np.cos(np.radians(np.tile(latitudes_flat_ssp1, 
                                                                                                  len(time_pop_ssp1))))))

np.tile(np.tile(latitudes_flat_ssp1, len(times_pop_scenario)), len(times_pop_scenario))

pop_ssp1_df = pd.DataFrame({
"Year": np.repeat(times_pop_scenario, len(latitudes_flat_ssp1)),
"Lat": np.tile(latitudes_flat_ssp1, len(times_pop_scenario)),
"Lon": np.tile(longitudes_flat_ssp1, len(times_pop_scenario)),
"population": total_pop_ssp1[:].flatten(),
"population_density": total_pop_ssp1_density
})

##### Limiting to Europe
pop_ssp1_df = pop_ssp1_df.loc[(pop_ssp1_df["Lat"] <= europe_boundaries[3])&
                              (pop_ssp1_df["Lat"] >= europe_boundaries[1])&
                              (pop_ssp1_df["Lon"] <= europe_boundaries[2])&
                              (pop_ssp1_df["Lon"] >= europe_boundaries[0])]

pop_ssp1_df['population_density'] = np.log1p(pop_ssp1_df['population_density'])
pop_ssp1_df['population_density_scaled'] = (pop_ssp1_df['population_density']-mean_population_density)/std_population_density

##### Merging with historical data
pop_ssp1_hist_df = pd.concat([pop_hist_df, pop_ssp1_df], ignore_index=True)

pop_ssp1_hist_df['Year'] = pd.to_datetime(pop_ssp1_hist_df['Year']).dt.year

pop_ssp1_hist_df['population_density'] = pd.to_numeric(pop_ssp1_hist_df['population_density'])
pop_ssp1_hist_df
pop_ssp1_hist_df = pop_ssp1_hist_df[(pop_ssp1_hist_df['Year']<= 2050) & 
                                    (pop_ssp1_hist_df['Year'] >= 2015)].sort_values(by=['Year',
                                                                                        'Lon',
                                                                                        'Lat']).reset_index(drop=True)

pop_ssp1_hist_df = pop_ssp1_hist_df.rename(columns={'date':'Year'})
pop_ssp1_hist_df
mask = pop_ssp1_hist_df.set_index(['Year', 'Lon', 'Lat']).index.isin(
    fpc_ssp3_df.set_index(['Year', 'Lon', 'Lat']).index
)

pop_ssp1_hist_df = pop_ssp1_hist_df[mask].reset_index(drop=True)

#### Plotting
pop_ssp1_2050_df = pop_ssp1_hist_df[pop_ssp1_hist_df['Year']==2050]

pop_ssp1_2050_df = geopandas.GeoDataFrame(
    pop_ssp1_2050_df, 
    geometry=geopandas.points_from_xy(pop_ssp1_2050_df.Lon, 
                                      pop_ssp1_2050_df.Lat), 
    crs="EPSG:4326"
)

fig, ax = plt.subplots(figsize=(12,10))
europe.plot(color='lightgrey', edgecolor='black', ax=ax)

pop_ssp1_2050_df.plot(
    ax=ax,
    column='population_density',
    cmap = 'gist_earth',
    markersize = 150,
    marker='s',
    legend=True,
    legend_kwds = {
        'label': 'Population Density (N/km2)',
        'orientation':'vertical'}
    )

ax.set_xlim(europe_boundaries[0], europe_boundaries[2])
ax.set_ylim(europe_boundaries[1], europe_boundaries[3])
ax.set_title("Population Density SSP1 (2050)", fontsize=14)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

plt.tight_layout()
plt.show()

##### Saving Data
csv_filename = "total_population_ssp1_europe.csv"
pop_ssp1_hist_df.to_csv(csv_filename, index=False)

print(f"Population for SSP1 (all rows) saved to {csv_filename}")

#### SSP3
pop_ssp3 = nc.Dataset(r"C:\Data\Population Density\SSP3\population_ssp3soc_0p5deg_annual_2006-2100.nc4")

time_pop_ssp3 = pop_ssp3.variables["time"]

lat_pop_ssp3 = pop_ssp3.variables["lat"]

lon_pop_ssp3 = pop_ssp3.variables["lon"]

latitudes_ssp3 = lat_pop_ssp3[:]

longitudes_ssp3 = lon_pop_ssp3[:]

latitudes_2d_ssp3, longitudes_2d_ssp3 = np.meshgrid(latitudes_ssp3, 
                                          longitudes_ssp3, 
                                          indexing="ij")

latitudes_flat_ssp3 = latitudes_2d_ssp3.flatten()

longitudes_flat_ssp3 = longitudes_2d_ssp3.flatten()

total_pop_ssp3 = pop_ssp3.variables["number_of_people"]
total_pop_ssp3_density = total_pop_ssp3[:].flatten()/((111.32 * 0.5) * (111.32 * 0.5 * np.cos(np.radians(np.tile(latitudes_flat_ssp3, len(time_pop_ssp3))))))

pop_ssp3_df = pd.DataFrame({
"Year": np.repeat(times_pop_scenario, len(latitudes_flat_ssp3)),
"Lat": np.tile(latitudes_flat_ssp3, len(times_pop_scenario)),
"Lon": np.tile(longitudes_flat_ssp3, len(times_pop_scenario)),
"population": total_pop_ssp3[:].flatten(),
"population_density": total_pop_ssp3_density
})

##### Limiting to Europe
pop_ssp3_df = pop_ssp3_df.loc[(pop_ssp3_df["Lat"] <= europe_boundaries[3]) &
                              (pop_ssp3_df["Lat"] >= europe_boundaries[1])&
                              (pop_ssp3_df["Lon"] <= europe_boundaries[2])&
                              (pop_ssp3_df["Lon"] >= europe_boundaries[0])]



pop_ssp3_df['population_density'] = np.log1p(pop_ssp3_df['population_density'])

pop_ssp3_df['population_density_scaled'] = (pop_ssp3_df['population_density']-mean_population_density)/std_population_density

##### Merging with historical data
pop_ssp3_hist_df = pd.concat([pop_hist_df, pop_ssp3_df], ignore_index=True)

pop_ssp3_hist_df['Year'] = pd.to_datetime(pop_ssp3_hist_df['Year']).dt.year

pop_ssp3_hist_df['population_density'] = pd.to_numeric(pop_ssp3_hist_df['population_density'])

pop_ssp3_hist_df = pop_ssp3_hist_df[(pop_ssp3_hist_df['Year']<= 2050) & 
                                    (pop_ssp3_hist_df['Year'] >= 2015)].sort_values(by=['Year',
                                                                                        'Lon',
                                                                                        'Lat']).reset_index(drop=True)


mask = pop_ssp3_hist_df.set_index(['Year', 'Lon', 'Lat']).index.isin(
    fpc_ssp3_df.set_index(['Year', 'Lon', 'Lat']).index
)

pop_ssp3_hist_df = pop_ssp3_hist_df[mask].reset_index(drop=True)

#### Plotting
pop_ssp3_2050_df = pop_ssp3_hist_df[pop_ssp3_hist_df['Year']== 2050]
pop_ssp3_2050_df['population_density'] = pd.to_numeric(pop_ssp3_2050_df['population_density'])

pop_ssp3_2050_df = geopandas.GeoDataFrame(
    pop_ssp3_2050_df, 
    geometry=geopandas.points_from_xy(pop_ssp3_2050_df.Lon, 
                                      pop_ssp3_2050_df.Lat), 
    crs="EPSG:4326"
)

fig, ax = plt.subplots(figsize=(12,10))
europe.plot(color='lightgrey', edgecolor='black', ax=ax)

pop_ssp3_2050_df.plot(
    ax=ax,
    column='population_density',
    cmap = 'gist_earth',
    markersize = 150,
    marker='s',
    legend=True,
    legend_kwds = {
        'label': 'Population Density (N/km2)',
        'orientation':'vertical'}
    )

ax.set_xlim(europe_boundaries[0], europe_boundaries[2])
ax.set_ylim(europe_boundaries[1], europe_boundaries[3])
ax.set_title("Population Density SSP3 (2050)", fontsize=14)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

plt.tight_layout()
plt.show()

##### Saving Data
csv_filename = "population_ssp3_europe.csv"
pop_ssp3_hist_df.to_csv(csv_filename, index=False)

print(f"Population for SSP3 (all rows) saved to {csv_filename}")

#### SSP5
pop_ssp5 = nc.Dataset(r"C:\Data\Population Density\SSP5\population_ssp5soc_0p5deg_annual_2006-2100.nc4")

time_pop_ssp5 = pop_ssp5.variables["time"]

lat_pop_ssp5 = pop_ssp5.variables["lat"]

lon_pop_ssp5 = pop_ssp5.variables["lon"]


latitudes_ssp5 = lat_pop_ssp5[:]

longitudes_ssp5 = lon_pop_ssp5[:]

latitudes_2d_ssp5, longitudes_2d_ssp5 = np.meshgrid(latitudes_ssp5, 
                                          longitudes_ssp5, 
                                          indexing="ij")

latitudes_flat_ssp5 = latitudes_2d_ssp5.flatten()

longitudes_flat_ssp5 = longitudes_2d_ssp5.flatten()

total_pop_ssp5 = pop_ssp5.variables["number_of_people"]
total_pop_ssp5_density = total_pop_ssp5[:].flatten()/((111.32 * 0.5) * 
                                                      (111.32 * 0.5 * 
                                                       np.cos(np.radians(np.tile(latitudes_flat_ssp5, 
                                                                                 len(time_pop_ssp5))))))


pop_ssp5_df = pd.DataFrame({
"Year": np.repeat(times_pop_scenario, len(latitudes_flat_ssp5)),
"Lat": np.tile(latitudes_flat_ssp5, len(times_pop_scenario)),
"Lon": np.tile(longitudes_flat_ssp5, len(times_pop_scenario)),
"population": total_pop_ssp5[:].flatten(),
"population_density": total_pop_ssp5_density
})

##### Limiting to Europe
pop_ssp5_df = pop_ssp5_df.loc[(pop_ssp5_df["Lat"] <= europe_boundaries[3]) &
                              (pop_ssp5_df["Lat"] >= europe_boundaries[1]) &
                              (pop_ssp5_df["Lon"] <= europe_boundaries[2]) &
                              (pop_ssp5_df["Lon"] >= europe_boundaries[0])]

print(pop_ssp5_df)

pop_ssp5_df['population_density'] = np.log1p(pop_ssp5_df['population_density'])
pop_ssp5_df['population_density_scaled'] = (pop_ssp5_df['population_density']-mean_population_density)/std_population_density


##### Mergin with historical data
pop_ssp5_hist_df = pd.concat([pop_hist_df, pop_ssp5_df], ignore_index=True)
pop_ssp5_hist_df['population_density'] = pd.to_numeric(pop_ssp5_hist_df['population_density'])
pop_ssp5_hist_df['Year'] = pd.to_datetime(pop_ssp5_hist_df['Year']).dt.year

pop_ssp5_hist_df = pop_ssp5_hist_df[(pop_ssp5_hist_df['Year']<= 2050) &
                                    (pop_ssp5_hist_df['Year'] >= 2015)].sort_values(by=['Year','Lon','Lat']).reset_index(drop=True)

mask = pop_ssp5_hist_df.set_index(['Year', 'Lon', 'Lat']).index.isin(
    fpc_ssp3_df.set_index(['Year', 'Lon', 'Lat']).index
)

pop_ssp5_hist_df = pop_ssp5_hist_df[mask].reset_index(drop=True)

#### Plotting
pop_ssp5_2050_df = pop_ssp5_hist_df[pop_ssp5_hist_df['Year']==2050]
pop_ssp5_2050_df['population_density'] = pd.to_numeric(pop_ssp5_2050_df['population_density'])

pop_ssp5_2050_df = geopandas.GeoDataFrame(
    pop_ssp5_2050_df, 
    geometry=geopandas.points_from_xy(pop_ssp5_2050_df.Lon, 
                                      pop_ssp5_2050_df.Lat), 
    crs="EPSG:4326"
)

fig, ax = plt.subplots(figsize=(12,10))
europe.plot(color='lightgrey', edgecolor='black', ax=ax)

pop_ssp5_2050_df.plot(
    ax=ax,
    column='population_density',
    cmap = 'gist_earth',
    markersize = 150,
    marker='s',
    legend=True,
    legend_kwds = {
        'label': 'Population Density (N/km2)',
        'orientation':'vertical'}
    )

ax.set_xlim(europe_boundaries[0], europe_boundaries[2])
ax.set_ylim(europe_boundaries[1], europe_boundaries[3])
ax.set_title("Population Density SSP5 (2050)", fontsize=14)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

plt.tight_layout()
plt.show()


##### Saving Data
csv_filename = "population_ssp5_europe.csv"
pop_ssp5_hist_df.to_csv(csv_filename, index=False)

print(f"Population for SSP5 (all rows) saved to {csv_filename}")


### Nighttime light
#### SSP1
directory_ssp1 = [r"C:\Data\Nighttime Light Downscaling\Historical", 
                  r"C:\Data\Nighttime Light Downscaling\SSP1_NTL"]
output_dir_ssp1 = r"C:\Data\Nighttime Light Downscaling\SSP1_Parquet_05deg"

resample_tif_to_csv(directory_ssp1, output_dir_ssp1, 
                              boundaries= europe_boundaries, 
                              time_period = "ssp1", resolution= resolution, 
                              variable = 'ntl', target_df=fpc_ssp3_df)


##### Merging SSP1
df_merged_ssp1 = pd.read_csv(r"C:\Data\Nighttime Light Downscaling\SSP1_Parquet_05deg\df_ssp1_05deg.csv")

mask = df_merged_ssp1.set_index(['Lon',
                                 'Lat']).index.isin(fpc_ssp3_df.set_index(['Lon', 
                                                                           'Lat']).index)
df_merged_ssp1 = df_merged_ssp1[mask].sort_values(by=['Year', 
                                                      'Lon', 
                                                      'Lat']).reset_index(drop=True)

df_annual_ntl_ssp1 = annual_interpolation(df_merged_ssp1, 'ntl', 'ntl_scaled')

mask = df_annual_ntl_ssp1.set_index(['Year', 'Lon', 'Lat']).index.isin(
    fpc_ssp3_df.set_index(['Year', 'Lon', 'Lat']).index
)

df_annual_ntl_ssp1 = df_annual_ntl_ssp1[mask]

df_annual_ntl_ssp1.to_csv(r"C:\Data\Nighttime Light Downscaling\SSP1_Parquet_05deg\df_final_ssp1_ntl.csv", index=False)
df_annual_ntl_ssp1 = pd.read_csv(r"C:\Data\Nighttime Light Downscaling\SSP1_Parquet_05deg\df_final_ssp1_ntl.csv")
df_annual_ntl_ssp1 = df_annual_ntl_ssp1[df_annual_ntl_ssp1['Lon'].mul(100).mod(100).astype(int).isin([25, 75])].reset_index(drop=True)

#### SSP3
directory_ssp3 = [r"C:\Data\Nighttime Light Downscaling\Historical", r"C:\Data\Nighttime Light Downscaling\SSP3_NTL"]
output_dir_ssp3 = r"C:\Data\Nighttime Light Downscaling\SSP3_Parquet_05deg"

resample_tif_to_csv(directory_ssp3, output_dir_ssp3, 
                              boundaries= europe_boundaries, 
                              time_period = "ssp3", resolution= resolution, 
                              variable = 'ntl', target_df=fpc_ssp3_df)

df_merged_ssp3 = pd.read_csv(r"C:\Data\Nighttime Light Downscaling\SSP3_Parquet_05deg\df_ssp3_05deg.csv")
df_merged_ssp3
mask = df_merged_ssp3.set_index(['Year',
                                 'Lon',
                                 'Lat']).index.isin(fpc_ssp3_df.set_index(['Year',
                                                                           'Lon', 
                                                                           'Lat']).index)

df_merged_ssp3 = df_merged_ssp3[mask].sort_values(by=['Year', 
                                                      'Lon', 
                                                      'Lat']).reset_index(drop=True)

df_annual_ntl_ssp3 = annual_interpolation(df_merged_ssp3, 'ntl', 'ntl_scaled')

df_annual_ntl_ssp3.to_csv(r"C:\Data\Nighttime Light Downscaling\SSP3_Parquet_05deg\df_final_ssp3_ntl.csv", index=False)
df_annual_ntl_ssp3 = pd.read_csv(r"C:\Data\Nighttime Light Downscaling\SSP3_Parquet_05deg\df_final_ssp3_ntl.csv")

#### SSP5
directory_ssp5 = [r"C:\Data\Nighttime Light Downscaling\Historical", r"C:\Data\Nighttime Light Downscaling\SSP5_NTL"]
output_dir_ssp5 = r"C:\Data\Nighttime Light Downscaling\SSP5_Parquet_05deg"

resample_tif_to_csv(directory_ssp5, output_dir_ssp5, 
                              boundaries= europe_boundaries, 
                              time_period = "ssp5", resolution= resolution, 
                              variable = 'ntl', target_df=fpc_ssp3_df)


df_merged_ssp5 = pd.read_csv(r"C:\Data\Nighttime Light Downscaling\SSP5_Parquet_05deg\df_ssp5_05deg.csv")

mask = df_merged_ssp5.set_index(['Year',
                                 'Lon',
                                 'Lat']).index.isin(fpc_ssp3_df.set_index(['Year',
                                                                           'Lon', 
                                                                           'Lat']).index)

df_merged_ssp5 = df_merged_ssp5[mask].sort_values(by=['Year', 
                                                      'Lon', 
                                                      'Lat']).reset_index(drop=True)

df_annual_ntl_ssp5 = annual_interpolation(df_merged_ssp5, 'ntl', 'ntl_scaled')

df_annual_ntl_ssp5.to_csv(r"C:\Data\Nighttime Light Downscaling\SSP5_Parquet_05deg\df_final_ssp5_ntl.csv", index=False)
df_annual_ntl_ssp5 = pd.read_csv(r"C:\Data\Nighttime Light Downscaling\SSP5_Parquet_05deg\df_final_ssp5_ntl.csv")


### Road Impact
#### Type 1
type_1_road = r"C:\Data\Road Impact\Road Data\GRIP4_density_tp1\grip4_tp1_dens_m_km2.asc"
table_type_1_road = np.loadtxt(type_1_road, skiprows=6)
n_rows, n_cols = table_type_1_road.shape

lon = -180 + np.arange(n_cols)*0.083333333333333
lat = 90 - np.arange(n_rows)*0.083333333333333

lon_grid, lat_grid = np.meshgrid(lon, lat)

table_type_1_road_df = pd.DataFrame({
    'Lon': lon_grid.ravel(),
    'Lat': lat_grid.ravel(),
    'road':table_type_1_road.ravel()})

table_type_1_road_df = snap_to_grid(table_type_1_road_df, resolution, -0.25)
table_type_1_road_df['road']

mask = table_type_1_road_df.set_index(['Lon', 'Lat']).index.isin(
    fpc_ssp3_df.set_index(['Lon', 'Lat']).index
)

table_type_1_road_df = table_type_1_road_df[mask].sort_values(by=['Lon', 
                                                                  'Lat']).reset_index(drop=True)
table_type_1_road_df = table_type_1_road_df.replace(-9999.00, 0)

table_type_1_road_df['road'] = np.log1p(table_type_1_road_df['road'])


table_type_1_road_df_agg = table_type_1_road_df.groupby(['Lon', 
                                                         'Lat'], as_index=False).agg(
    road=('road', 'mean'),
    var = ('road', 'var'))

table_type_1_road_df_agg = geopandas.GeoDataFrame(
    table_type_1_road_df_agg, 
    geometry=geopandas.points_from_xy(table_type_1_road_df_agg.Lon, 
                                      table_type_1_road_df_agg.Lat), 
    crs="EPSG:4326"
)

fig, ax = plt.subplots(figsize=(12,10))
europe.plot(color='lightgrey', edgecolor='black', ax=ax)

table_type_1_road_df_agg.plot(
    ax=ax,
    column='road',
    cmap = 'gist_earth',
    markersize = 150,
    marker='s',
    legend=True,
    legend_kwds = {
        'label': 'Road Density (km/km2)',
        'orientation':'vertical'}
    )

ax.set_xlim(europe_boundaries[0], europe_boundaries[2])
ax.set_ylim(europe_boundaries[1], europe_boundaries[3])
ax.set_title("Road Type 1", fontsize=14)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

plt.tight_layout()
plt.show()

#### Type 2
type_2_road = r"C:\Data\Road Impact\Road Data\GRIP4_density_tp2\grip4_tp2_dens_m_km2.asc"
table_type_2_road = np.loadtxt(type_2_road, skiprows=6)
n_rows, n_cols = table_type_2_road.shape

lon = -180 + np.arange(n_cols)*0.083333333333333
lat = 90 - np.arange(n_rows)*0.083333333333333

lon_grid, lat_grid = np.meshgrid(lon, lat)

table_type_2_road_df = pd.DataFrame({
    'Lon': lon_grid.ravel(),
    'Lat': lat_grid.ravel(),
    'road':table_type_2_road.ravel()})

table_type_2_road_df = snap_to_grid(table_type_2_road_df, resolution, -0.25)

mask = table_type_2_road_df.set_index(['Lon', 'Lat']).index.isin(
    fpc_ssp3_df.set_index(['Lon', 'Lat']).index
)

table_type_2_road_df = table_type_2_road_df[mask].sort_values(by=['Lon', 
                                                                  'Lat']).reset_index(drop=True)
table_type_2_road_df = table_type_2_road_df.replace(-9999.00, 0)

table_type_2_road_df['road'] = np.log1p(table_type_2_road_df['road'])

table_type_2_road_df_agg = table_type_2_road_df.groupby(['Lon', 
                                                         'Lat'], as_index=False).agg(
    road=('road', 'mean'),
    var = ('road', 'var'))

table_type_2_road_df_agg = geopandas.GeoDataFrame(
    table_type_2_road_df_agg, 
    geometry=geopandas.points_from_xy(table_type_2_road_df_agg.Lon, 
                                      table_type_2_road_df_agg.Lat), 
    crs="EPSG:4326"
)
table_type_2_road_df

fig, ax = plt.subplots(figsize=(12,10))
europe.plot(color='lightgrey', edgecolor='black', ax=ax)

table_type_2_road_df_agg.plot(
    ax=ax,
    column='road',
    cmap = 'gist_earth',
    markersize = 150,
    marker='s',
    legend=True,
    legend_kwds = {
        'label': 'Road Density (km/km2)',
        'orientation':'vertical'}
    )

ax.set_xlim(europe_boundaries[0], europe_boundaries[2])
ax.set_ylim(europe_boundaries[1], europe_boundaries[3])
ax.set_title("Road Type 2", fontsize=14)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

plt.tight_layout()
plt.show()


#### Type 3
type_3_road = r"C:\Data\Road Impact\Road Data\GRIP4_density_tp3\grip4_tp3_dens_m_km2.asc"
table_type_3_road = np.loadtxt(type_3_road, skiprows=6)
n_rows, n_cols = table_type_3_road.shape

lon = -180 + np.arange(n_cols)*0.083333333333333
lat = 90 - np.arange(n_rows)*0.083333333333333

lon_grid, lat_grid = np.meshgrid(lon, lat)

table_type_3_road_df = pd.DataFrame({
    'Lon': lon_grid.ravel(),
    'Lat': lat_grid.ravel(),
    'road':table_type_3_road.ravel()})

table_type_3_road_df = snap_to_grid(table_type_3_road_df, resolution, -0.25)

mask = table_type_3_road_df.set_index(['Lon', 'Lat']).index.isin(
    fpc_ssp3_df.set_index(['Lon', 'Lat']).index
)

table_type_3_road_df = table_type_3_road_df[mask].sort_values(by=['Lon', 
                                                                  'Lat']).reset_index(drop=True)
table_type_3_road_df = table_type_3_road_df.replace(-9999.00, 0)

table_type_3_road_df['road'] = np.log1p(table_type_3_road_df['road'])

table_type_3_road_df_agg = table_type_3_road_df.groupby(['Lon', 'Lat'], as_index=False).agg(
    road=('road', 'mean'),
    var = ('road', 'var'))



table_type_3_road_df_agg = geopandas.GeoDataFrame(
    table_type_3_road_df_agg, 
    geometry=geopandas.points_from_xy(table_type_3_road_df_agg.Lon, 
                                      table_type_3_road_df_agg.Lat), 
    crs="EPSG:4326"
)


fig, ax = plt.subplots(figsize=(12,10))
europe.plot(color='lightgrey', edgecolor='black', ax=ax)

table_type_3_road_df_agg.plot(
    ax=ax,
    column='road',
    cmap = 'gist_earth',
    markersize = 150,
    marker='s',
    legend=True,
    legend_kwds = {
        'label': 'Road Density (km/km2)',
        'orientation':'vertical'}
    )

ax.set_xlim(europe_boundaries[0], europe_boundaries[2])
ax.set_ylim(europe_boundaries[1], europe_boundaries[3])
ax.set_title("Road Type 3", fontsize=14)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

plt.tight_layout()
plt.show()

#### Type 4
type_4_road = r"C:\Data\Road Impact\Road Data\GRIP4_density_tp4\grip4_tp4_dens_m_km2.asc"
table_type_4_road = np.loadtxt(type_4_road, skiprows=6)
n_rows, n_cols = table_type_4_road.shape

lon = -180 + np.arange(n_cols)*0.083333333333333
lat = 90 - np.arange(n_rows)*0.083333333333333

lon_grid, lat_grid = np.meshgrid(lon, lat)

table_type_4_road_df = pd.DataFrame({
    'Lon': lon_grid.ravel(),
    'Lat': lat_grid.ravel(),
    'road':table_type_4_road.ravel()})

table_type_4_road_df = table_type_4_road_df.sort_values(by=['Lon',
                                                            'Lat']).reset_index(drop=True)

table_type_4_road_df = snap_to_grid(table_type_4_road_df, resolution, -0.25)

mask = table_type_4_road_df.set_index(['Lon', 'Lat']).index.isin(
    fpc_ssp3_df.set_index(['Lon', 'Lat']).index
)

table_type_4_road_df = table_type_4_road_df[mask].sort_values(by=['Lon', 
                                                                  'Lat']).reset_index(drop=True)
table_type_4_road_df = table_type_4_road_df.replace(-9999.00, 0)

table_type_4_road_df['road'] = np.log1p(table_type_4_road_df['road'])

table_type_4_road_df_agg = table_type_4_road_df.groupby(['Lon', 
                                                         'Lat'], 
                                                        as_index=False).agg(
    road=('road', 'mean'),
    var = ('road', 'var'))

table_type_4_road_df_agg = geopandas.GeoDataFrame(
    table_type_4_road_df_agg, 
    geometry=geopandas.points_from_xy(table_type_4_road_df_agg.Lon, 
                                      table_type_4_road_df_agg.Lat), 
    crs="EPSG:4326"
)
table_type_4_road_df_agg

fig, ax = plt.subplots(figsize=(12,10))
europe.plot(color='lightgrey', edgecolor='black', ax=ax)

table_type_4_road_df_agg.plot(
    ax=ax,
    column='road',
    cmap = 'gist_earth',
    markersize = 150,
    marker='s',
    legend=True,
    legend_kwds = {
        'label': 'Road Density (km/km2)',
        'orientation':'vertical'}
    )

ax.set_xlim(europe_boundaries[0], europe_boundaries[2])
ax.set_ylim(europe_boundaries[1], europe_boundaries[3])
ax.set_title("Road Type 4", fontsize=14)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

plt.tight_layout()
plt.show()

#### Type 5
type_5_road = r"C:\Data\Road Impact\Road Data\GRIP4_density_tp5\grip4_tp5_dens_m_km2.asc"
table_type_5_road = np.loadtxt(type_5_road, skiprows=6)
n_rows, n_cols = table_type_5_road.shape

lon = -180 + np.arange(n_cols)*0.083333333333333
lat = 90 - np.arange(n_rows)*0.083333333333333

lon_grid, lat_grid = np.meshgrid(lon, lat)

table_type_5_road_df = pd.DataFrame({
    'Lon': lon_grid.ravel(),
    'Lat': lat_grid.ravel(),
    'road':table_type_5_road.ravel()})

table_type_5_road_df[table_type_5_road_df['road']==-9999] = 0
table_type_5_road_df = snap_to_grid(table_type_5_road_df, resolution, -0.25)

mask = table_type_5_road_df.set_index(['Lon', 'Lat']).index.isin(
    fpc_ssp3_df.set_index(['Lon', 'Lat']).index
)

table_type_5_road_df = table_type_5_road_df[mask].sort_values(by=['Lon', 
                                                                  'Lat']).reset_index(drop=True)
table_type_5_road_df = table_type_5_road_df.replace(-9999.00, 0)

table_type_5_road_df['road'] = np.log1p(table_type_5_road_df['road'])

table_type_5_road_df_agg = table_type_5_road_df.groupby(['Lon', 
                                                         'Lat'], 
                                                        as_index=False).agg(
                                                            road=('road', 'mean'),
                                                            var = ('road', 'var'))

table_type_5_road_df_agg = geopandas.GeoDataFrame(
    table_type_5_road_df_agg, 
    geometry=geopandas.points_from_xy(table_type_5_road_df_agg.Lon, 
                                      table_type_5_road_df_agg.Lat), 
    crs="EPSG:4326"
)

fig, ax = plt.subplots(figsize=(12,10))
europe.plot(color='lightgrey', edgecolor='black', ax=ax)

table_type_5_road_df_agg.plot(
    ax=ax,
    column='road',
    cmap = 'gist_earth',
    markersize = 150,
    marker='s',
    legend=True,
    legend_kwds = {
        'label': 'Road Denstiy (km/km2)',
        'orientation':'vertical'}
    )

ax.set_xlim(europe_boundaries[0], europe_boundaries[2])
ax.set_ylim(europe_boundaries[1], europe_boundaries[3])
ax.set_title("Road Type 5", fontsize=14)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

plt.tight_layout()
plt.show()

### OECD Membership & Land Area
cc = coco.CountryConverter()

#### OECD
europe_iso3 = pd.Series(european_countries, name='iso3')
europe_oecd = (cc.pandas_convert(series = europe_iso3, to='ISO3')).to_frame()
print(europe_oecd)

oecd = pd.DataFrame(OECD, columns=['OECD'])
print(oecd)

europe_oecd = europe_oecd.assign(OECD=europe_oecd.iso3.isin(oecd.OECD).astype(int), 
                                 country=european_countries)
print(europe_oecd)

europe_oecd = europe_oecd.sort_values(by=['country'], ignore_index = True)
print(europe_oecd)

#### Land Area
url = "https://raw.githubusercontent.com/datumorphism/dataset-european-countries/master/dataset/european_countries.csv"
df_land_area = pd.read_csv(url)
df_land_area = df_land_area.drop(columns=['alpha_2', 'alt_names', 
                                          'numeric', 'area_comment'])

df_land_area = df_land_area.sort_values(by=['country_name'], ignore_index = True)
print(df_land_area)

europe_oecd['land_area'] = df_land_area['area_km2']
print(europe_oecd)

#### Combined Dataframe
europe_land_area_oecd_df = pd.DataFrame([])
europe_land_area_oecd_df['Lon'] = fpc_ssp3_df[fpc_ssp3_df['Year']==2015]['Lon'].values
europe_land_area_oecd_df['Lat'] = fpc_ssp3_df[fpc_ssp3_df['Year']==2015]['Lat'].values
coords = list(zip(europe_land_area_oecd_df['Lat'], europe_land_area_oecd_df['Lon']))
results = rg.search(coords)
europe_land_area_oecd_df['country'] = [coco.convert(r['cc'], to='short_name') for r in results]
europe_land_area_oecd_df = europe_land_area_oecd_df.replace('Czechia','Czech Republic')
print(europe_land_area_oecd_df)

europe_land_area_oecd_df['land_area'] = np.nan
area_map_land_area = europe_oecd.dropna(subset=['country', 
                                                'land_area']).set_index('country')['land_area']

europe_land_area_oecd_df['land_area'] = europe_land_area_oecd_df['land_area'].fillna(europe_land_area_oecd_df['country'].map(area_map_land_area))

europe_land_area_oecd_df['oecd'] = np.nan
area_map_oecd = europe_oecd.dropna(subset=['country',
                                           'OECD']).set_index('country')['OECD']
europe_land_area_oecd_df['oecd'] = europe_land_area_oecd_df['oecd'].fillna(europe_land_area_oecd_df['country'].map(area_map_oecd))

europe_land_area_oecd_df.to_csv(r"C:\Data\Road Impact\OECD & Land Area\land_area_oecd_df.csv", index=False)
europe_land_area_oecd_df = pd.read_csv(r"C:\Data\Road Impact\OECD & Land Area\land_area_oecd_df.csv")

#### Plotting
marks = europe_land_area_oecd_df['country'].unique()
marks[marks == 'Czech Republic'] = 'Czechia'
marks = marks[(marks != ['Russia']) & 
              (marks != ['Ukraine']) & 
              (marks != ['Belarus'])]

marked_countries = europe[europe['ADMIN'].isin(marks)]

ax = europe.plot(figsize=(15,10),color='lightgrey',edgecolor='black')
marked_countries.plot(ax=ax, color='grey', edgecolor='black')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

### GDP
#### SSP1
directory_gdp_ssp1 = [r"C:\Data\Road Impact\GDP\Historical", 
                      r"C:\Data\Road Impact\GDP\SSP1"]
output_gdp_ssp1 = r"C:\Data\Road Impact\GDP\SSP1_Parquet_05deg"

resample_tif_to_csv(directory_gdp_ssp1, 
                    output_gdp_ssp1, 
                    boundaries = europe_boundaries, 
                    time_period = 'ssp1',
                    resolution= resolution, variable =  'gdp', 
                    target_df=fpc_ssp3_df, training_year = 2005)


df_merged_gdp_ssp1 = pd.read_csv(r"C:\Data\Road Impact\GDP\SSP1_Parquet_05deg\df_ssp1_05deg.csv")
df_merged_gdp_ssp1.sort_values(by=['Year', 'Lon', 'Lat'])

df_annual_gdp_ssp1 = annual_interpolation(df_merged_gdp_ssp1, 
                                          'gdp', 
                                          'gdp_scaled')

df_annual_gdp_ssp1 = df_annual_gdp_ssp1.reset_index(drop=True)

df_annual_gdp_ssp1.to_csv(r"C:\Data\Road Impact\GDP\SSP1_Parquet_05deg\final_gdp_ssp1.csv", 
                          index=False)
df_annual_gdp_ssp1 = pd.read_csv(r"C:\Data\Road Impact\GDP\SSP1_Parquet_05deg\final_gdp_ssp1.csv").sort_values(by=['Year', 'Lon', 'Lat'])

mask = df_annual_gdp_ssp1.set_index([ 'Lon', 'Lat']).index.isin(
    fpc_ssp3_df.set_index([ 'Lon', 'Lat']).index
)

df_annual_gdp_ssp1 = df_annual_gdp_ssp1[mask].reset_index(drop=True)

##### Plotting
df_annual_gdp_ssp1_2050 = df_annual_gdp_ssp1[df_annual_gdp_ssp1['Year']==2050]
df_annual_gdp_ssp1_2050 = df_annual_gdp_ssp1_2050.drop(columns=['Year'])

df_annual_gdp_ssp1_2050 = geopandas.GeoDataFrame(
    df_annual_gdp_ssp1_2050, 
    geometry=geopandas.points_from_xy(df_annual_gdp_ssp1_2050.Lon, 
                                      df_annual_gdp_ssp1_2050.Lat), 
    crs="EPSG:4326"
)


fig, ax = plt.subplots(figsize=(12,10))
europe.plot(color='lightgrey', edgecolor='black', ax=ax)

df_annual_gdp_ssp1_2050.plot(
    ax=ax,
    column='gdp',
    cmap = 'gist_earth',
    markersize = 150,
    marker='s',
    legend=True,
    legend_kwds = {
        'label': 'GDP (PPP 2005 international dollars/km2)',
        'orientation':'vertical'}
    )

ax.set_xlim(europe_boundaries[0], europe_boundaries[2])
ax.set_ylim(europe_boundaries[1], europe_boundaries[3])
ax.set_title("GDP Density SSP1 (2050)" , fontsize=14)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

plt.tight_layout()
plt.show()

#### SSP3
directory_gdp_ssp3 = [r"C:\Data\Road Impact\GDP\Historical", 
                      r"C:\Data\Road Impact\GDP\SSP3\SSP3"]

output_gdp_ssp3 = r"C:\Data\Road Impact\GDP\SSP3_Parquet_05deg"

resample_tif_to_csv(directory_gdp_ssp3, 
                              output_gdp_ssp3, 
                              boundaries= europe_boundaries,
                              time_period = 'ssp3',
                              resolution = resolution, variable = 'gdp',
                              target_df = fpc_ssp3_df, training_year = 2005)


df_merged_gdp_ssp3 = pd.read_csv(r"C:\Data\Road Impact\GDP\SSP3_Parquet_05deg\df_ssp3_05deg.csv")

df_annual_gdp_ssp3 = annual_interpolation(df_merged_gdp_ssp3, 'gdp', 'gdp_scaled')

df_annual_gdp_ssp3 = df_annual_gdp_ssp3.reset_index(drop=True)

df_annual_gdp_ssp3.to_csv(r"C:\Data\Road Impact\GDP\SSP3_Parquet_05deg\final_gdp_ssp3.csv", 
                          index=False)

df_annual_gdp_ssp3 = pd.read_csv(r"C:\Data\Road Impact\GDP\SSP3_Parquet_05deg\final_gdp_ssp3.csv").sort_values(by=['Year', 'Lon', 'Lat'])

mask = df_annual_gdp_ssp3.set_index(['Lon', 'Lat']).index.isin(
    fpc_ssp3_df.set_index(['Lon', 'Lat']).index
)

df_annual_gdp_ssp3 = df_annual_gdp_ssp3[mask].reset_index(drop=True)

##### Plotting
df_annual_gdp_ssp3_2050 = df_annual_gdp_ssp3[df_annual_gdp_ssp3['Year']==2050]

df_annual_gdp_ssp3_2050 = geopandas.GeoDataFrame(
    df_annual_gdp_ssp3_2050, 
    geometry=geopandas.points_from_xy(df_annual_gdp_ssp3_2050.Lon, 
                                      df_annual_gdp_ssp3_2050.Lat), 
    crs="EPSG:4326"
)

fig, ax = plt.subplots(figsize=(12,10))
europe.plot(color='lightgrey', edgecolor='black', ax=ax)

df_annual_gdp_ssp3_2050.plot(
    ax=ax,
    column='gdp',
    cmap = 'gist_earth',
    markersize = 150,
    marker='s',
    legend=True,
    legend_kwds = {
        'label': 'GDP (PPP 2005 international dollars/km2)',
        'orientation':'vertical'}
    )

ax.set_xlim(europe_boundaries[0], europe_boundaries[2])
ax.set_ylim(europe_boundaries[1], europe_boundaries[3])
ax.set_title("GDP Density SSP3 (2050)", fontsize=14)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

plt.tight_layout()
plt.show()

#### SSP5
directory_gdp_ssp5 = r"C:\Data\Road Impact\GDP\SSP5"
output_gdp_ssp5 = r"C:\Data\Road Impact\GDP\SSP5_Parquet_05deg"

resample_tif_to_csv(directory_gdp_ssp5, 
                              output_gdp_ssp5, 
                              boundaries= europe_boundaries, 
                              time_period = 'ssp5',
                              resolution = resolution, variable = 'gdp', 
                              target_df=fpc_ssp3_df, training_year = 2005)


df_merged_gdp_ssp5 = pd.read_csv(r"C:\Data\Road Impact\GDP\SSP5_Parquet_05deg\df_ssp5_05deg.csv").sort_values(by=['Year','Lon','Lat'])

df_annual_gdp_ssp5 = annual_interpolation(df_merged_gdp_ssp5, 'gdp', 'gdp_scaled').reset_index(drop=True)

df_annual_gdp_ssp5.to_csv(r"C:\Data\Road Impact\GDP\SSP5_Parquet_05deg\final_gdp_ssp5.csv", index=False)
df_annual_gdp_ssp5 = pd.read_csv(r"C:\Data\Road Impact\GDP\SSP5_Parquet_05deg\final_gdp_ssp5.csv")

mask = df_annual_gdp_ssp5.set_index(['Lon', 'Lat']).index.isin(
    fpc_ssp3_df.set_index(['Lon', 'Lat']).index
)

df_annual_gdp_ssp5 = df_annual_gdp_ssp5[mask]

##### Plotting
df_gdp_ssp5_2050 = df_annual_gdp_ssp5[df_annual_gdp_ssp5['Year']==2050]
df_gdp_ssp5_2050 = df_gdp_ssp5_2050.drop(columns=['Year'])

df_gdp_ssp5_2050 = geopandas.GeoDataFrame(
    df_gdp_ssp5_2050, 
    geometry=geopandas.points_from_xy(df_gdp_ssp5_2050.Lon, 
                                      df_gdp_ssp5_2050.Lat), 
    crs="EPSG:4326"
)

fig, ax = plt.subplots(figsize=(12,10))
europe.plot(color='lightgrey', edgecolor='black', ax=ax)

df_gdp_ssp5_2050.plot(
    ax=ax,
    column='gdp_scaled',
    cmap = 'gist_earth',
    markersize = 10,
    legend=True,
    legend_kwds = {
        'label': 'GDP (PPP 2005 international dollars/km2)',
        'orientation':'vertical'}
    )

ax.set_xlim(europe_boundaries[0], europe_boundaries[2])
ax.set_ylim(europe_boundaries[1], europe_boundaries[3])
ax.set_title("GDP Density SSP5 (2050)", fontsize=14)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

plt.tight_layout()
plt.show()

### Population Density
"""
See other population denstiy
"""

# Model: Road Impact
## Model Fitting
train_gdp_df_fit = df_annual_gdp_ssp5[df_annual_gdp_ssp5['Year']==2005][['gdp_scaled', 
                                                                         'var_scaled']].reset_index(drop=True)
train_gdp_df_fit
pop_hist_2005_df_train = pop_hist_2005_df.sort_values(by=['Lon', 
                                                    'Lat'])['population_density_scaled'].reset_index(drop=True)

pop_hist_2005_df_train = pop_hist_2005_df_train.reset_index(drop=True)

oecd_train = europe_land_area_oecd_df[['oecd', 'country']].reset_index(drop=True)
land_area_train = pd.DataFrame(europe_land_area_oecd_df['land_area'].reset_index(drop=True))
land_area_train_mean = land_area_train.mean()
land_area_train_std = np.sqrt(land_area_train.var())
land_area_train['land_area_scaled'] = (land_area_train-land_area_train_mean)/land_area_train_std

dataframes_train = [oecd_train, land_area_train['land_area_scaled'],
                          train_gdp_df_fit['gdp_scaled'],
                          pop_hist_2005_df_train]

road_X = pd.concat(dataframes_train, axis=1, ignore_index=True)
road_X = road_X.rename(columns={0:'oecd', 1: 'country', 
                                            2:'land_area_scaled', 
                                            3:'gdp_scaled',
                                            4:'population_density_scaled'})

countries = road_X['country'].unique()
cv_splits = []

for country in countries:
    test_idx = np.where(road_X['country'] == country)[0]
    train_idx = np.where(road_X['country'] != country)[0]
    if len(test_idx) < 5 or len(train_idx) < 5:
        continue
    cv_splits.append((train_idx, test_idx))

param_grid = {
    'alpha_1':[1e-4, 1e-5, 1e-6, 1e-7, 1e-8],
    'alpha_2':[1e-4, 1e-5, 1e-6, 1e-7, 1e-8],
    'lambda_1':[1e-4, 1e-5, 1e-6, 1e-7, 1e-8],
    'lambda_2':[1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
    }

### Correlation
road_train_X = road_X.drop(columns=['country'])
road_train_X_corr = road_train_X.corr()

### Type 1
grid_search_type1 = GridSearchCV(
    BayesianRidge(),
    param_grid,
    cv = cv_splits,
    scoring=['r2','neg_mean_squared_error', 'neg_mean_absolute_error'],
    n_jobs=-1,
    verbose=0,
    refit='neg_mean_squared_error')

grid_search_type1.fit(road_train_X, table_type_1_road_df_agg['road'])
results = grid_search_type1.cv_results_
best_idx_type1 = grid_search_type1.best_index_
road_train_X
grid_search_type1_rmse = np.sqrt(-grid_search_type1.cv_results_['mean_test_neg_mean_squared_error'][best_idx_type1])
print(grid_search_type1_rmse)

grid_search_type1_mae = -grid_search_type1.cv_results_['mean_test_neg_mean_absolute_error'][best_idx_type1]
print(grid_search_type1_mae)
np.sqrt(np.diag(grid_search_type1.best_estimator_.sigma_))

print(grid_search_type1.best_params_)
print(np.sqrt(-grid_search_type1.best_score_))

road_model_type1_bayesian = BayesianRidge(**grid_search_type1.best_params_)
road_model_type1_bayesian.fit(road_train_X, table_type_1_road_df_agg['road'])

road_model_type1_bayesian_coef = road_model_type1_bayesian.coef_
print(road_model_type1_bayesian.coef_)

### Type 2
grid_search_type2 = GridSearchCV(
    BayesianRidge(),
    param_grid,
    cv = cv_splits,
    scoring=['r2','neg_mean_squared_error', 'neg_mean_absolute_error'],
    n_jobs=-1,
    verbose=0,
    refit='neg_mean_squared_error')

grid_search_type2.fit(road_train_X, table_type_2_road_df_agg['road'])

best_idx_type2 = grid_search_type2.best_index_
grid_search_type2_rmse = np.sqrt(-grid_search_type2.cv_results_['mean_test_neg_mean_squared_error'][best_idx_type2])
print(grid_search_type2_rmse)

grid_search_type2_mae = -grid_search_type2.cv_results_['mean_test_neg_mean_absolute_error'][best_idx_type2]
print(grid_search_type2_mae)

print(grid_search_type2.best_params_)
print(np.sqrt(-grid_search_type2.best_score_))

road_model_type2_bayesian = BayesianRidge(**grid_search_type2.best_params_)
road_model_type2_bayesian.fit(road_train_X, table_type_2_road_df_agg['road'])

road_model_type2_bayesian_coef = road_model_type2_bayesian.coef_
print(road_model_type2_bayesian.coef_)

### Type 3
grid_search_type3 = GridSearchCV(
    BayesianRidge(),
    param_grid,
    cv = cv_splits,
    scoring=['r2','neg_mean_squared_error', 'neg_mean_absolute_error'],
    n_jobs=-1,
    verbose=0,
    refit='neg_mean_squared_error')

grid_search_type3.fit(road_train_X, table_type_3_road_df_agg['road'])

best_idx_type3 = grid_search_type3.best_index_
grid_search_type3_rmse = np.sqrt(-grid_search_type3.cv_results_['mean_test_neg_mean_squared_error'][best_idx_type3])
print(grid_search_type3_rmse)

grid_search_type3_mae = -grid_search_type3.cv_results_['mean_test_neg_mean_absolute_error'][best_idx_type3]
print(grid_search_type3_mae)


print(grid_search_type3.best_params_)
print(np.sqrt(-grid_search_type3.best_score_))


road_model_type3_bayesian = BayesianRidge(**grid_search_type3.best_params_)
road_model_type3_bayesian.fit(road_train_X, table_type_3_road_df_agg['road'])

road_model_type3_bayesian_coef = road_model_type3_bayesian.coef_
print(road_model_type3_bayesian.coef_)

### Type 4
grid_search_type4 = GridSearchCV(
    BayesianRidge(),
    param_grid,
    cv = cv_splits,
    scoring=['r2','neg_mean_squared_error', 'neg_mean_absolute_error'],
    n_jobs=-1,
    verbose=0,
    refit='neg_mean_squared_error',
    return_train_score=True)

grid_search_type4.fit(road_train_X, table_type_4_road_df_agg['road'])

best_idx_type4 = grid_search_type4.best_index_
grid_search_typ4_rmse = np.sqrt(-grid_search_type4.cv_results_['mean_test_neg_mean_squared_error'][best_idx_type4])
print(grid_search_typ4_rmse)

grid_search_type4_mae = -grid_search_type4.cv_results_['mean_test_neg_mean_absolute_error'][best_idx_type4]
print(grid_search_type4_mae)

grid_search_type4.best_params_

road_model_type4_bayesian = BayesianRidge(**grid_search_type4.best_params_)
road_model_type4_bayesian.fit(road_train_X, table_type_4_road_df_agg['road'])

road_model_type4_bayesian_coef = road_model_type4_bayesian.coef_
print(road_model_type4_bayesian.coef_)

### Type 5
grid_search_type5 = GridSearchCV(
    BayesianRidge(),
    param_grid,
    cv = cv_splits,
    scoring=['r2','neg_mean_squared_error', 'neg_mean_absolute_error'],
    n_jobs=-1,
    verbose=0,
    refit='neg_mean_squared_error')

grid_search_type5.fit(road_train_X, table_type_5_road_df_agg['road'])


best_idx_type5 = grid_search_type5.best_index_
grid_search_type5_rmse = np.sqrt(-grid_search_type5.cv_results_['mean_test_neg_mean_squared_error'][best_idx_type5])
print(grid_search_type5_rmse)

grid_search_type5_mae = -grid_search_type5.cv_results_['mean_test_neg_mean_absolute_error'][best_idx_type5]
print(grid_search_type5_mae)

print(grid_search_type5.best_params_)
print(np.sqrt(-grid_search_type5.best_score_))

road_model_type5_bayesian = BayesianRidge(**grid_search_type5.best_params_)
road_model_type5_bayesian.fit(road_train_X, table_type_5_road_df_agg['road'])

road_model_type5_bayesian_coef = road_model_type5_bayesian.coef_
print(road_model_type5_bayesian.coef_)


## SSP1
n_years = len(df_annual_gdp_ssp1['Year'].unique())

### GDP
df_annual_gdp_ssp1 = df_annual_gdp_ssp1[(df_annual_gdp_ssp1['Year']<=2050) &
                                        (df_annual_gdp_ssp1['Year']>=2015)]

df_annual_gdp_ssp1_test = df_annual_gdp_ssp1['gdp_scaled'].reset_index(drop=True)
df_annual_gdp_ssp1_test
print(df_annual_gdp_ssp1)

### Pop
pop_ssp1_hist_df = pop_ssp1_hist_df.sort_values(by=['Year','Lon', 'Lat'])
pop_ssp1_hist_df_test = pop_ssp1_hist_df['population_density_scaled'].reset_index(drop=True)
print(pop_ssp1_hist_df_test)

### OECD
oecd_test = pd.DataFrame(np.tile(europe_land_area_oecd_df['oecd'], n_years))
oecd_test = oecd_test.rename(columns={0:'oecd'})
print(oecd_test)

### Land Area
land_area_test = pd.DataFrame(np.tile(europe_land_area_oecd_df['land_area'], n_years))
land_area_test = land_area_test.rename(columns={0:'land_area'})
land_area_test['land_area_scaled'] = (pd.DataFrame(land_area_test['land_area'])-land_area_train_mean)/land_area_train_std
print(land_area_test)

### Test Data
dataframe_test_ssp1 = [oecd_test, land_area_test['land_area_scaled'], 
                       df_annual_gdp_ssp1_test, pop_ssp1_hist_df_test]
dataframe_test_ssp1
road_test_X_ssp1 = pd.concat(dataframe_test_ssp1, axis = 1)
road_test_X_ssp1 = pd.DataFrame(road_test_X_ssp1)
print(road_test_X_ssp1)

### BayesianRidge
#### Type 1
mean_pred_type1, std_pred_type1 = road_model_type1_bayesian.predict(road_test_X_ssp1, 
                                                                    return_std=True)

road_results_ssp1_type1_bayesian_df = pd.DataFrame([])
road_results_ssp1_type1_bayesian_df['road_type1'] = mean_pred_type1
road_results_ssp1_type1_bayesian_df['road_std'] = std_pred_type1
road_results_ssp1_type1_bayesian_df['Year'] = df_annual_gdp_ssp1['Year'].reset_index(drop=True)
road_results_ssp1_type1_bayesian_df['Lon'] = df_annual_gdp_ssp1['Lon'].reset_index(drop=True)
road_results_ssp1_type1_bayesian_df['Lat'] = df_annual_gdp_ssp1['Lat'].reset_index(drop=True)
road_results_ssp1_type1_bayesian_df['road_estimate_weighted'] = weighted_road_densities(road_results_ssp1_type1_bayesian_df, weight, 'road_type1')
road_results_ssp1_type1_bayesian_df['road_var'] = 0.35*(std_pred_type1 ** 2)
print(road_results_ssp1_type1_bayesian_df)


##### Plotting
road_results_ssp1_type1_bayesian_df_2015 = road_results_ssp1_type1_bayesian_df[road_results_ssp1_type1_bayesian_df['Year']==2015]

road_results_ssp1_type1_bayesian_df_2015 = geopandas.GeoDataFrame(
    road_results_ssp1_type1_bayesian_df_2015, 
    geometry=geopandas.points_from_xy(road_results_ssp1_type1_bayesian_df_2015.Lon, 
                                      road_results_ssp1_type1_bayesian_df_2015.Lat), 
    crs="EPSG:4326"
)


fig, ax = plt.subplots(figsize=(12,10))
europe.plot(color='lightgrey', edgecolor='black', ax=ax)

road_results_ssp1_type1_bayesian_df_2015.plot(
    ax=ax,
    column='road_type1',
    cmap = 'gist_earth',
    markersize = 150,
    marker='s',
    legend=True,
    vmin=0,
    vmax=12,
    legend_kwds = {
        'label': 'Road Density (km/km2)',
        'orientation':'vertical'}
    )

ax.set_xlim(europe_boundaries[0], europe_boundaries[2])
ax.set_ylim(europe_boundaries[1], europe_boundaries[3])
ax.set_title("Road Density Type 1, 2015 (SSP1) [log-transformed]", fontsize=14)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

plt.tight_layout()
plt.show()

road_results_ssp1_type1_bayesian_df_2050 = road_results_ssp1_type1_bayesian_df[road_results_ssp1_type1_bayesian_df['Year']==2050]

road_results_ssp1_type1_bayesian_df_2050 = geopandas.GeoDataFrame(
    road_results_ssp1_type1_bayesian_df_2050, 
    geometry=geopandas.points_from_xy(road_results_ssp1_type1_bayesian_df_2050.Lon, 
                                      road_results_ssp1_type1_bayesian_df_2050.Lat), 
    crs="EPSG:4326"
)

fig, ax = plt.subplots(figsize=(12,10))
europe.plot(color='lightgrey', edgecolor='black', ax=ax)

road_results_ssp1_type1_bayesian_df_2050.plot(
    ax=ax,
    column='road_type1',
    cmap = 'gist_earth',
    markersize = 150,
    marker='s',
    legend=True,
    vmin=0,
    vmax=12,
    legend_kwds = {
        'label': 'Road Density (km/km2)',
        'orientation':'vertical'}
    )

ax.set_xlim(europe_boundaries[0], europe_boundaries[2])
ax.set_ylim(europe_boundaries[1], europe_boundaries[3])
ax.set_title("Road Density Type 1, 2050 (SSP1) [log-transformed]", fontsize=14)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

plt.tight_layout()
plt.show()

ssp1_type1_diff = road_results_ssp1_type1_bayesian_df_2050[['road_type1']].reset_index(drop=True)-road_results_ssp1_type1_bayesian_df_2015[['road_type1']].reset_index(drop=True)

n_cells = len(ssp1_type1_diff)
ssp1_type1_increase = np.sum(ssp1_type1_diff['road_type1'] > 0.005)
ssp1_type1_decrease = np.sum(ssp1_type1_diff['road_type1'] < -0.005)
ssp1_type1_unchanged = np.sum(np.isclose(ssp1_type1_diff['road_type1'], 0, atol=0.005))

pct_increased = (int(ssp1_type1_increase)/ n_cells) * 100
print(f'increased: {pct_increased:.1f}%')
pct_decreased = (int(ssp1_type1_decrease) / n_cells) * 100
print(f'decreased: {pct_decreased:.1f}%')
pct_unchanged = (int(ssp1_type1_unchanged) / n_cells) * 100
print(f'unchanged: {pct_unchanged:.1f}%')

#### Type 2
mean_pred_type2, std_pred_type2 = road_model_type2_bayesian.predict(road_test_X_ssp1, 
                                                                    return_std=True)
road_results_ssp1_type2_bayesian_df = pd.DataFrame([])
road_results_ssp1_type2_bayesian_df['road_type2'] = mean_pred_type2
road_results_ssp1_type2_bayesian_df['road_std'] = std_pred_type2
road_results_ssp1_type2_bayesian_df['Year'] = df_annual_gdp_ssp1['Year'].reset_index(drop=True)
road_results_ssp1_type2_bayesian_df['Lon'] = df_annual_gdp_ssp1['Lon'].reset_index(drop=True)
road_results_ssp1_type2_bayesian_df['Lat'] = df_annual_gdp_ssp1['Lat'].reset_index(drop=True)
road_results_ssp1_type2_bayesian_df['road_estimate_weighted'] = weighted_road_densities(road_results_ssp1_type2_bayesian_df, weight, 'road_type2')
road_results_ssp1_type2_bayesian_df['road_var'] = 0.25*(std_pred_type2 ** 2)
print(road_results_ssp1_type2_bayesian_df)

##### Plotting
road_results_ssp1_type2_bayesian_df_2015 = road_results_ssp1_type2_bayesian_df[road_results_ssp1_type2_bayesian_df['Year']==2015]

road_results_ssp1_type2_bayesian_df_2015 = geopandas.GeoDataFrame(
    road_results_ssp1_type2_bayesian_df_2015, 
    geometry=geopandas.points_from_xy(road_results_ssp1_type2_bayesian_df_2015.Lon, 
                                      road_results_ssp1_type2_bayesian_df_2015.Lat), 
    crs="EPSG:4326"
)

fig, ax = plt.subplots(figsize=(12,10))
europe.plot(color='lightgrey', edgecolor='black', ax=ax)

road_results_ssp1_type2_bayesian_df_2015.plot(
    ax=ax,
    column='road_type2',
    cmap = 'gist_earth',
    markersize = 150,
    marker='s',
    legend=True,
    vmin=0,
    vmax=6,
    legend_kwds = {
        'label': 'Road Density (km/km2)',
        'orientation':'vertical'}
    )

ax.set_xlim(europe_boundaries[0], europe_boundaries[2])
ax.set_ylim(europe_boundaries[1], europe_boundaries[3])
ax.set_title("Road Density Type 2, 2015 (SSP1) [log-transformed]", fontsize=14)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

plt.tight_layout()
plt.show()

road_results_ssp1_type2_bayesian_df_2050 = road_results_ssp1_type2_bayesian_df[road_results_ssp1_type2_bayesian_df['Year']==2050]

road_results_ssp1_type2_bayesian_df_2050 = geopandas.GeoDataFrame(
    road_results_ssp1_type2_bayesian_df_2050, 
    geometry=geopandas.points_from_xy(road_results_ssp1_type2_bayesian_df_2050.Lon, 
                                      road_results_ssp1_type2_bayesian_df_2050.Lat), 
    crs="EPSG:4326"
)

fig, ax = plt.subplots(figsize=(12,10))
europe.plot(color='lightgrey', edgecolor='black', ax=ax)

road_results_ssp1_type2_bayesian_df_2050.plot(
    ax=ax,
    column='road_type2',
    cmap = 'gist_earth',
    markersize = 150,
    marker='s',
    legend=True,
    vmin=0,
    vmax=6,
    legend_kwds = {
        'label': 'Road Density (km/km2)',
        'orientation':'vertical'}
    )

ax.set_xlim(europe_boundaries[0], europe_boundaries[2])
ax.set_ylim(europe_boundaries[1], europe_boundaries[3])
ax.set_title("Road Density Type 2, 2050 (SSP1) [log-transformed]", fontsize=14)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

plt.tight_layout()
plt.show()

ssp1_type2_diff = road_results_ssp1_type2_bayesian_df_2050[['road_type2']].reset_index(drop=True)-road_results_ssp1_type2_bayesian_df_2015[['road_type2']].reset_index(drop=True)

n_cells = len(ssp1_type2_diff)
ssp1_type2_increase = np.sum(ssp1_type2_diff['road_type2'] > 0.005)
ssp1_type2_decrease = np.sum(ssp1_type2_diff['road_type2'] < -0.005)
ssp1_type2_unchanged = np.sum(np.isclose(ssp1_type2_diff['road_type2'], 0, atol=0.005))

pct_increased = (int(ssp1_type2_increase)/ n_cells) * 100
print(f'increased: {pct_increased:.1f}%')
pct_decreased = (int(ssp1_type2_decrease) / n_cells) * 100
print(f'decreased: {pct_decreased:.1f}%')
pct_unchanged = (int(ssp1_type2_unchanged) / n_cells) * 100
print(f'unchanged: {pct_unchanged:.1f}%')

#### Type 3
mean_pred_type3, std_pred_type3 = road_model_type3_bayesian.predict(road_test_X_ssp1, return_std=True)

road_results_ssp1_type3_bayesian_df = pd.DataFrame([])
road_results_ssp1_type3_bayesian_df['road_type3'] = mean_pred_type3
road_results_ssp1_type3_bayesian_df['road_std'] = std_pred_type3
road_results_ssp1_type3_bayesian_df['Year'] = df_annual_gdp_ssp1['Year'].reset_index(drop=True)
road_results_ssp1_type3_bayesian_df['Lon'] = df_annual_gdp_ssp1['Lon'].reset_index(drop=True)
road_results_ssp1_type3_bayesian_df['Lat'] = df_annual_gdp_ssp1['Lat'].reset_index(drop=True)
road_results_ssp1_type3_bayesian_df['road_estimate_weighted'] = weighted_road_densities(road_results_ssp1_type3_bayesian_df, weight, 'road_type3')
road_results_ssp1_type3_bayesian_df['road_var'] = 0.18*(std_pred_type3 ** 2)
print(road_results_ssp1_type3_bayesian_df)

##### Plotting
road_results_ssp1_type3_bayesian_df_2015 = road_results_ssp1_type3_bayesian_df[road_results_ssp1_type3_bayesian_df['Year']==2015]

road_results_ssp1_type3_bayesian_df_2015 = geopandas.GeoDataFrame(
    road_results_ssp1_type3_bayesian_df_2015, 
    geometry=geopandas.points_from_xy(road_results_ssp1_type3_bayesian_df_2015.Lon, 
                                      road_results_ssp1_type3_bayesian_df_2015.Lat), 
    crs="EPSG:4326"
)

fig, ax = plt.subplots(figsize=(12,10))
europe.plot(color='lightgrey', edgecolor='black', ax=ax)

road_results_ssp1_type3_bayesian_df_2015.plot(
    ax=ax,
    column='road_type3',
    cmap = 'gist_earth',
    markersize = 150,
    marker='s',
    legend=True,
    vmin=0,
    vmax=7,
    legend_kwds = {
        'label': 'Road Density (km/km2)',
        'orientation':'vertical'}
    )

ax.set_xlim(europe_boundaries[0], europe_boundaries[2])
ax.set_ylim(europe_boundaries[1], europe_boundaries[3])
ax.set_title("Road Density Type 3, 2015 (SSP1) [log-transformed]", fontsize=14)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

plt.tight_layout()
plt.show()

road_results_ssp1_type3_bayesian_df_2050 = road_results_ssp1_type3_bayesian_df[road_results_ssp1_type3_bayesian_df['Year']==2050]

road_results_ssp1_type3_bayesian_df_2050 = geopandas.GeoDataFrame(
    road_results_ssp1_type3_bayesian_df_2050, 
    geometry=geopandas.points_from_xy(road_results_ssp1_type3_bayesian_df_2050.Lon, 
                                      road_results_ssp1_type3_bayesian_df_2050.Lat), 
    crs="EPSG:4326"
)

fig, ax = plt.subplots(figsize=(12,10))
europe.plot(color='lightgrey', edgecolor='black', ax=ax)

road_results_ssp1_type3_bayesian_df_2050.plot(
    ax=ax,
    column='road_type3',
    cmap = 'gist_earth',
    markersize = 150,
    marker='s',
    legend=True,
    vmin=0,
    vmax=7,
    legend_kwds = {
        'label': 'Road Density (km/km2)',
        'orientation':'vertical'}
    )

ax.set_xlim(europe_boundaries[0], europe_boundaries[2])
ax.set_ylim(europe_boundaries[1], europe_boundaries[3])
ax.set_title("Road Density Type 3, 2050 (SSP1) [log-transformed]", fontsize=14)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

plt.tight_layout()
plt.show()

ssp1_type3_diff = road_results_ssp1_type3_bayesian_df_2050[['road_type3']].reset_index(drop=True)-road_results_ssp1_type3_bayesian_df_2015[['road_type3']].reset_index(drop=True)

n_cells = len(ssp1_type3_diff)
ssp1_type3_increase = np.sum(ssp1_type3_diff['road_type3'] > 0.005)
ssp1_type3_decrease = np.sum(ssp1_type3_diff['road_type3'] < -0.005)
ssp1_type3_unchanged = np.sum(np.isclose(ssp1_type3_diff['road_type3'], 0, atol=0.005))

pct_increased = (int(ssp1_type3_increase)/ n_cells) * 100
print(f'increased: {pct_increased:.1f}%')
pct_decreased = (int(ssp1_type3_decrease) / n_cells) * 100
print(f'decreased: {pct_decreased:.1f}%')
pct_unchanged = (int(ssp1_type3_unchanged) / n_cells) * 100
print(f'unchanged: {pct_unchanged:.1f}%')

#### Type 4
mean_pred_type4, std_pred_type4 = road_model_type4_bayesian.predict(road_test_X_ssp1, return_std=True)

road_results_ssp1_type4_bayesian_df = pd.DataFrame([])
road_results_ssp1_type4_bayesian_df['road_type4'] = mean_pred_type4
road_results_ssp1_type4_bayesian_df['road_std'] = std_pred_type4
road_results_ssp1_type4_bayesian_df['Year'] = df_annual_gdp_ssp1['Year'].reset_index(drop=True)
road_results_ssp1_type4_bayesian_df['Lon'] = df_annual_gdp_ssp1['Lon'].reset_index(drop=True)
road_results_ssp1_type4_bayesian_df['Lat'] = df_annual_gdp_ssp1['Lat'].reset_index(drop=True)
road_results_ssp1_type4_bayesian_df['road_estimate_weighted'] = weighted_road_densities(road_results_ssp1_type4_bayesian_df, weight, 'road_type4')
road_results_ssp1_type4_bayesian_df['road_var'] = 0.12*(std_pred_type4 ** 2)
print(road_results_ssp1_type1_bayesian_df)

##### Plotting
road_results_ssp1_type4_bayesian_df_2015= road_results_ssp1_type4_bayesian_df[road_results_ssp1_type3_bayesian_df['Year']==2015]

road_results_ssp1_type4_bayesian_df_2015 = geopandas.GeoDataFrame(
    road_results_ssp1_type4_bayesian_df_2015, 
    geometry=geopandas.points_from_xy(road_results_ssp1_type4_bayesian_df_2015.Lon, 
                                      road_results_ssp1_type4_bayesian_df_2015.Lat), 
    crs="EPSG:4326"
)

fig, ax = plt.subplots(figsize=(12,10))
europe.plot(color='lightgrey', edgecolor='black', ax=ax)

road_results_ssp1_type4_bayesian_df_2015.plot(
    ax=ax,
    column='road_type4',
    cmap = 'gist_earth',
    markersize = 150,
    marker='s',
    legend=True,
    vmin=0,
    vmax=9,
    legend_kwds = {
        'label': 'Road Density (km/km2)',
        'orientation':'vertical'}
    )

ax.set_xlim(europe_boundaries[0], europe_boundaries[2])
ax.set_ylim(europe_boundaries[1], europe_boundaries[3])
ax.set_title("Road Density Type 4, 2015 (SSP1) [log-transformed]", fontsize=14)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

plt.tight_layout()
plt.show()

road_results_ssp1_type4_bayesian_df_2050 = road_results_ssp1_type4_bayesian_df[road_results_ssp1_type3_bayesian_df['Year']==2050]

road_results_ssp1_type4_bayesian_df_2050 = geopandas.GeoDataFrame(
    road_results_ssp1_type4_bayesian_df_2050, 
    geometry=geopandas.points_from_xy(road_results_ssp1_type4_bayesian_df_2050.Lon, 
                                      road_results_ssp1_type4_bayesian_df_2050.Lat), 
    crs="EPSG:4326"
)


fig, ax = plt.subplots(figsize=(12,10))
europe.plot(color='lightgrey', edgecolor='black', ax=ax)

road_results_ssp1_type4_bayesian_df_2050.plot(
    ax=ax,
    column='road_type4',
    cmap = 'gist_earth',
    markersize = 150,
    marker='s',
    legend=True,
    vmin=0,
    vmax=9,
    legend_kwds = {
        'label': 'Road Density (km/km2)',
        'orientation':'vertical'}
    )

ax.set_xlim(europe_boundaries[0], europe_boundaries[2])
ax.set_ylim(europe_boundaries[1], europe_boundaries[3])
ax.set_title("Road Density Type 4, 2050 (SSP1) [log-transformed]", fontsize=14)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

plt.tight_layout()
plt.show()

ssp1_type4_diff = road_results_ssp1_type4_bayesian_df_2050[['road_type4']].reset_index(drop=True)-road_results_ssp1_type4_bayesian_df_2015[['road_type4']].reset_index(drop=True)

n_cells = len(ssp1_type4_diff)
ssp1_type4_increase = np.sum(ssp1_type4_diff['road_type4'] > 0.005)
ssp1_type4_decrease = np.sum(ssp1_type4_diff['road_type4'] < -0.005)
ssp1_type4_unchanged = np.sum(np.isclose(ssp1_type4_diff['road_type4'], 0, atol=0.005))

pct_increased = (int(ssp1_type4_increase)/ n_cells) * 100
print(f'increased: {pct_increased:.1f}%')
pct_decreased = (int(ssp1_type4_decrease) / n_cells) * 100
print(f'decreased: {pct_decreased:.1f}%')
pct_unchanged = (int(ssp1_type4_unchanged) / n_cells) * 100
print(f'unchanged: {pct_unchanged:.1f}%')

#### Type 5
mean_pred_type5, std_pred_type5 = road_model_type5_bayesian.predict(road_test_X_ssp1, return_std=True)

road_results_ssp1_type5_bayesian_df = pd.DataFrame([])
road_results_ssp1_type5_bayesian_df['road_type5'] = mean_pred_type5
road_results_ssp1_type5_bayesian_df['road_std'] = std_pred_type5
road_results_ssp1_type5_bayesian_df['Year'] = df_annual_gdp_ssp1['Year'].reset_index(drop=True)
road_results_ssp1_type5_bayesian_df['Lon'] = df_annual_gdp_ssp1['Lon'].reset_index(drop=True)
road_results_ssp1_type5_bayesian_df['Lat'] = df_annual_gdp_ssp1['Lat'].reset_index(drop=True)
road_results_ssp1_type5_bayesian_df['road_estimate_weighted'] = weighted_road_densities(road_results_ssp1_type5_bayesian_df, 
                                                                                        weight, 
                                                                                        'road_type5')
road_results_ssp1_type5_bayesian_df['road_var'] = 0.10*(std_pred_type5 ** 2)
print(road_results_ssp1_type3_bayesian_df['road_var'])

##### Plotting
road_results_ssp1_type5_bayesian_df_2015 = road_results_ssp1_type5_bayesian_df[road_results_ssp1_type5_bayesian_df['Year']==2015]

road_results_ssp1_type5_bayesian_df_2015 = geopandas.GeoDataFrame(
    road_results_ssp1_type5_bayesian_df_2015, 
    geometry=geopandas.points_from_xy(road_results_ssp1_type5_bayesian_df_2015.Lon, 
                                      road_results_ssp1_type5_bayesian_df_2015.Lat), 
    crs="EPSG:4326"
)


fig, ax = plt.subplots(figsize=(12,10))
europe.plot(color='lightgrey', edgecolor='black', ax=ax)

road_results_ssp1_type5_bayesian_df_2015.plot(
    ax=ax,
    column='road_type5',
    cmap = 'gist_earth',
    markersize = 150,
    marker='s',
    legend=True,
    vmin=0,
    vmax=10,
    legend_kwds = {
        'label': 'Road Density (km/km2)',
        'orientation':'vertical'}
    )

ax.set_xlim(europe_boundaries[0], europe_boundaries[2])
ax.set_ylim(europe_boundaries[1], europe_boundaries[3])
ax.set_title("Road Density Type 5, 2015 (SSP1) [log-transformed]", fontsize=14)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

plt.tight_layout()
plt.show()


road_results_ssp1_type5_bayesian_df_2050 = road_results_ssp1_type5_bayesian_df[road_results_ssp1_type5_bayesian_df['Year']==2050]

road_results_ssp1_type5_bayesian_df_2050 = geopandas.GeoDataFrame(
    road_results_ssp1_type5_bayesian_df_2050, 
    geometry=geopandas.points_from_xy(road_results_ssp1_type5_bayesian_df_2050.Lon, 
                                      road_results_ssp1_type5_bayesian_df_2050.Lat), 
    crs="EPSG:4326"
)

fig, ax = plt.subplots(figsize=(12,10))
europe.plot(color='lightgrey', edgecolor='black', ax=ax)

road_results_ssp1_type5_bayesian_df_2050.plot(
    ax=ax,
    column='road_type5',
    cmap = 'gist_earth',
    markersize = 150,
    marker='s',
    legend=True,
    vmin=0,
    vmax=10,
    legend_kwds = {
        'label': 'Road Density (km/km2)',
        'orientation':'vertical'}
    )

ax.set_xlim(europe_boundaries[0], europe_boundaries[2])
ax.set_ylim(europe_boundaries[1], europe_boundaries[3])
ax.set_title("Road Density Type 5, 2050 (SSP1) [log-transformed]", fontsize=14)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

plt.tight_layout()
plt.show()

ssp1_type5_diff = road_results_ssp1_type5_bayesian_df_2050[['road_type5']].reset_index(drop=True)-road_results_ssp1_type5_bayesian_df_2015[['road_type5']].reset_index(drop=True)

n_cells = len(ssp1_type5_diff)
ssp1_type5_increase = np.sum(ssp1_type5_diff['road_type5'] > 0.005)
ssp1_type5_decrease = np.sum(ssp1_type5_diff['road_type5'] < -0.005)
ssp1_type5_unchanged = np.sum(np.isclose(ssp1_type5_diff['road_type5'], 0, atol=0.005))

pct_increased = (int(ssp1_type5_increase)/ n_cells) * 100
print(f'increased: {pct_increased:.1f}%')
pct_decreased = (int(ssp1_type5_decrease) / n_cells) * 100
print(f'decreased: {pct_decreased:.1f}%')
pct_unchanged = (int(ssp1_type5_unchanged) / n_cells) * 100
print(f'unchanged: {pct_unchanged:.1f}%')

### Total Road
total_road_results_ssp1_bayesian = pd.DataFrame(road_results_ssp1_type1_bayesian_df['road_estimate_weighted']+road_results_ssp1_type2_bayesian_df['road_estimate_weighted']+road_results_ssp1_type3_bayesian_df['road_estimate_weighted']+road_results_ssp1_type4_bayesian_df['road_estimate_weighted']+road_results_ssp1_type5_bayesian_df['road_estimate_weighted'])
total_road_results_ssp1_bayesian['road_var'] = road_results_ssp1_type1_bayesian_df['road_var']+road_results_ssp1_type2_bayesian_df['road_var']+road_results_ssp1_type3_bayesian_df['road_var']+road_results_ssp1_type4_bayesian_df['road_var']+road_results_ssp1_type5_bayesian_df['road_var']
print(total_road_results_ssp1_bayesian)

## SSP3
### GDP
df_annual_gdp_ssp3 = df_annual_gdp_ssp3[(df_annual_gdp_ssp3['Year']<=2050) &
                                        (df_annual_gdp_ssp3['Year']>=2015)]

df_annual_gdp_ssp3_test = df_annual_gdp_ssp3['gdp_scaled'].reset_index(drop=True)
print(df_annual_gdp_ssp3_test)

### Population
pop_ssp3_hist_df
pop_ssp3_hist_df = pop_ssp3_hist_df.sort_values(by=['Year','Lon', 'Lat'])
pop_ssp3_hist_df_test = pop_ssp3_hist_df['population_density_scaled'].reset_index(drop=True)
print(pop_ssp3_hist_df_test)

### Test
dataframe_test_ssp3 = [oecd_test, land_area_test['land_area_scaled'], 
                       df_annual_gdp_ssp3_test, pop_ssp3_hist_df_test]
road_test_X_ssp3 = pd.concat(dataframe_test_ssp3, axis = 1)
print(road_test_X_ssp3)

### BayesianRidge
#### Type 1
mean_pred_type1, std_pred_type1 = road_model_type1_bayesian.predict(road_test_X_ssp3, 
                                                                    return_std=True)
road_results_ssp3_type1_bayesian_df = pd.DataFrame([])
road_results_ssp3_type1_bayesian_df['road_type1'] = mean_pred_type1
road_results_ssp3_type1_bayesian_df['road_std'] = std_pred_type1
road_results_ssp3_type1_bayesian_df['Year'] = df_annual_gdp_ssp3['Year'].reset_index(drop=True)
road_results_ssp3_type1_bayesian_df['Lon'] = df_annual_gdp_ssp3['Lon'].reset_index(drop=True)
road_results_ssp3_type1_bayesian_df['Lat'] = df_annual_gdp_ssp3['Lat'].reset_index(drop=True)
road_results_ssp3_type1_bayesian_df['road_estimate_weighted'] = weighted_road_densities(road_results_ssp3_type1_bayesian_df, 
                                                                                        weight, 
                                                                                        'road_type1')
road_results_ssp3_type1_bayesian_df['road_var'] = 0.35*(std_pred_type1 ** 2)
print(road_results_ssp3_type1_bayesian_df['road_type1'].max())

##### Plotting
road_results_ssp3_type1_bayesian_df_2015 = road_results_ssp3_type1_bayesian_df[road_results_ssp3_type1_bayesian_df['Year']==2015]

road_results_ssp3_type1_bayesian_df_2015 = geopandas.GeoDataFrame(
    road_results_ssp3_type1_bayesian_df_2015, 
    geometry=geopandas.points_from_xy(road_results_ssp3_type1_bayesian_df_2015.Lon, 
                                      road_results_ssp3_type1_bayesian_df_2015.Lat), 
    crs="EPSG:4326"
)


fig, ax = plt.subplots(figsize=(12,10))
europe.plot(color='lightgrey', edgecolor='black', ax=ax)

road_results_ssp3_type1_bayesian_df_2015.plot(
    ax=ax,
    column='road_type1',
    cmap = 'gist_earth',
    markersize = 150,
    marker='s',
    legend=True,
    vmin=0,
    vmax=12,
    legend_kwds = {
        'label': 'Road Density (km/km2)',
        'orientation':'vertical'}
    )

ax.set_xlim(europe_boundaries[0], europe_boundaries[2])
ax.set_ylim(europe_boundaries[1], europe_boundaries[3])
ax.set_title("Road Density Type 1, 2015 (SSP3) [log-transformed]", fontsize=14)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

plt.tight_layout()
plt.show()



road_results_ssp3_type1_bayesian_df_2050 = road_results_ssp3_type1_bayesian_df[road_results_ssp3_type1_bayesian_df['Year']==2050]
road_results_ssp3_type1_bayesian_df_2050['road_type1'].max()
road_results_ssp3_type1_bayesian_df_2050 = geopandas.GeoDataFrame(
    road_results_ssp3_type1_bayesian_df_2050, 
    geometry=geopandas.points_from_xy(road_results_ssp3_type1_bayesian_df_2050.Lon, 
                                      road_results_ssp3_type1_bayesian_df_2050.Lat), 
    crs="EPSG:4326"
)

fig, ax = plt.subplots(figsize=(12,10))
europe.plot(color='lightgrey', edgecolor='black', ax=ax)

road_results_ssp3_type1_bayesian_df_2050.plot(
    ax=ax,
    column='road_type1',
    cmap = 'gist_earth',
    markersize = 150,
    marker='s',
    legend=True,
    vmin=0,
    vmax=12,
    legend_kwds = {
        'label': 'Road Density (km/km2)',
        'orientation':'vertical'}
    )

ax.set_xlim(europe_boundaries[0], europe_boundaries[2])
ax.set_ylim(europe_boundaries[1], europe_boundaries[3])
ax.set_title("Road Density Type 1, 2050 (SSP3) [log-transformed]", fontsize=14)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

plt.tight_layout()
plt.show()

ssp3_type1_diff = road_results_ssp3_type1_bayesian_df_2050[['road_type1']].reset_index(drop=True)-road_results_ssp3_type1_bayesian_df_2015[['road_type1']].reset_index(drop=True)

n_cells = len(ssp3_type1_diff)
ssp3_type1_increase = np.sum(ssp3_type1_diff['road_type1'] > 0.005)
ssp3_type1_decrease = np.sum(ssp3_type1_diff['road_type1'] < -0.005)
ssp3_type1_unchanged = np.sum(np.isclose(ssp3_type1_diff['road_type1'], 0, atol=0.005))

pct_increased = (int(ssp3_type1_increase)/ n_cells) * 100
print(f'increased: {pct_increased:.1f}%')
pct_decreased = (int(ssp3_type1_decrease) / n_cells) * 100
print(f'decreased: {pct_decreased:.1f}%')
pct_unchanged = (int(ssp3_type1_unchanged) / n_cells) * 100
print(f'unchanged: {pct_unchanged:.1f}%')

#### Type 2
mean_pred_type2, std_pred_type2 = road_model_type2_bayesian.predict(road_test_X_ssp3, 
                                                                    return_std=True)
road_results_ssp3_type2_bayesian_df = pd.DataFrame([])
road_results_ssp3_type2_bayesian_df['road_type2'] = mean_pred_type2
road_results_ssp3_type2_bayesian_df['road_std'] = std_pred_type2
road_results_ssp3_type2_bayesian_df['Year'] = df_annual_gdp_ssp3['Year'].reset_index(drop=True)
road_results_ssp3_type2_bayesian_df['Lon'] = df_annual_gdp_ssp3['Lon'].reset_index(drop=True)
road_results_ssp3_type2_bayesian_df['Lat'] = df_annual_gdp_ssp3['Lat'].reset_index(drop=True)
road_results_ssp3_type2_bayesian_df['road_estimate_weighted'] = weighted_road_densities(road_results_ssp3_type2_bayesian_df, 
                                                                                        weight, 
                                                                                        'road_type2')
road_results_ssp3_type2_bayesian_df['road_var'] = 0.25*(std_pred_type2 ** 2)
print(road_results_ssp3_type2_bayesian_df)

##### Plotting
road_results_ssp3_type2_bayesian_df_2015 = road_results_ssp3_type2_bayesian_df[road_results_ssp3_type2_bayesian_df['Year']==2015]

road_results_ssp3_type2_bayesian_df_2015 = geopandas.GeoDataFrame(
    road_results_ssp3_type2_bayesian_df_2015, 
    geometry=geopandas.points_from_xy(road_results_ssp3_type2_bayesian_df_2015.Lon, 
                                      road_results_ssp3_type2_bayesian_df_2015.Lat), 
    crs="EPSG:4326"
)


fig, ax = plt.subplots(figsize=(12,10))
europe.plot(color='lightgrey', edgecolor='black', ax=ax)

road_results_ssp3_type2_bayesian_df_2015.plot(
    ax=ax,
    column='road_type2',
    cmap = 'gist_earth',
    markersize = 150,
    marker='s',
    legend=True,
    vmin=0,
    vmax=6,
    legend_kwds = {
        'label': 'Road Density (km/km2)',
        'orientation':'vertical'}
    )

ax.set_xlim(europe_boundaries[0], europe_boundaries[2])
ax.set_ylim(europe_boundaries[1], europe_boundaries[3])
ax.set_title("Road Density Type 2, 2015 (SSP3) [log-transformed]", fontsize=14)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

plt.tight_layout()
plt.show()

road_results_ssp3_type2_bayesian_df_2050 = road_results_ssp3_type2_bayesian_df[road_results_ssp3_type2_bayesian_df['Year']==2050]

road_results_ssp3_type2_bayesian_df_2050 = geopandas.GeoDataFrame(
    road_results_ssp3_type2_bayesian_df_2050, 
    geometry=geopandas.points_from_xy(road_results_ssp3_type2_bayesian_df_2050.Lon, 
                                      road_results_ssp3_type2_bayesian_df_2050.Lat), 
    crs="EPSG:4326"
)

fig, ax = plt.subplots(figsize=(12,10))
europe.plot(color='lightgrey', edgecolor='black', ax=ax)

road_results_ssp3_type2_bayesian_df_2050.plot(
    ax=ax,
    column='road_type2',
    cmap = 'gist_earth',
    markersize = 150,
    marker='s',
    legend=True,
    vmin=0,
    vmax=6,
    legend_kwds = {
        'label': 'Road Density (km/km2)',
        'orientation':'vertical'}
    )

ax.set_xlim(europe_boundaries[0], europe_boundaries[2])
ax.set_ylim(europe_boundaries[1], europe_boundaries[3])
ax.set_title("Road Density Type 2, 2050 (SSP3) [log-transformed]", fontsize=14)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

plt.tight_layout()
plt.show()

ssp3_type2_diff = road_results_ssp3_type2_bayesian_df_2050[['road_type2']].reset_index(drop=True)-road_results_ssp3_type2_bayesian_df_2015[['road_type2']].reset_index(drop=True)

n_cells = len(ssp3_type2_diff)
ssp3_type2_increase = np.sum(ssp3_type2_diff['road_type2'] > 0.005)
ssp3_type2_decrease = np.sum(ssp3_type2_diff['road_type2'] < -0.005)
ssp3_type2_unchanged = np.sum(np.isclose(ssp3_type2_diff['road_type2'], 0, atol=0.005))

pct_increased = (int(ssp3_type2_increase)/ n_cells) * 100
print(f'increased: {pct_increased:.1f}%')
pct_decreased = (int(ssp3_type2_decrease) / n_cells) * 100
print(f'decreased: {pct_decreased:.1f}%')
pct_unchanged = (int(ssp3_type2_unchanged) / n_cells) * 100
print(f'unchanged: {pct_unchanged:.1f}%')

#### Type 3
mean_pred_type3, std_pred_type3 = road_model_type3_bayesian.predict(road_test_X_ssp3, 
                                                                    return_std=True)
road_results_ssp3_type3_bayesian_df = pd.DataFrame([])
road_results_ssp3_type3_bayesian_df['road_type3'] = mean_pred_type3
road_results_ssp3_type3_bayesian_df['road_std'] = std_pred_type3
road_results_ssp3_type3_bayesian_df['Year'] = df_annual_gdp_ssp3['Year'].reset_index(drop=True)
road_results_ssp3_type3_bayesian_df['Lon'] = df_annual_gdp_ssp3['Lon'].reset_index(drop=True)
road_results_ssp3_type3_bayesian_df['Lat'] = df_annual_gdp_ssp3['Lat'].reset_index(drop=True)
road_results_ssp3_type3_bayesian_df['road_estimate_weighted'] = weighted_road_densities(road_results_ssp3_type3_bayesian_df,
                                                                                        weight,
                                                                                        'road_type3')
road_results_ssp3_type3_bayesian_df['road_var'] = 0.18*(std_pred_type3 ** 2)
print(road_results_ssp3_type3_bayesian_df)

##### Plotting
road_results_ssp3_type3_bayesian_df_2015 = road_results_ssp3_type3_bayesian_df[road_results_ssp3_type3_bayesian_df['Year']==2015]

road_results_ssp3_type3_bayesian_df_2015 = geopandas.GeoDataFrame(
    road_results_ssp3_type3_bayesian_df_2015, 
    geometry=geopandas.points_from_xy(road_results_ssp3_type3_bayesian_df_2015.Lon, 
                                      road_results_ssp3_type3_bayesian_df_2015.Lat), 
    crs="EPSG:4326"
)

fig, ax = plt.subplots(figsize=(12,10))
europe.plot(color='lightgrey', edgecolor='black', ax=ax)

road_results_ssp3_type3_bayesian_df_2015.plot(
    ax=ax,
    column='road_type3',
    cmap = 'gist_earth',
    markersize = 150,
    marker='s',
    legend=True,
    vmin=0,
    vmax=7,
    legend_kwds = {
        'label': 'Road Density (km/km2)',
        'orientation':'vertical'}
    )

ax.set_xlim(europe_boundaries[0], europe_boundaries[2])
ax.set_ylim(europe_boundaries[1], europe_boundaries[3])
ax.set_title("Road Density Type 3, 2015 (SSP3) [log-transformed]", fontsize=14)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

plt.tight_layout()
plt.show()

road_results_ssp3_type3_bayesian_df_2050 = road_results_ssp3_type3_bayesian_df[road_results_ssp3_type3_bayesian_df['Year']==2050]

road_results_ssp3_type3_bayesian_df_2050 = geopandas.GeoDataFrame(
    road_results_ssp3_type3_bayesian_df_2050, 
    geometry=geopandas.points_from_xy(road_results_ssp3_type3_bayesian_df_2050.Lon, 
                                      road_results_ssp3_type3_bayesian_df_2050.Lat), 
    crs="EPSG:4326"
)

fig, ax = plt.subplots(figsize=(12,10))
europe.plot(color='lightgrey', edgecolor='black', ax=ax)

road_results_ssp3_type3_bayesian_df_2050.plot(
    ax=ax,
    column='road_type3',
    cmap = 'gist_earth',
    markersize = 150,
    marker='s',
    legend=True,
    vmin=0,
    vmax=7,
    legend_kwds = {
        'label': 'Road Density (km/km2)',
        'orientation':'vertical'}
    )

ax.set_xlim(europe_boundaries[0], europe_boundaries[2])
ax.set_ylim(europe_boundaries[1], europe_boundaries[3])
ax.set_title("Road Density Type 3, 2050 (SSP3) [log-transformed]", fontsize=14)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

plt.tight_layout()
plt.show()

ssp3_type3_diff = road_results_ssp3_type3_bayesian_df_2050[['road_type3']].reset_index(drop=True)-road_results_ssp3_type3_bayesian_df_2015[['road_type3']].reset_index(drop=True)

n_cells = len(ssp3_type3_diff)
ssp3_type3_increase = np.sum(ssp3_type3_diff['road_type3'] > 0.005)
ssp3_type3_decrease = np.sum(ssp3_type3_diff['road_type3'] < -0.005)
ssp3_type3_unchanged = np.sum(np.isclose(ssp3_type3_diff['road_type3'], 0, atol=0.005))

pct_increased = (int(ssp3_type3_increase)/ n_cells) * 100
print(f'increased: {pct_increased:.1f}%')
pct_decreased = (int(ssp3_type3_decrease) / n_cells) * 100
print(f'decreased: {pct_decreased:.1f}%')
pct_unchanged = (int(ssp3_type3_unchanged) / n_cells) * 100
print(f'unchanged: {pct_unchanged:.1f}%')

#### Type 4
mean_pred_type4, std_pred_type4 = road_model_type4_bayesian.predict(road_test_X_ssp3, 
                                                                    return_std=True)
road_results_ssp3_type4_bayesian_df = pd.DataFrame([])
road_results_ssp3_type4_bayesian_df['road_type4'] = mean_pred_type4
road_results_ssp3_type4_bayesian_df['road_std'] = std_pred_type4
road_results_ssp3_type4_bayesian_df['Year'] = df_annual_gdp_ssp3['Year'].reset_index(drop=True)
road_results_ssp3_type4_bayesian_df['Lon'] = df_annual_gdp_ssp3['Lon'].reset_index(drop=True)
road_results_ssp3_type4_bayesian_df['Lat'] = df_annual_gdp_ssp3['Lat'].reset_index(drop=True)
road_results_ssp3_type4_bayesian_df['road_estimate_weighted'] = weighted_road_densities(road_results_ssp3_type4_bayesian_df, 
                                                                                        weight, 
                                                                                        'road_type4')
road_results_ssp3_type4_bayesian_df['road_var'] = 0.12*(std_pred_type4 ** 2)
print(road_results_ssp3_type4_bayesian_df)

##### Plotting
road_results_ssp3_type4_bayesian_df_2015= road_results_ssp3_type4_bayesian_df[road_results_ssp3_type3_bayesian_df['Year']==2015]

road_results_ssp3_type4_bayesian_df_2015 = geopandas.GeoDataFrame(
    road_results_ssp3_type4_bayesian_df_2015, 
    geometry=geopandas.points_from_xy(road_results_ssp3_type4_bayesian_df_2015.Lon, 
                                      road_results_ssp3_type4_bayesian_df_2015.Lat), 
    crs="EPSG:4326"
)

fig, ax = plt.subplots(figsize=(12,10))
europe.plot(color='lightgrey', edgecolor='black', ax=ax)

road_results_ssp3_type4_bayesian_df_2015.plot(
    ax=ax,
    column='road_type4',
    cmap = 'gist_earth',
    markersize = 150,
    marker='s',
    legend=True,
    vmin=0,
    vmax=9,
    legend_kwds = {
        'label': 'Road Density (km/km2)',
        'orientation':'vertical'}
    )

ax.set_xlim(europe_boundaries[0], europe_boundaries[2])
ax.set_ylim(europe_boundaries[1], europe_boundaries[3])
ax.set_title("Road Density Type 4, 2015 (SSP3) [log-transformed]", fontsize=14)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

plt.tight_layout()
plt.show()


road_results_ssp3_type4_bayesian_df_2050 = road_results_ssp3_type4_bayesian_df[road_results_ssp3_type3_bayesian_df['Year']==2050]

road_results_ssp3_type4_bayesian_df_2050 = geopandas.GeoDataFrame(
    road_results_ssp3_type4_bayesian_df_2050, 
    geometry=geopandas.points_from_xy(road_results_ssp3_type4_bayesian_df_2050.Lon, 
                                      road_results_ssp3_type4_bayesian_df_2050.Lat), 
    crs="EPSG:4326"
)

fig, ax = plt.subplots(figsize=(12,10))
europe.plot(color='lightgrey', edgecolor='black', ax=ax)

road_results_ssp3_type4_bayesian_df_2050.plot(
    ax=ax,
    column='road_type4',
    cmap = 'gist_earth',
    markersize = 150,
    marker='s',
    legend=True,
    vmin=0,
    vmax=9,
    legend_kwds = {
        'label': 'Road Density (km/km2)',
        'orientation':'vertical'}
    )

ax.set_xlim(europe_boundaries[0], europe_boundaries[2])
ax.set_ylim(europe_boundaries[1], europe_boundaries[3])
ax.set_title("Road Density Type 4, 2050 (SSP3) [log-transformed]", fontsize=14)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

plt.tight_layout()
plt.show()

ssp3_type4_diff = road_results_ssp3_type4_bayesian_df_2050[['road_type4']].reset_index(drop=True)-road_results_ssp3_type4_bayesian_df_2015[['road_type4']].reset_index(drop=True)

n_cells = len(ssp3_type4_diff)
ssp3_type4_increase = np.sum(ssp3_type4_diff['road_type4'] > 0.005)
ssp3_type4_decrease = np.sum(ssp3_type4_diff['road_type4'] < -0.005)
ssp3_type4_unchanged = np.sum(np.isclose(ssp3_type4_diff['road_type4'], 0, atol=0.005))

pct_increased = (int(ssp3_type4_increase)/ n_cells) * 100
print(f'increased: {pct_increased:.1f}%')
pct_decreased = (int(ssp3_type4_decrease) / n_cells) * 100
print(f'decreased: {pct_decreased:.1f}%')
pct_unchanged = (int(ssp3_type4_unchanged) / n_cells) * 100
print(f'unchanged: {pct_unchanged:.1f}%')

#### Type 5
mean_pred_type5, std_pred_type5 = road_model_type5_bayesian.predict(road_test_X_ssp3, 
                                                                    return_std=True)

road_results_ssp3_type5_bayesian_df = pd.DataFrame([])
road_results_ssp3_type5_bayesian_df['road_type5'] = mean_pred_type5
road_results_ssp3_type5_bayesian_df['road_std'] = std_pred_type5
road_results_ssp3_type5_bayesian_df['Year'] = df_annual_gdp_ssp3['Year'].reset_index(drop=True)
road_results_ssp3_type5_bayesian_df['Lon'] = df_annual_gdp_ssp3['Lon'].reset_index(drop=True)
road_results_ssp3_type5_bayesian_df['Lat'] = df_annual_gdp_ssp3['Lat'].reset_index(drop=True)
road_results_ssp3_type5_bayesian_df['road_estimate_weighted'] = weighted_road_densities(road_results_ssp3_type5_bayesian_df, 
                                                                                        weight, 
                                                                                        'road_type5')
road_results_ssp3_type5_bayesian_df['road_var'] = 0.10*(std_pred_type5 ** 2)

##### Plotting
road_results_ssp3_type5_bayesian_df_2015 = road_results_ssp3_type5_bayesian_df[road_results_ssp3_type5_bayesian_df['Year']==2015]

road_results_ssp3_type5_bayesian_df_2015 = geopandas.GeoDataFrame(
    road_results_ssp3_type5_bayesian_df_2015, 
    geometry=geopandas.points_from_xy(road_results_ssp3_type5_bayesian_df_2015.Lon, 
                                      road_results_ssp3_type5_bayesian_df_2015.Lat), 
    crs="EPSG:4326"
)

fig, ax = plt.subplots(figsize=(12,10))
europe.plot(color='lightgrey', edgecolor='black', ax=ax)

road_results_ssp3_type5_bayesian_df_2015.plot(
    ax=ax,
    column='road_type5',
    cmap = 'gist_earth',
    markersize = 150,
    marker='s',
    legend=True,
    vmin=0,
    vmax=10,
    legend_kwds = {
        'label': 'Road Density (km/km2)',
        'orientation':'vertical'}
    )

ax.set_xlim(europe_boundaries[0], europe_boundaries[2])
ax.set_ylim(europe_boundaries[1], europe_boundaries[3])
ax.set_title("Road Density Type 5, 2015 (SSP3) [log-transformed]", fontsize=14)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

plt.tight_layout()
plt.show()


road_results_ssp3_type5_bayesian_df_2050 = road_results_ssp3_type5_bayesian_df[road_results_ssp3_type5_bayesian_df['Year']==2050]

road_results_ssp3_type5_bayesian_df_2050 = geopandas.GeoDataFrame(
    road_results_ssp3_type5_bayesian_df_2050, 
    geometry=geopandas.points_from_xy(road_results_ssp3_type5_bayesian_df_2050.Lon, 
                                      road_results_ssp3_type5_bayesian_df_2050.Lat), 
    crs="EPSG:4326"
)

fig, ax = plt.subplots(figsize=(12,10))
europe.plot(color='lightgrey', edgecolor='black', ax=ax)

road_results_ssp3_type5_bayesian_df_2050.plot(
    ax=ax,
    column='road_type5',
    cmap = 'gist_earth',
    markersize = 150,
    marker='s',
    legend=True,
    vmin=0,
    vmax=10,
    legend_kwds = {
        'label': 'Road Density (km/km2)',
        'orientation':'vertical'}
    )

ax.set_xlim(europe_boundaries[0], europe_boundaries[2])
ax.set_ylim(europe_boundaries[1], europe_boundaries[3])
ax.set_title("Road Density Type 5, 2050 (SSP3) [log-transformed]", fontsize=14)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

plt.tight_layout()
plt.show()

ssp3_type5_diff = road_results_ssp3_type5_bayesian_df_2050[['road_type5']].reset_index(drop=True)-road_results_ssp3_type5_bayesian_df_2015[['road_type5']].reset_index(drop=True)

n_cells = len(ssp3_type5_diff)
ssp3_type5_increase = np.sum(ssp3_type5_diff['road_type5'] > 0.005)
ssp3_type5_decrease = np.sum(ssp3_type5_diff['road_type5'] < -0.005)
ssp3_type5_unchanged = np.sum(np.isclose(ssp3_type5_diff['road_type5'], 0, atol=0.005))

pct_increased = (int(ssp3_type5_increase)/ n_cells) * 100
print(f'increased: {pct_increased:.1f}%')
pct_decreased = (int(ssp3_type5_decrease) / n_cells) * 100
print(f'decreased: {pct_decreased:.1f}%')
pct_unchanged = (int(ssp3_type5_unchanged) / n_cells) * 100
print(f'unchanged: {pct_unchanged:.1f}%')


#### Total Road
total_road_results_ssp3_bayesian = pd.DataFrame(road_results_ssp3_type1_bayesian_df['road_estimate_weighted']+road_results_ssp3_type2_bayesian_df['road_estimate_weighted']+road_results_ssp3_type3_bayesian_df['road_estimate_weighted']+road_results_ssp3_type4_bayesian_df['road_estimate_weighted']+road_results_ssp3_type5_bayesian_df['road_estimate_weighted'])
total_road_results_ssp3_bayesian['road_var'] = road_results_ssp3_type1_bayesian_df['road_var']+road_results_ssp3_type2_bayesian_df['road_var']+road_results_ssp3_type3_bayesian_df['road_var']+road_results_ssp3_type4_bayesian_df['road_var']+road_results_ssp3_type5_bayesian_df['road_var']
print(total_road_results_ssp3_bayesian)





## SSP5
### GDP
df_annual_gdp_ssp5 = df_annual_gdp_ssp5[(df_annual_gdp_ssp5['Year']<=2050) &
                                        (df_annual_gdp_ssp5['Year']>=2015)]
df_annual_gdp_ssp5_test = df_annual_gdp_ssp5['gdp_scaled'].reset_index(drop=True)
print(df_annual_gdp_ssp5_test)

### Population
pop_ssp5_hist_df = pop_ssp5_hist_df.sort_values(by=['Year','Lon', 'Lat'])
pop_ssp5_hist_df_test = pop_ssp5_hist_df['population_density_scaled'].reset_index(drop=True)
print(pop_ssp5_hist_df_test)

### Test
dataframe_test_ssp5 = [oecd_test, land_area_test['land_area_scaled'], df_annual_gdp_ssp5_test, pop_ssp5_hist_df_test]
road_test_X_ssp5 = pd.concat(dataframe_test_ssp5, axis = 1)
road_test_X_ssp5 = road_test_X_ssp5.rename(columns={'total population density':'population_density'})
print(road_test_X_ssp5)

### BayesianRidge
#### Type 1
mean_pred_type1, std_pred_type1 = road_model_type1_bayesian.predict(road_test_X_ssp5, return_std=True)

road_results_ssp5_type1_bayesian_df = pd.DataFrame([])
road_results_ssp5_type1_bayesian_df['road_type1'] = mean_pred_type1
road_results_ssp5_type1_bayesian_df['road_std'] = std_pred_type1
road_results_ssp5_type1_bayesian_df['Year'] = df_annual_gdp_ssp5['Year'].reset_index(drop=True)
road_results_ssp5_type1_bayesian_df['Lon'] = df_annual_gdp_ssp5['Lon'].reset_index(drop=True)
road_results_ssp5_type1_bayesian_df['Lat'] = df_annual_gdp_ssp5['Lat'].reset_index(drop=True)
road_results_ssp5_type1_bayesian_df['road_estimate_weighted'] = weighted_road_densities(road_results_ssp5_type1_bayesian_df, 
                                                                                        weight, 'road_type1')
road_results_ssp5_type1_bayesian_df['road_var'] = 0.35*(std_pred_type1 ** 2)
print(road_results_ssp5_type1_bayesian_df)

##### Plotting
road_results_ssp5_type1_bayesian_df_2015 = road_results_ssp5_type1_bayesian_df[road_results_ssp5_type1_bayesian_df['Year']==2015]

road_results_ssp5_type1_bayesian_df_2015 = geopandas.GeoDataFrame(
    road_results_ssp5_type1_bayesian_df_2015, 
    geometry=geopandas.points_from_xy(road_results_ssp5_type1_bayesian_df_2015.Lon, 
                                      road_results_ssp5_type1_bayesian_df_2015.Lat), 
    crs="EPSG:4326"
)

fig, ax = plt.subplots(figsize=(12,10))
europe.plot(color='lightgrey', edgecolor='black', ax=ax)

road_results_ssp5_type1_bayesian_df_2015.plot(
    ax=ax,
    column='road_type1',
    cmap = 'gist_earth',
    markersize = 150,
    marker='s',
    legend=True,
    vmin=0,
    vmax=12,
    legend_kwds = {
        'label': 'Road Density (km/km2)',
        'orientation':'vertical'}
    )

ax.set_xlim(europe_boundaries[0], europe_boundaries[2])
ax.set_ylim(europe_boundaries[1], europe_boundaries[3])
ax.set_title("Road Density Type 1, 2015 (SSP5) [log-transformed]", fontsize=14)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

plt.tight_layout()
plt.show()

road_results_ssp5_type1_bayesian_df_2050 = road_results_ssp5_type1_bayesian_df[road_results_ssp5_type1_bayesian_df['Year']==2050]

road_results_ssp5_type1_bayesian_df_2050 = geopandas.GeoDataFrame(
    road_results_ssp5_type1_bayesian_df_2050, 
    geometry=geopandas.points_from_xy(road_results_ssp5_type1_bayesian_df_2050.Lon, 
                                      road_results_ssp5_type1_bayesian_df_2050.Lat), 
    crs="EPSG:4326"
)

fig, ax = plt.subplots(figsize=(12,10))
europe.plot(color='lightgrey', edgecolor='black', ax=ax)

road_results_ssp5_type1_bayesian_df_2050.plot(
    ax=ax,
    column='road_type1',
    cmap = 'gist_earth',
    markersize = 150,
    marker='s',
    legend=True,
    vmin=0,
    vmax=12,
    legend_kwds = {
        'label': 'Road Density (km/km2)',
        'orientation':'vertical'}
    )

ax.set_xlim(europe_boundaries[0], europe_boundaries[2])
ax.set_ylim(europe_boundaries[1], europe_boundaries[3])
ax.set_title("Road Density Type 1, 2050 (SSP5) [log-transformed]", fontsize=14)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

plt.tight_layout()
plt.show()

ssp5_type1_diff = road_results_ssp5_type1_bayesian_df_2050[['road_type1']].reset_index(drop=True)-road_results_ssp5_type1_bayesian_df_2015[['road_type1']].reset_index(drop=True)

n_cells = len(ssp5_type1_diff)
ssp5_type1_increase = np.sum(ssp5_type1_diff['road_type1'] > 0.005)
ssp5_type1_decrease = np.sum(ssp5_type1_diff['road_type1'] < -0.005)
ssp5_type1_unchanged = np.sum(np.isclose(ssp5_type1_diff['road_type1'], 0, atol=0.005))

pct_increased = (int(ssp5_type1_increase)/ n_cells) * 100
print(f'increased: {pct_increased:.1f}%')
pct_decreased = (int(ssp5_type1_decrease) / n_cells) * 100
print(f'decreased: {pct_decreased:.1f}%')
pct_unchanged = (int(ssp5_type1_unchanged) / n_cells) * 100
print(f'unchanged: {pct_unchanged:.1f}%')

#### Type 2
mean_pred_type2, std_pred_type2 = road_model_type2_bayesian.predict(road_test_X_ssp5, return_std=True)

road_results_ssp5_type2_bayesian_df = pd.DataFrame([])
road_results_ssp5_type2_bayesian_df['road_type2'] = mean_pred_type2
road_results_ssp5_type2_bayesian_df['road_std'] = std_pred_type2
road_results_ssp5_type2_bayesian_df['Year'] = df_annual_gdp_ssp5['Year'].reset_index(drop=True)
road_results_ssp5_type2_bayesian_df['Lon'] = df_annual_gdp_ssp5['Lon'].reset_index(drop=True)
road_results_ssp5_type2_bayesian_df['Lat'] = df_annual_gdp_ssp5['Lat'].reset_index(drop=True)
road_results_ssp5_type2_bayesian_df['road_estimate_weighted'] = weighted_road_densities(road_results_ssp5_type2_bayesian_df, 
                                                                                        weight, 'road_type2')
road_results_ssp5_type2_bayesian_df['road_var'] = 0.25*(std_pred_type2 ** 2)
print(road_results_ssp5_type2_bayesian_df)


##### Plotting
road_results_ssp5_type2_bayesian_df_2015 = road_results_ssp5_type2_bayesian_df[road_results_ssp5_type2_bayesian_df['Year']==2015]

road_results_ssp5_type2_bayesian_df_2015 = geopandas.GeoDataFrame(
    road_results_ssp5_type2_bayesian_df_2015, 
    geometry=geopandas.points_from_xy(road_results_ssp5_type2_bayesian_df_2015.Lon, 
                                      road_results_ssp5_type2_bayesian_df_2015.Lat), 
    crs="EPSG:4326"
)

fig, ax = plt.subplots(figsize=(12,10))
europe.plot(color='lightgrey', edgecolor='black', ax=ax)

road_results_ssp5_type2_bayesian_df_2015.plot(
    ax=ax,
    column='road_type2',
    cmap = 'gist_earth',
    markersize = 150,
    marker='s',
    legend=True,
    vmin=0,
    vmax=6,
    legend_kwds = {
        'label': 'Road Density (km/km2)',
        'orientation':'vertical'}
    )

ax.set_xlim(europe_boundaries[0], europe_boundaries[2])
ax.set_ylim(europe_boundaries[1], europe_boundaries[3])
ax.set_title("Road Density Type 2, 2015 (SSP5) [log-transformed]", fontsize=14)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

plt.tight_layout()
plt.show()

road_results_ssp5_type2_bayesian_df_2050 = road_results_ssp5_type2_bayesian_df[road_results_ssp5_type2_bayesian_df['Year']==2050]

road_results_ssp5_type2_bayesian_df_2050 = geopandas.GeoDataFrame(
    road_results_ssp5_type2_bayesian_df_2050, 
    geometry=geopandas.points_from_xy(road_results_ssp5_type2_bayesian_df_2050.Lon, 
                                      road_results_ssp5_type2_bayesian_df_2050.Lat), 
    crs="EPSG:4326"
)

fig, ax = plt.subplots(figsize=(12,10))
europe.plot(color='lightgrey', edgecolor='black', ax=ax)

road_results_ssp5_type2_bayesian_df_2050.plot(
    ax=ax,
    column='road_type2',
    cmap = 'gist_earth',
    markersize = 150,
    marker='s',
    legend=True,
    vmin=0,
    vmax=6,
    legend_kwds = {
        'label': 'Road Density (km/km2)',
        'orientation':'vertical'}
    )

ax.set_xlim(europe_boundaries[0], europe_boundaries[2])
ax.set_ylim(europe_boundaries[1], europe_boundaries[3])
ax.set_title("Road Density Type 2, 2050 (SSP5) [log-transformed]", fontsize=14)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

plt.tight_layout()
plt.show()

ssp5_type2_diff = road_results_ssp5_type2_bayesian_df_2050[['road_type2']].reset_index(drop=True)-road_results_ssp5_type2_bayesian_df_2015[['road_type2']].reset_index(drop=True)

n_cells = len(ssp5_type2_diff)
ssp5_type2_increase = np.sum(ssp5_type2_diff['road_type2'] > 0.005)
ssp5_type2_decrease = np.sum(ssp5_type2_diff['road_type2'] < -0.005)
ssp5_type2_unchanged = np.sum(np.isclose(ssp5_type2_diff['road_type2'], 0, atol=0.005))

pct_increased = (int(ssp5_type2_increase)/ n_cells) * 100
print(f'increased: {pct_increased:.1f}%')
pct_decreased = (int(ssp5_type2_decrease) / n_cells) * 100
print(f'decreased: {pct_decreased:.1f}%')
pct_unchanged = (int(ssp5_type2_unchanged) / n_cells) * 100
print(f'unchanged: {pct_unchanged:.1f}%')

#### Type 3
mean_pred_type3, std_pred_type3 = road_model_type3_bayesian.predict(road_test_X_ssp5, return_std=True)
road_results_ssp5_type3_bayesian_df = pd.DataFrame([])
road_results_ssp5_type3_bayesian_df['road_type3'] = mean_pred_type3
road_results_ssp5_type3_bayesian_df['road_std'] = std_pred_type3
road_results_ssp5_type3_bayesian_df['Year'] = df_annual_gdp_ssp5['Year'].reset_index(drop=True)
road_results_ssp5_type3_bayesian_df['Lon'] = df_annual_gdp_ssp5['Lon'].reset_index(drop=True)
road_results_ssp5_type3_bayesian_df['Lat'] = df_annual_gdp_ssp5['Lat'].reset_index(drop=True)
road_results_ssp5_type3_bayesian_df['road_estimate_weighted'] = weighted_road_densities(road_results_ssp5_type3_bayesian_df, 
                                                                                        weight, 
                                                                                        'road_type3')
road_results_ssp5_type3_bayesian_df['road_var'] = 0.18*(std_pred_type3 ** 2)
print(road_results_ssp5_type3_bayesian_df)


##### Plotting
road_results_ssp5_type3_bayesian_df_2015 = road_results_ssp5_type3_bayesian_df[road_results_ssp5_type3_bayesian_df['Year']==2015]

road_results_ssp5_type3_bayesian_df_2015 = geopandas.GeoDataFrame(
    road_results_ssp5_type3_bayesian_df_2015, 
    geometry=geopandas.points_from_xy(road_results_ssp5_type3_bayesian_df_2015.Lon, 
                                      road_results_ssp5_type3_bayesian_df_2015.Lat), 
    crs="EPSG:4326"
)

fig, ax = plt.subplots(figsize=(12,10))
europe.plot(color='lightgrey', edgecolor='black', ax=ax)

road_results_ssp5_type3_bayesian_df_2015.plot(
    ax=ax,
    column='road_type3',
    cmap = 'gist_earth',
    markersize = 150,
    marker='s',
    legend=True,
    vmin=0,
    vmax=7,
    legend_kwds = {
        'label': 'Road Density (km/km2)',
        'orientation':'vertical'}
    )

ax.set_xlim(europe_boundaries[0], europe_boundaries[2])
ax.set_ylim(europe_boundaries[1], europe_boundaries[3])
ax.set_title("Road Density Type 3, 2015 (SSP5) [log-transformed]", fontsize=14)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

plt.tight_layout()
plt.show()

road_results_ssp5_type3_bayesian_df_2050 = road_results_ssp5_type3_bayesian_df[road_results_ssp5_type3_bayesian_df['Year']==2050]

road_results_ssp5_type3_bayesian_df_2050 = geopandas.GeoDataFrame(
    road_results_ssp5_type3_bayesian_df_2050, 
    geometry=geopandas.points_from_xy(road_results_ssp5_type3_bayesian_df_2050.Lon, 
                                      road_results_ssp5_type3_bayesian_df_2050.Lat), 
    crs="EPSG:4326"
)

fig, ax = plt.subplots(figsize=(12,10))
europe.plot(color='lightgrey', edgecolor='black', ax=ax)

road_results_ssp5_type3_bayesian_df_2050.plot(
    ax=ax,
    column='road_type3',
    cmap = 'gist_earth',
    markersize = 150,
    marker='s',
    legend=True,
    vmin=0,
    vmax=7,
    legend_kwds = {
        'label': 'Road Density (km/km2)',
        'orientation':'vertical'}
    )

ax.set_xlim(europe_boundaries[0], europe_boundaries[2])
ax.set_ylim(europe_boundaries[1], europe_boundaries[3])
ax.set_title("Road Density Type 3, 2050 (SSP5) [log-transformed]", fontsize=14)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

plt.tight_layout()
plt.show()

ssp5_type3_diff = road_results_ssp5_type3_bayesian_df_2050[['road_type3']].reset_index(drop=True)-road_results_ssp5_type3_bayesian_df_2015[['road_type3']].reset_index(drop=True)

n_cells = len(ssp5_type3_diff)
ssp5_type3_increase = np.sum(ssp5_type3_diff['road_type3'] > 0.005)
ssp5_type3_decrease = np.sum(ssp5_type3_diff['road_type3'] < -0.005)
ssp5_type3_unchanged = np.sum(np.isclose(ssp5_type3_diff['road_type3'], 0, atol=0.005))

pct_increased = (int(ssp5_type3_increase)/ n_cells) * 100
print(f'increased: {pct_increased:.1f}%')
pct_decreased = (int(ssp5_type3_decrease) / n_cells) * 100
print(f'decreased: {pct_decreased:.1f}%')
pct_unchanged = (int(ssp5_type3_unchanged) / n_cells) * 100
print(f'unchanged: {pct_unchanged:.1f}%')


#### Type 4
mean_pred_type4, std_pred_type4 = road_model_type4_bayesian.predict(road_test_X_ssp5, return_std=True)
road_results_ssp5_type4_bayesian_df = pd.DataFrame([])
road_results_ssp5_type4_bayesian_df['road_type4'] = mean_pred_type4
road_results_ssp5_type4_bayesian_df['road_std'] = std_pred_type4
road_results_ssp5_type4_bayesian_df['Year'] = df_annual_gdp_ssp5['Year'].reset_index(drop=True)
road_results_ssp5_type4_bayesian_df['Lon'] = df_annual_gdp_ssp5['Lon'].reset_index(drop=True)
road_results_ssp5_type4_bayesian_df['Lat'] = df_annual_gdp_ssp5['Lat'].reset_index(drop=True)
road_results_ssp5_type4_bayesian_df['road_estimate_weighted'] = weighted_road_densities(road_results_ssp5_type4_bayesian_df, 
                                                                                        weight, 
                                                                                        'road_type4')
road_results_ssp5_type4_bayesian_df['road_var'] = 0.12*(std_pred_type4 ** 2)
print(road_results_ssp5_type4_bayesian_df)

##### Plotting
road_results_ssp5_type4_bayesian_df_2015= road_results_ssp5_type4_bayesian_df[road_results_ssp5_type3_bayesian_df['Year']==2015]

road_results_ssp5_type4_bayesian_df_2015 = geopandas.GeoDataFrame(
    road_results_ssp5_type4_bayesian_df_2015, 
    geometry=geopandas.points_from_xy(road_results_ssp5_type4_bayesian_df_2015.Lon, 
                                      road_results_ssp5_type4_bayesian_df_2015.Lat), 
    crs="EPSG:4326"
)


fig, ax = plt.subplots(figsize=(12,10))
europe.plot(color='lightgrey', edgecolor='black', ax=ax)

road_results_ssp5_type4_bayesian_df_2015.plot(
    ax=ax,
    column='road_type4',
    cmap = 'gist_earth',
    markersize = 150,
    marker='s',
    legend=True,
    vmin=0,
    vmax=9,
    legend_kwds = {
        'label': 'Road Density (km/km2)',
        'orientation':'vertical'}
    )

ax.set_xlim(europe_boundaries[0], europe_boundaries[2])
ax.set_ylim(europe_boundaries[1], europe_boundaries[3])
ax.set_title("Road Density Type 4, 2015 (SSP5) [log-transformed]", fontsize=14)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

plt.tight_layout()
plt.show()


road_results_ssp5_type4_bayesian_df_2050 = road_results_ssp5_type4_bayesian_df[road_results_ssp5_type3_bayesian_df['Year']==2050]

road_results_ssp5_type4_bayesian_df_2050 = geopandas.GeoDataFrame(
    road_results_ssp5_type4_bayesian_df_2050, 
    geometry=geopandas.points_from_xy(road_results_ssp5_type4_bayesian_df_2050.Lon, 
                                      road_results_ssp5_type4_bayesian_df_2050.Lat), 
    crs="EPSG:4326"
)


fig, ax = plt.subplots(figsize=(12,10))
europe.plot(color='lightgrey', edgecolor='black', ax=ax)

road_results_ssp5_type4_bayesian_df_2050.plot(
    ax=ax,
    column='road_type4',
    cmap = 'gist_earth',
    markersize = 150,
    marker='s',
    legend=True,
    vmin=0,
    vmax=9,
    legend_kwds = {
        'label': 'Road Density (km/km2)',
        'orientation':'vertical'}
    )

ax.set_xlim(europe_boundaries[0], europe_boundaries[2])
ax.set_ylim(europe_boundaries[1], europe_boundaries[3])
ax.set_title("Road Density Type 4, 2050 (SSP5) [log-transformed]", fontsize=14)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

plt.tight_layout()
plt.show()

ssp5_type4_diff = road_results_ssp5_type4_bayesian_df_2050[['road_type4']].reset_index(drop=True)-road_results_ssp5_type4_bayesian_df_2015[['road_type4']].reset_index(drop=True)

n_cells = len(ssp5_type4_diff)
ssp5_type4_increase = np.sum(ssp5_type4_diff['road_type4'] > 0.005)
ssp5_type4_decrease = np.sum(ssp5_type4_diff['road_type4'] < -0.005)
ssp5_type4_unchanged = np.sum(np.isclose(ssp5_type4_diff['road_type4'], 0, atol=0.005))

pct_increased = (int(ssp5_type4_increase)/ n_cells) * 100
print(f'increased: {pct_increased:.1f}%')
pct_decreased = (int(ssp5_type4_decrease) / n_cells) * 100
print(f'decreased: {pct_decreased:.1f}%')
pct_unchanged = (int(ssp5_type4_unchanged) / n_cells) * 100
print(f'unchanged: {pct_unchanged:.1f}%')


#### Type 5
mean_pred_type5, std_pred_type5 = road_model_type5_bayesian.predict(road_test_X_ssp5, return_std=True)
road_results_ssp5_type5_bayesian_df = pd.DataFrame([])
road_results_ssp5_type5_bayesian_df['road_type5'] = mean_pred_type5
road_results_ssp5_type5_bayesian_df['road_std'] = std_pred_type5
road_results_ssp5_type5_bayesian_df['Year'] = df_annual_gdp_ssp5['Year'].reset_index(drop=True)
road_results_ssp5_type5_bayesian_df['Lon'] = df_annual_gdp_ssp5['Lon'].reset_index(drop=True)
road_results_ssp5_type5_bayesian_df['Lat'] = df_annual_gdp_ssp5['Lat'].reset_index(drop=True)
road_results_ssp5_type5_bayesian_df['road_estimate_weighted'] = weighted_road_densities(road_results_ssp5_type5_bayesian_df, 
                                                                                        weight, 
                                                                                        'road_type5')
road_results_ssp5_type5_bayesian_df['road_var'] = 0.10*(std_pred_type5 ** 2)
print(road_results_ssp5_type5_bayesian_df)

##### Plotting
road_results_ssp5_type5_bayesian_df_2015 = road_results_ssp5_type5_bayesian_df[road_results_ssp5_type5_bayesian_df['Year']==2015]

road_results_ssp5_type5_bayesian_df_2015 = geopandas.GeoDataFrame(
    road_results_ssp5_type5_bayesian_df_2015, 
    geometry=geopandas.points_from_xy(road_results_ssp5_type5_bayesian_df_2015.Lon, 
                                      road_results_ssp5_type5_bayesian_df_2015.Lat), 
    crs="EPSG:4326"
)

fig, ax = plt.subplots(figsize=(12,10))
europe.plot(color='lightgrey', edgecolor='black', ax=ax)

road_results_ssp5_type5_bayesian_df_2015.plot(
    ax=ax,
    column='road_type5',
    cmap = 'gist_earth',
    markersize = 150,
    marker='s',
    legend=True,
    vmin=0,
    vmax=10,
    legend_kwds = {
        'label': 'Road Density (km/km2)',
        'orientation':'vertical'}
    )

ax.set_xlim(europe_boundaries[0], europe_boundaries[2])
ax.set_ylim(europe_boundaries[1], europe_boundaries[3])
ax.set_title("Road Density Type 5, 2015 (SSP5) [log-transformed]", fontsize=14)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

plt.tight_layout()
plt.show()

road_results_ssp5_type5_bayesian_df_2050 = road_results_ssp5_type5_bayesian_df[road_results_ssp5_type5_bayesian_df['Year']==2050]

road_results_ssp5_type5_bayesian_df_2050 = geopandas.GeoDataFrame(
    road_results_ssp5_type5_bayesian_df_2050, 
    geometry=geopandas.points_from_xy(road_results_ssp5_type5_bayesian_df_2050.Lon, 
                                      road_results_ssp5_type5_bayesian_df_2050.Lat), 
    crs="EPSG:4326"
)


fig, ax = plt.subplots(figsize=(12,10))
europe.plot(color='lightgrey', edgecolor='black', ax=ax)

road_results_ssp5_type5_bayesian_df_2050.plot(
    ax=ax,
    column='road_type5',
    cmap = 'gist_earth',
    markersize = 150,
    marker='s',
    legend=True,
    vmin=0,
    vmax=10,
    legend_kwds = {
        'label': 'Road Density (km/km2)',
        'orientation':'vertical'}
    )

ax.set_xlim(europe_boundaries[0], europe_boundaries[2])
ax.set_ylim(europe_boundaries[1], europe_boundaries[3])
ax.set_title("Road Density Type 5, 2050 (SSP5) [log-transformed]", fontsize=14)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

plt.tight_layout()
plt.show()

ssp5_type5_diff = pd.DataFrame(road_results_ssp5_type5_bayesian_df_2050[['road_type5']].reset_index(drop=True)-road_results_ssp5_type5_bayesian_df_2015[['road_type5']].reset_index(drop=True))

n_cells = len(ssp5_type5_diff)

ssp5_increase = np.sum(ssp5_type5_diff['road_type5'] > 0.005)
ssp5_decrease = np.sum(ssp5_type5_diff['road_type5'] < -0.005)
ssp5_unchanged = np.sum(np.isclose(ssp5_type5_diff['road_type5'], 0, atol=0.005))

pct_increased = (int(ssp5_increase)/ n_cells) * 100
print(f'increased: {pct_increased:.1f}%')
pct_decreased = (int(ssp5_decrease) / n_cells) * 100
print(f'decreased: {pct_decreased:.1f}%')
pct_unchanged = (int(ssp5_unchanged) / n_cells) * 100
print(f'unchanged: {pct_unchanged:.1f}%')


road_results_ssp5_type5_bayesian_df['road_type5'].hist(bins=100)

#### Total Road
total_road_results_ssp5_bayesian = pd.DataFrame(road_results_ssp5_type1_bayesian_df['road_estimate_weighted']+road_results_ssp5_type2_bayesian_df['road_estimate_weighted']+road_results_ssp5_type3_bayesian_df['road_estimate_weighted']+road_results_ssp5_type4_bayesian_df['road_estimate_weighted']+road_results_ssp5_type5_bayesian_df['road_estimate_weighted'])
total_road_results_ssp5_bayesian['road_var'] = road_results_ssp5_type1_bayesian_df['road_var']+road_results_ssp5_type2_bayesian_df['road_var']+road_results_ssp5_type3_bayesian_df['road_var']+road_results_ssp5_type4_bayesian_df['road_var']+road_results_ssp5_type5_bayesian_df['road_var']
print(total_road_results_ssp5_bayesian)

# The general model

def hfi(eqi, pop_density, ntl, road):
    """
    

    Parameters
    ----------
    eqi : TYPE
        DESCRIPTION.
    pop_density : TYPE
        DESCRIPTION.
    ntl : TYPE
        DESCRIPTION.
    road : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    return pd.DataFrame(((1-eqi)+pop_density + ntl + road), columns=['HFI'])



def eqi(lai, gpp, fvc):
    """

    Parameters
    ----------
    lai : TYPE
        DESCRIPTION.
    gpp : TYPE
        DESCRIPTION.
    fvc : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    return pd.DataFrame((lai.values+gpp.values+fvc.values)/3, columns=['EQI'])


## SSP1
### EQI
eqi_ssp1 = eqi(lai_ssp1_df['Total'], 
               gpp_ssp1_df['Total'], 
               fpc_ssp1_df['total_veg_cover'])

eqi_ssp1['Lon'] = lai_ssp1_df['Lon']
eqi_ssp1['Lat'] = lai_ssp1_df['Lat']
eqi_ssp1['Year'] = lai_ssp1_df['Year']

mask = eqi_ssp1.set_index(['Lon', 'Lat']).index.isin(
    fpc_ssp3_df.set_index([ 'Lon', 'Lat']).index
)

eqi_ssp1 = eqi_ssp1[mask]

eqi_ssp1_scaled = pd.DataFrame(scaler.fit_transform(eqi_ssp1[['EQI']]),
                               columns=['EQI'])

eqi_ssp1_scaled['Year'] = fpc_ssp3_df['Year']

### HFI
#### POP
pop_ssp1_hist_df_test = pop_ssp1_hist_df_test.reset_index(drop=True)
pop_ssp1_hist_df_test_scaled = pd.DataFrame(scaler.fit_transform(pop_ssp1_hist_df[['population_density']]),
                                            columns=['population_density'])

pop_ssp1_hist_df_test_scaled['Year'] = pop_ssp1_hist_df['Year']

#### NTL
df_annual_ntl_ssp1 = df_annual_ntl_ssp1.reset_index(drop=True)
df_annual_ntl_ssp1_scaled = pd.DataFrame(scaler.fit_transform(df_annual_ntl_ssp1[['ntl']]), 
                                         columns=['ntl'])

df_annual_ntl_ssp1_scaled_var = df_annual_ntl_ssp1[['var']]/(df_annual_ntl_ssp1['ntl'].max()-df_annual_ntl_ssp1['ntl'].min())**2
df_annual_ntl_ssp1_scaled['Year']= df_annual_ntl_ssp1['Year']

#### Road
total_road_results_ssp1_bayesian = total_road_results_ssp1_bayesian.reset_index(drop=True)
plt.hist(total_road_results_ssp1_bayesian['road_estimate_weighted'], bins=100)
total_road_results_ssp1_bayesian_scaled_var = total_road_results_ssp1_bayesian[['road_var']]/(total_road_results_ssp1_bayesian['road_estimate_weighted'].max()-total_road_results_ssp1_bayesian['road_estimate_weighted'].min())**2
total_road_results_ssp1_bayesian_scaled = pd.DataFrame(scaler.fit_transform(total_road_results_ssp1_bayesian[['road_estimate_weighted']]),
                                                       columns=['road'])
total_road_results_ssp1_bayesian_scaled['Year'] = df_annual_ntl_ssp1['Year']

#### HFI Calculation
hfi_ssp1_df = [eqi_ssp1_scaled['EQI'], 
               pop_ssp1_hist_df_test_scaled['population_density'],
               df_annual_ntl_ssp1_scaled['ntl'], 
               total_road_results_ssp1_bayesian_scaled['road']]
hfi_ssp1_df = pd.concat(hfi_ssp1_df, axis=1)
hfi_ssp1_df.corr()

hfi_ssp1 = hfi(eqi_ssp1_scaled['EQI'], 
               pop_ssp1_hist_df_test_scaled['population_density'],
               df_annual_ntl_ssp1_scaled['ntl'], 
               total_road_results_ssp1_bayesian_scaled['road'])

#### Var
ssp1_pd_r_covariance = np.cov(pop_ssp1_hist_df['population_density'], 
                              total_road_results_ssp1_bayesian['road_estimate_weighted'])[0,1]
ssp1_pd_r_covariance_scaled = ssp1_pd_r_covariance/((pop_ssp1_hist_df['population_density'].max()-pop_ssp1_hist_df['population_density'].min())*(total_road_results_ssp1_bayesian['road_estimate_weighted'].max()-total_road_results_ssp1_bayesian['road_estimate_weighted'].min()))

ssp1_ntl_r_covariance = np.cov(df_annual_ntl_ssp1['ntl'], 
                               total_road_results_ssp1_bayesian['road_estimate_weighted'])[0,1]
ssp1_ntl_r_covariance_scaled = ssp1_ntl_r_covariance/((df_annual_ntl_ssp1['ntl'].max()-df_annual_ntl_ssp1['ntl'].min())*(total_road_results_ssp1_bayesian['road_estimate_weighted'].max()-total_road_results_ssp1_bayesian['road_estimate_weighted'].min()))

ssp1_pd_ntl_covariance = np.cov(pop_ssp1_hist_df['population_density'], 
                                df_annual_ntl_ssp1['ntl'])[0,1]
ssp1_pd_ntl_covariance_scaled = ssp1_pd_ntl_covariance/((pop_ssp1_hist_df['population_density'].max()-pop_ssp1_hist_df['population_density'].min())*( df_annual_ntl_ssp1['ntl'].max()- df_annual_ntl_ssp1['ntl'].min()))

df_annual_gdp_ssp1_var_scaled = df_annual_gdp_ssp1['var']/(df_annual_gdp_ssp1['gdp'].max()-df_annual_gdp_ssp1['gdp'].min())**2

hfi_ssp1_var = total_road_results_ssp1_bayesian_scaled_var['road_var']+df_annual_ntl_ssp1_scaled_var['var']+df_annual_gdp_ssp1_var_scaled.reset_index(drop=True)+2*ssp1_pd_r_covariance_scaled+2*ssp1_ntl_r_covariance_scaled+2*ssp1_pd_ntl_covariance_scaled

hfi_ssp1 = pd.DataFrame(hfi_ssp1, columns=['HFI'])
hfi_ssp1['Lon'] = fpc_ssp3_df['Lon']
hfi_ssp1['Lat'] = fpc_ssp3_df['Lat']
hfi_ssp1['Year'] = fpc_ssp3_df['Year']
hfi_ssp1['HFI_scaled'] = scaler.fit_transform(hfi_ssp1[['HFI']])
hfi_ssp1['var'] = hfi_ssp1_var
print(hfi_ssp1)

#### Plotting HFI
##### 2015
hfi_ssp1_2015 = hfi_ssp1[hfi_ssp1['Year']==2015]

hfi_ssp1_2015 = geopandas.GeoDataFrame(
    hfi_ssp1_2015, 
    geometry=geopandas.points_from_xy(hfi_ssp1_2015.Lon, 
                                      hfi_ssp1_2015.Lat), 
    crs="EPSG:4326"
)


fig, ax = plt.subplots(figsize=(12,10))
europe.plot(color='lightgrey', edgecolor='black', ax=ax)

hfi_ssp1_2015.plot(
    ax=ax,
    column='HFI',
    cmap = 'gist_earth',
    marker='s',
    markersize = 120,
    legend=True,
    legend_kwds = {
        'label': 'HFI',
        'orientation':'vertical'}
    )

ax.set_xlim(europe_boundaries[0], europe_boundaries[2])
ax.set_ylim(europe_boundaries[1], europe_boundaries[3])
ax.set_title("HFI, 2015 (SSP1)", fontsize=14)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

plt.tight_layout()
plt.show()


fig, ax = plt.subplots(figsize=(8, 6))
europe.plot(color='lightgrey', edgecolor='black', ax=ax)

hfi_ssp1_2015.plot(
    ax=ax,
    column='var',
    cmap = 'gist_earth',
    markersize = 120,
    marker='s',
    legend=True,
    legend_kwds = {
        'label': 'Var',
        'orientation':'vertical'}
    )


ax.set_xlim(europe_boundaries[0], europe_boundaries[2])
ax.set_ylim(europe_boundaries[1], europe_boundaries[3])
ax.set_title("Var(HFI) 2015 (SSP1)", fontsize=14)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

plt.tight_layout()
plt.show()



##### 2050
hfi_ssp1_2050 = hfi_ssp1[hfi_ssp1['Year']==2050]

hfi_ssp1_2050 = geopandas.GeoDataFrame(
    hfi_ssp1_2050, 
    geometry=geopandas.points_from_xy(hfi_ssp1_2050.Lon, 
                                      hfi_ssp1_2050.Lat), 
    crs="EPSG:4326"
)

fig, ax = plt.subplots(figsize=(12,10))
europe.plot(color='lightgrey', edgecolor='black', ax=ax)

hfi_ssp1_2050.plot(
    ax=ax,
    column='HFI',
    cmap = 'gist_earth',
    marker='s',
    markersize = 120,
    legend=True,
    legend_kwds = {
        'label': 'HFI',
        'orientation':'vertical'}
    )

ax.set_xlim(europe_boundaries[0], europe_boundaries[2])
ax.set_ylim(europe_boundaries[1], europe_boundaries[3])
ax.set_title("HFI 2050 (SSP1)", fontsize=14)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(8, 6))
europe.plot(color='lightgrey', edgecolor='black', ax=ax)

hfi_ssp1_2050.plot(
    ax=ax,
    column='var',
    cmap = 'gist_earth',
    markersize = 120,
    marker='s',
    legend=True,
    legend_kwds = {
        'label': 'Var',
        'orientation':'vertical'}
    )


ax.set_xlim(europe_boundaries[0], europe_boundaries[2])
ax.set_ylim(europe_boundaries[1], europe_boundaries[3])
ax.set_title("Var(HFI) 2050 (SSP1)", fontsize=14)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

plt.tight_layout()
plt.show()


##### Difference Plot
hfi_ssp1_2015_2050 = hfi_ssp1_2050['HFI'].reset_index(drop=True)-hfi_ssp1_2015['HFI'].reset_index(drop=True)
var_hfi_ssp1_2015_2050 = hfi_ssp1_2050['var'].reset_index(drop=True)-hfi_ssp1_2015['var'].reset_index(drop=True)

hfi_ssp1_2015_2050 = geopandas.GeoDataFrame(
    hfi_ssp1_2015_2050, 
    geometry=geopandas.points_from_xy(hfi_ssp1_2015.Lon, 
                                      hfi_ssp1_2015.Lat), 
    crs="EPSG:4326"
)

fig, ax = plt.subplots(figsize=(12,10))
europe.plot(color='lightgrey', edgecolor='black', ax=ax)

hfi_ssp1_2015_2050.plot(
    ax=ax,
    column='HFI',
    cmap = 'gist_earth',
    marker='s',
    markersize = 120,
    legend=True,
    legend_kwds = {
        'label': 'HFI Difference',
        'orientation':'vertical'}
    )

ax.set_xlim(europe_boundaries[0], europe_boundaries[2])
ax.set_ylim(europe_boundaries[1], europe_boundaries[3])
ax.set_title("HFI: 2015 vs. 2050 (SSP1)", fontsize=14)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(figsize=(8, 6))
h = axes.hist2d(hfi_ssp1_2015['HFI'], hfi_ssp1_2050['HFI'],bins=100, cmap='hot_r', range=[[0, 3.5], [0, 3.5]])
plt.colorbar(h[3], ax = axes, label='count')
plt.plot([0.5,3], [0.5,3], color='black')
axes.set_xlabel('HFI 2015')
axes.set_ylabel('HFI 2050')
axes.set_title('HFI SSP1')
plt.show()

var_hfi_ssp1_2015_2050 = geopandas.GeoDataFrame(
    var_hfi_ssp1_2015_2050, 
    geometry=geopandas.points_from_xy(hfi_ssp1_2015.Lon, 
                                      hfi_ssp1_2015.Lat), 
    crs="EPSG:4326"
)


fig, ax = plt.subplots(figsize=(12,10))
europe.plot(color='lightgrey', edgecolor='black', ax=ax)

var_hfi_ssp1_2015_2050.plot(
    ax=ax,
    column='var',
    cmap = 'gist_earth',
    marker='s',
    markersize = 120,
    legend=True,
    legend_kwds = {
        'label': 'Var(HFI)',
        'orientation':'vertical'}
    )

ax.set_xlim(europe_boundaries[0], europe_boundaries[2])
ax.set_ylim(europe_boundaries[1], europe_boundaries[3])
ax.set_title("Var(HFI): 2015 vs. 2050 (SSP1)", fontsize=14)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

plt.tight_layout()
plt.show()

n_cells = len(hfi_ssp1_2015_2050['HFI'])

ssp1_increase = np.sum(hfi_ssp1_2015_2050['HFI'] > 0.005)
ssp1_decrease = np.sum(hfi_ssp1_2015_2050['HFI'] < -0.005)
ssp1_unchanged = np.sum(np.isclose(hfi_ssp1_2015_2050['HFI'], 0, atol=0.005))

pct_increased = (ssp1_increase / n_cells) * 100
print(f'increased: {pct_increased:.1f}%')
pct_decreased = (ssp1_decrease / n_cells) * 100
print(f'decreased: {pct_decreased:.1f}%')
pct_unchanged = (ssp1_unchanged / n_cells) * 100
print(f'unchanged: {pct_unchanged:.1f}%')


#### Sensitivity
##### EQI
eqi_ssp1_constant = pd.DataFrame(np.tile(eqi_ssp1_scaled[eqi_ssp1_scaled['Year']==2015]['EQI'], n_years), columns=['EQI_2015'])

pop_ssp1_hist_df_test_scaled

hfi_eqi_const_ssp1 = hfi(eqi_ssp1_constant['EQI_2015'], 
                         pop_ssp1_hist_df_test_scaled['population_density'], df_annual_ntl_ssp1_scaled['ntl'], total_road_results_ssp1_bayesian_scaled['road'])

hfi_eqi_const_ssp1['Year'] = fpc_ssp3_df['Year']

hfi_eqi_const_ssp1_2015 = hfi_eqi_const_ssp1[hfi_eqi_const_ssp1['Year']==2015]
hfi_eqi_const_ssp1_2050 = hfi_eqi_const_ssp1[hfi_eqi_const_ssp1['Year']==2050]

##### PD
pop_ssp1_constant = pd.DataFrame(np.tile(pop_ssp1_hist_df_test_scaled[pop_ssp1_hist_df_test_scaled['Year']==2015]['population_density'], n_years), columns=['PD_2015'])

hfi_pd_const_ssp1 = hfi(eqi_ssp1_scaled['EQI'], 
                         pop_ssp1_constant['PD_2015'], df_annual_ntl_ssp1_scaled['ntl'], total_road_results_ssp1_bayesian_scaled['road'])
hfi_pd_const_ssp1
hfi_pd_const_ssp1['Year'] = fpc_ssp3_df['Year']
hfi_pd_const_ssp1

hfi_pd_const_ssp1_2015 = hfi_pd_const_ssp1[hfi_pd_const_ssp1['Year']==2015]
hfi_pd_const_ssp1_2050 = hfi_pd_const_ssp1[hfi_pd_const_ssp1['Year']==2050]

##### NTL
ntl_ssp1_constant = pd.DataFrame(np.tile(df_annual_ntl_ssp1_scaled[df_annual_ntl_ssp1_scaled['Year']==2015]['ntl'], n_years), columns=['NTL_2015'])

hfi_ntl_const_ssp1 = hfi(eqi_ssp1_scaled['EQI'], 
                         pop_ssp1_hist_df_test_scaled['population_density'], ntl_ssp1_constant['NTL_2015'], total_road_results_ssp1_bayesian_scaled['road'])

hfi_ntl_const_ssp1['Year'] = fpc_ssp3_df['Year']


hfi_ntl_const_ssp1_2015 = hfi_ntl_const_ssp1[hfi_ntl_const_ssp1['Year']==2015]
hfi_ntl_const_ssp1_2050 = hfi_ntl_const_ssp1[hfi_ntl_const_ssp1['Year']==2050]

##### R
road_ssp1_constant = pd.DataFrame(np.tile(total_road_results_ssp1_bayesian_scaled[total_road_results_ssp1_bayesian_scaled['Year']==2015]['road'], n_years), columns=['R_2015'])
road_ssp1_constant

hfi_r_const_ssp1 = hfi(eqi_ssp1_scaled['EQI'], 
                         pop_ssp1_hist_df_test_scaled['population_density'], df_annual_ntl_ssp1_scaled['ntl'], road_ssp1_constant['R_2015'])

hfi_r_const_ssp1['Year'] = fpc_ssp3_df['Year']

hfi_r_const_ssp1_2015=hfi_r_const_ssp1[hfi_r_const_ssp1['Year']==2015]
hfi_r_const_ssp1_2050 = hfi_r_const_ssp1[hfi_r_const_ssp1['Year']==2050]

##### Plotting
datasets = [
    (hfi_eqi_const_ssp1_2015['HFI'], hfi_eqi_const_ssp1_2050['HFI'], 'EQI'),
    (hfi_pd_const_ssp1_2015['HFI'],  hfi_pd_const_ssp1_2050['HFI'],  'PD'),
    (hfi_ntl_const_ssp1_2015['HFI'], hfi_ntl_const_ssp1_2050['HFI'], 'NTL'),
    (hfi_r_const_ssp1_2015['HFI'],   hfi_r_const_ssp1_2050['HFI'],   'R'),
]

fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharey=True, sharex=True)
axes_flatten = axes.flatten()
for ax, (x, y, label) in zip(axes_flatten, datasets):
    h = ax.hist2d(x, y, bins=100, cmap='hot_r', range=[[0.5, 4], [0.5, 4]])
    plt.colorbar(h[3], ax=ax, label='Count')
    ax.plot([0.5, 3.5], [0.5, 3.5], color='black', alpha=0.5)
    ax.set_title(label)
    ax.set_xlabel('HFI 2015')
    ax.set_ylabel('HFI 2050')
    
fig.suptitle('HFI SSP1: 2015 vs 2050 (SSP1)', fontsize=14)
plt.tight_layout()
plt.show()

## SSP3
### EQI
mask = lai_ssp3_df.set_index(['Lon', 'Lat']).index.isin(
    fpc_ssp3_df.set_index([ 'Lon', 'Lat']).index
)

lai_ssp3_df = lai_ssp3_df[mask].reset_index(drop=True)

mask = gpp_ssp3_df.set_index(['Lon', 'Lat']).index.isin(
    fpc_ssp3_df.set_index([ 'Lon', 'Lat']).index
)

gpp_ssp3_df = gpp_ssp3_df[mask].reset_index(drop=True)

mask = fpc_ssp3_df.set_index(['Lon', 'Lat']).index.isin(
    fpc_ssp3_df.set_index([ 'Lon', 'Lat']).index
)

fpc_ssp3_df = fpc_ssp3_df[mask].reset_index(drop=True)

eqi_ssp3 = eqi(lai_ssp3_df['Total'], gpp_ssp3_df['Total'], fpc_ssp3_df['total_veg_cover'])

eqi_ssp3['Lon'] = lai_ssp3_df['Lon']
eqi_ssp3['Lat'] = lai_ssp3_df['Lat']
eqi_ssp3['Year'] = lai_ssp3_df['Year']


mask = eqi_ssp3.set_index(['Lon', 'Lat']).index.isin(
    fpc_ssp3_df.set_index([ 'Lon', 'Lat']).index
)

eqi_ssp3 = eqi_ssp3[mask].reset_index(drop=True)

eqi_ssp3_scaled = pd.DataFrame(scaler.fit_transform(eqi_ssp3[['EQI']]),columns=['EQI']).reset_index(drop=True)
eqi_ssp3_scaled['Year'] = lai_ssp3_df['Year']

#### POP
pop_ssp3_hist_df_test = pop_ssp3_hist_df_test.reset_index(drop=True)
pop_ssp3_hist_df_test_scaled = pd.DataFrame(scaler.fit_transform(pop_ssp3_hist_df[['population_density']]),columns=['population_density'])
pop_ssp3_hist_df_test_scaled['Year'] = pop_ssp3_hist_df['Year']

#### NTL
df_annual_ntl_ssp3 = df_annual_ntl_ssp3.reset_index(drop=True)
df_annual_ntl_ssp3_scaled = pd.DataFrame(scaler.fit_transform(df_annual_ntl_ssp3[['ntl']]), columns=['ntl'])
df_annual_ntl_ssp3_scaled_var = df_annual_ntl_ssp3[['var']]/(df_annual_ntl_ssp3['ntl'].max()-df_annual_ntl_ssp3['ntl'].min())**2
df_annual_ntl_ssp3_scaled['Year']= df_annual_ntl_ssp3['Year']


#### Road
total_road_results_ssp3_bayesian = total_road_results_ssp3_bayesian.reset_index(drop=True)
plt.hist(total_road_results_ssp3_bayesian['road_estimate_weighted'], bins=100)
total_road_results_ssp3_bayesian_scaled_var = total_road_results_ssp3_bayesian[['road_var']]/(total_road_results_ssp3_bayesian['road_estimate_weighted'].max()-total_road_results_ssp3_bayesian['road_estimate_weighted'].min())**2
total_road_results_ssp3_bayesian_scaled = pd.DataFrame(scaler.fit_transform(total_road_results_ssp3_bayesian[['road_estimate_weighted']]), columns=['road'])
total_road_results_ssp3_bayesian_scaled['Year']=df_annual_ntl_ssp3['Year']

#### HFI Calculation
hfi_ssp3_df = [eqi_ssp3_scaled['EQI'], 
               pop_ssp3_hist_df_test_scaled['population_density'],
               df_annual_ntl_ssp3_scaled['ntl'], 
               total_road_results_ssp3_bayesian_scaled['road']]
hfi_ssp3_df = pd.concat(hfi_ssp3_df, axis=1)
hfi_ssp3_df.corr()


hfi_ssp3 = hfi(eqi_ssp3_scaled['EQI'], 
               pop_ssp3_hist_df_test_scaled['population_density'],
               df_annual_ntl_ssp3_scaled['ntl'], 
               total_road_results_ssp3_bayesian_scaled['road'])

#### Var
ssp3_pd_r_covariance = np.cov(pop_ssp3_hist_df['population_density'], total_road_results_ssp3_bayesian['road_estimate_weighted'])[0,1]
ssp3_pd_r_covariance_scaled = ssp3_pd_r_covariance/((pop_ssp3_hist_df['population_density'].max()-pop_ssp3_hist_df['population_density'].min())*(total_road_results_ssp3_bayesian['road_estimate_weighted'].max()-total_road_results_ssp3_bayesian['road_estimate_weighted'].min()))

ssp3_ntl_r_covariance = np.cov(df_annual_ntl_ssp3['ntl'], total_road_results_ssp3_bayesian['road_estimate_weighted'])[0,1]
ssp3_ntl_r_covariance_scaled = ssp3_ntl_r_covariance/((df_annual_ntl_ssp3['ntl'].max()-df_annual_ntl_ssp3['ntl'].min())*(total_road_results_ssp3_bayesian['road_estimate_weighted'].max()-total_road_results_ssp3_bayesian['road_estimate_weighted'].min()))

ssp3_pd_ntl_covariance = np.cov(pop_ssp3_hist_df['population_density'], df_annual_ntl_ssp3['ntl'])[0,1]
ssp3_pd_ntl_covariance_scaled = ssp3_pd_ntl_covariance/((pop_ssp3_hist_df['population_density'].max()-pop_ssp3_hist_df['population_density'].min())*( df_annual_ntl_ssp3['ntl'].max()- df_annual_ntl_ssp3['ntl'].min()))

df_annual_gdp_ssp3_var_scaled = df_annual_gdp_ssp3['var']/(df_annual_gdp_ssp3['gdp'].max()-df_annual_gdp_ssp3['gdp'].min())**2

hfi_ssp3_var = total_road_results_ssp3_bayesian_scaled_var['road_var']+df_annual_ntl_ssp3_scaled_var['var']+df_annual_gdp_ssp3_var_scaled.reset_index(drop=True)+2*ssp3_pd_r_covariance_scaled+2*ssp3_ntl_r_covariance_scaled+2*ssp3_pd_ntl_covariance_scaled

hfi_ssp3 = pd.DataFrame(hfi_ssp3, columns=['HFI'])
hfi_ssp3['Lon'] = fpc_ssp3_df['Lon']
hfi_ssp3['Lat'] = fpc_ssp3_df['Lat']
hfi_ssp3['Year'] = fpc_ssp3_df['Year']
hfi_ssp3['HFI_scaled'] = scaler.fit_transform(hfi_ssp3[['HFI']])
hfi_ssp3['var'] = hfi_ssp3_var
print(hfi_ssp3)


#### Plotting HFI

##### 2015
hfi_ssp3_2015 = hfi_ssp3[hfi_ssp3['Year']==2015]

hfi_ssp3_2015 = geopandas.GeoDataFrame(
    hfi_ssp3_2015, 
    geometry=geopandas.points_from_xy(hfi_ssp3_2015.Lon, 
                                      hfi_ssp3_2015.Lat), 
    crs="EPSG:4326"
)


fig, ax = plt.subplots(figsize=(12,10))
europe.plot(color='lightgrey', edgecolor='black', ax=ax)

hfi_ssp3_2015.plot(
    ax=ax,
    column='HFI',
    cmap = 'gist_earth',
    marker='s',
    markersize = 120,
    legend=True,
    legend_kwds = {
        'label': 'HFI',
        'orientation':'vertical'}
    )

ax.set_xlim(europe_boundaries[0], europe_boundaries[2])
ax.set_ylim(europe_boundaries[1], europe_boundaries[3])
ax.set_title("HFI, 2015 (SSP3)", fontsize=14)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(8, 6))
europe.plot(color='lightgrey', edgecolor='black', ax=ax)

hfi_ssp3_2015.plot(
    ax=ax,
    column='var',
    cmap = 'gist_earth',
    markersize = 120,
    marker='s',
    legend=True,
    legend_kwds = {
        'label': 'Var',
        'orientation':'vertical'}
    )


ax.set_xlim(europe_boundaries[0], europe_boundaries[2])
ax.set_ylim(europe_boundaries[1], europe_boundaries[3])
ax.set_title("Var(HFI) 2015 (SSP3)", fontsize=14)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(8, 6))
europe.plot(color='lightgrey', edgecolor='black', ax=ax)

hfi_ssp3_2015.plot(
    ax=ax,
    column='var',
    cmap = 'gist_earth',
    markersize = 120,
    marker='s',
    legend=True,
    legend_kwds = {
        'label': 'Var',
        'orientation':'vertical'}
    )


ax.set_xlim(europe_boundaries[0], europe_boundaries[2])
ax.set_ylim(europe_boundaries[1], europe_boundaries[3])
ax.set_title("Var(HFI) 2015 (SSP3)", fontsize=14)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

plt.tight_layout()
plt.show()

##### 2050
hfi_ssp3_2050 = hfi_ssp3[hfi_ssp3['Year']==2050]

hfi_ssp3_2050 = geopandas.GeoDataFrame(
    hfi_ssp3_2050, 
    geometry=geopandas.points_from_xy(hfi_ssp3_2050.Lon, 
                                      hfi_ssp3_2050.Lat), 
    crs="EPSG:4326"
)

fig, ax = plt.subplots(figsize=(12,10))
europe.plot(color='lightgrey', edgecolor='black', ax=ax)

hfi_ssp3_2050.plot(
    ax=ax,
    column='HFI',
    cmap = 'gist_earth',
    marker='s',
    markersize = 120,
    legend=True,
    legend_kwds = {
        'label': 'HFI',
        'orientation':'vertical'}
    )

ax.set_xlim(europe_boundaries[0], europe_boundaries[2])
ax.set_ylim(europe_boundaries[1], europe_boundaries[3])
ax.set_title("HFI 2050 (SSP3)", fontsize=14)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

plt.tight_layout()
plt.show()


fig, ax = plt.subplots(figsize=(8, 6))
europe.plot(color='lightgrey', edgecolor='black', ax=ax)

hfi_ssp3_2050.plot(
    ax=ax,
    column='var',
    cmap = 'gist_earth',
    markersize = 120,
    marker='s',
    legend=True,
    legend_kwds = {
        'label': 'Var',
        'orientation':'vertical'}
    )


ax.set_xlim(europe_boundaries[0], europe_boundaries[2])
ax.set_ylim(europe_boundaries[1], europe_boundaries[3])
ax.set_title("Var(HFI) 2050 (SSP3)", fontsize=14)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

plt.tight_layout()
plt.show()

##### Difference Plot
hfi_ssp3_2015_2050 = hfi_ssp3_2050['HFI'].reset_index(drop=True)-hfi_ssp3_2015['HFI'].reset_index(drop=True)
var_hfi_ssp3_2015_2050 = hfi_ssp3_2050['var'].reset_index(drop=True)-hfi_ssp3_2015['var'].reset_index(drop=True)
hfi_ssp3_2015_2050

hfi_ssp3_2015_2050 = geopandas.GeoDataFrame(
    hfi_ssp3_2015_2050, 
    geometry=geopandas.points_from_xy(hfi_ssp3_2015.Lon, 
                                      hfi_ssp3_2015.Lat), 
    crs="EPSG:4326"
)

fig, ax = plt.subplots(figsize=(12,10))
europe.plot(color='lightgrey', edgecolor='black', ax=ax)

hfi_ssp3_2015_2050.plot(
    ax=ax,
    column='HFI',
    cmap = 'gist_earth',
    marker='s',
    markersize = 120,
    legend=True,
    legend_kwds = {
        'label': 'HFI',
        'orientation':'vertical'}
    )

ax.set_xlim(europe_boundaries[0], europe_boundaries[2])
ax.set_ylim(europe_boundaries[1], europe_boundaries[3])
ax.set_title("HFI: 2015 vs. 2050 (SSP3)", fontsize=14)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(figsize=(8, 6))
h = axes.hist2d(hfi_ssp3_2015['HFI'], hfi_ssp3_2050['HFI'],bins=100, cmap='hot_r', range=[[0, 3.5], [0, 3.5]])
plt.colorbar(h[3], ax = axes, label='count')
plt.plot([0.5,3], [0.5,3], color='black')
axes.set_xlabel('HFI 2015')
axes.set_ylabel('HFI 2050')
axes.set_title('HFI SSP3')
plt.show()

var_hfi_ssp3_2015_2050 = geopandas.GeoDataFrame(
    var_hfi_ssp3_2015_2050, 
    geometry=geopandas.points_from_xy(hfi_ssp3_2015.Lon, 
                                      hfi_ssp3_2015.Lat), 
    crs="EPSG:4326"
)

fig, ax = plt.subplots(figsize=(12,10))
europe.plot(color='lightgrey', edgecolor='black', ax=ax)

var_hfi_ssp3_2015_2050.plot(
    ax=ax,
    column='var',
    cmap = 'gist_earth',
    marker='s',
    markersize = 120,
    legend=True,
    legend_kwds = {
        'label': 'HFI',
        'orientation':'vertical'}
    )

ax.set_xlim(europe_boundaries[0], europe_boundaries[2])
ax.set_ylim(europe_boundaries[1], europe_boundaries[3])
ax.set_title("Var(HFI): 2015 vs. 2050 (SSP3)", fontsize=14)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

plt.tight_layout()
plt.show()

n_cells = len(hfi_ssp3_2015_2050['HFI'])

ssp3_increase = np.sum(hfi_ssp3_2015_2050['HFI'] > 0.005)
ssp3_decrease = np.sum(hfi_ssp3_2015_2050['HFI'] < -0.005)
ssp3_unchanged = np.sum(np.isclose(hfi_ssp3_2015_2050['HFI'], 0, atol=0.005))

pct_increased = (ssp3_increase / n_cells) * 100
print(f'increased: {pct_increased:.1f}%')

pct_decreased = (ssp3_decrease / n_cells) * 100
print(f'decreased: {pct_decreased:.1f}%')
pct_unchanged = (ssp3_unchanged / n_cells) * 100
print(f'unchanged: {pct_unchanged:.1f}%')

#### Sensitivity
##### EQI
eqi_ssp3_constant = pd.DataFrame(np.tile(eqi_ssp3_scaled[eqi_ssp3_scaled['Year']==2015]['EQI'], n_years), columns=['EQI_2015'])

hfi_eqi_const_ssp3 = hfi(eqi_ssp3_constant['EQI_2015'], 
                         pop_ssp3_hist_df_test_scaled['population_density'], df_annual_ntl_ssp3_scaled['ntl'], total_road_results_ssp3_bayesian_scaled['road'])

hfi_eqi_const_ssp3['Year'] = fpc_ssp3_df['Year']

hfi_eqi_const_ssp3_2015 = hfi_eqi_const_ssp3[hfi_eqi_const_ssp3['Year']==2015]
hfi_eqi_const_ssp3_2050 = hfi_eqi_const_ssp3[hfi_eqi_const_ssp3['Year']==2015]


##### PD
pop_ssp3_constant = pd.DataFrame(np.tile(pop_ssp3_hist_df_test_scaled[pop_ssp3_hist_df_test_scaled['Year']==2015]['population_density'], n_years), columns=['PD_2015'])

hfi_pd_const_ssp3 = hfi(eqi_ssp3_scaled['EQI'], 
                         pop_ssp3_constant['PD_2015'], df_annual_ntl_ssp3_scaled['ntl'], total_road_results_ssp3_bayesian_scaled['road'])

hfi_pd_const_ssp3['Year'] = fpc_ssp3_df['Year']

hfi_pd_const_ssp3_2015 = hfi_pd_const_ssp3[hfi_pd_const_ssp3['Year']==2015]
hfi_pd_const_ssp3_2050 = hfi_pd_const_ssp3[hfi_pd_const_ssp3['Year']==2050]

##### NTL
ntl_ssp3_constant = pd.DataFrame(np.tile(df_annual_ntl_ssp3_scaled[df_annual_ntl_ssp3_scaled['Year']==2015]['ntl'], n_years), columns=['NTL_2015'])

hfi_ntl_const_ssp3 = hfi(eqi_ssp3_scaled['EQI'], 
                         pop_ssp3_hist_df_test_scaled['population_density'], ntl_ssp3_constant['NTL_2015'], total_road_results_ssp3_bayesian_scaled['road'])

hfi_ntl_const_ssp3['Year'] = fpc_ssp3_df['Year']

hfi_ntl_const_ssp3_2015 = hfi_ntl_const_ssp3[hfi_ntl_const_ssp3['Year']==2015]
hfi_ntl_const_ssp3_2050 = hfi_ntl_const_ssp3[hfi_ntl_const_ssp3['Year']==2050]

##### R
road_ssp3_constant = pd.DataFrame(np.tile(total_road_results_ssp3_bayesian_scaled[total_road_results_ssp3_bayesian_scaled['Year']==2015]['road'], n_years), columns=['R_2015'])

hfi_r_const_ssp3 = hfi(eqi_ssp3_scaled['EQI'], 
                         pop_ssp3_hist_df_test_scaled['population_density'], df_annual_ntl_ssp3_scaled['ntl'], road_ssp3_constant['R_2015'])

hfi_r_const_ssp3['Year'] = fpc_ssp3_df['Year']


hfi_r_const_ssp3_2015=hfi_r_const_ssp3[hfi_r_const_ssp3['Year']==2015]
hfi_r_const_ssp3_2050 = hfi_r_const_ssp3[hfi_r_const_ssp3['Year']==2050]

##### Plotting
datasets = [
    (hfi_eqi_const_ssp3_2015['HFI'], hfi_eqi_const_ssp3_2050['HFI'], 'EQI'),
    (hfi_pd_const_ssp3_2015['HFI'],  hfi_pd_const_ssp3_2050['HFI'],  'PD'),
    (hfi_ntl_const_ssp3_2015['HFI'], hfi_ntl_const_ssp3_2050['HFI'], 'NTL'),
    (hfi_r_const_ssp3_2015['HFI'],   hfi_r_const_ssp3_2050['HFI'],   'R'),
]

fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharey=True, sharex=True)
axes_flatten = axes.flatten()
for ax, (x, y, label) in zip(axes_flatten, datasets):
    h = ax.hist2d(x, y, bins=100, cmap='hot_r', range=[[0.5, 4], [0.5, 4]])
    plt.colorbar(h[3], ax=ax, label='Count')
    ax.plot([0.5, 3.5], [0.5, 3.5], color='black', alpha=0.5)
    ax.set_title(label)
    ax.set_xlabel('HFI 2015')
    ax.set_ylabel('HFI 2050')
    
fig.suptitle('HFI: 2015 vs 2050 (SSP3)', fontsize=14)
plt.tight_layout()
plt.show()

## SSP5
### EQI
mask = lai_ssp5_df.set_index(['Lon', 'Lat']).index.isin(
    fpc_ssp3_df.set_index([ 'Lon', 'Lat']).index
)

lai_ssp5_df = lai_ssp5_df[mask].reset_index(drop=True)

mask = gpp_ssp5_df.set_index(['Lon', 'Lat']).index.isin(
    fpc_ssp3_df.set_index([ 'Lon', 'Lat']).index
)

gpp_ssp5_df = gpp_ssp5_df[mask].reset_index(drop=True)

mask = fpc_ssp5_df.set_index(['Lon', 'Lat']).index.isin(
    fpc_ssp3_df.set_index([ 'Lon', 'Lat']).index
)

fpc_ssp5_df = fpc_ssp5_df[mask].reset_index(drop=True)
fpc_ssp5_df = fpc_ssp5_df[fpc_ssp5_df['Year']<=2050]
fpc_ssp5_df = fpc_ssp5_df[fpc_ssp5_df['Year']>=2015]
fpc_ssp5_df = fpc_ssp5_df.reset_index(drop=True)

eqi_ssp5 = eqi(lai_ssp5_df['Total'], gpp_ssp5_df['Total'], fpc_ssp5_df['total_veg_cover'])
eqi_ssp5['Lon'] = lai_ssp5_df['Lon']
eqi_ssp5['Lat'] = lai_ssp5_df['Lat']


eqi_ssp5_scaled = pd.DataFrame(scaler.fit_transform(eqi_ssp5[['EQI']]), columns=['EQI'])
eqi_ssp5_scaled['Lon'] = lai_ssp5_df['Lon']
eqi_ssp5_scaled['Lat'] = lai_ssp5_df['Lat']
eqi_ssp5_scaled['Year'] = lai_ssp5_df['Year']

### HFI
#### POP
pop_ssp5_hist_df = pop_ssp5_hist_df.reset_index(drop=True)
pop_ssp5_hist_df_scaled = pd.DataFrame(scaler.fit_transform(pop_ssp5_hist_df[['population_density']]), columns=['population_density'])
pop_ssp5_hist_df_scaled['Year'] = pop_ssp5_hist_df['Year']


#### NTL
df_annual_ntl_ssp5[['ntl']].min()
df_annual_ntl_ssp5_scaled = pd.DataFrame(scaler.fit_transform(df_annual_ntl_ssp5[['ntl']]), columns=['ntl'])
df_annual_ntl_ssp5_scaled_var = df_annual_ntl_ssp5[['var']]/(df_annual_ntl_ssp5['ntl'].max()-df_annual_ntl_ssp5['ntl'].min())**2
df_annual_ntl_ssp5_scaled_var
df_annual_ntl_ssp5_scaled['Year'] =df_annual_ntl_ssp5['Year']
print(df_annual_ntl_ssp5_scaled)

#### Road
total_road_results_ssp5_bayesian_scaled = pd.DataFrame(scaler.fit_transform(total_road_results_ssp5_bayesian[['road_estimate_weighted']]), columns=['road'])
total_road_results_ssp5_bayesian_scaled_var = total_road_results_ssp5_bayesian[['road_var']]/(total_road_results_ssp5_bayesian['road_estimate_weighted'].max()-total_road_results_ssp5_bayesian['road_estimate_weighted'].min())**2
total_road_results_ssp5_bayesian_scaled['Year']=df_annual_ntl_ssp5['Year']
print(total_road_results_ssp5_bayesian_scaled)

#### Corr
hfi_ssp5_df = [eqi_ssp5_scaled, pop_ssp5_hist_df_scaled, df_annual_ntl_ssp5_scaled, total_road_results_ssp5_bayesian_scaled]
hfi_ssp5_df = pd.concat(hfi_ssp5_df, axis=1)
hfi_ssp5_corr = hfi_ssp5_df.corr()
print(hfi_ssp5_corr)

#### HFI calculation
hfi_ssp5 = hfi(eqi_ssp5_scaled['EQI'], 
              pop_ssp5_hist_df_scaled['population_density'],
              df_annual_ntl_ssp5_scaled['ntl'],
              total_road_results_ssp5_bayesian_scaled['road'])
hfi_ssp5
df_annual_ntl_ssp5_scaled 

hfi_ssp5['HFI_scaled'] = scaler.fit_transform(hfi_ssp5[['HFI']])

#### Var
ssp5_pd_r_covariance = np.cov(pop_ssp5_hist_df['population_density'], total_road_results_ssp5_bayesian['road_estimate_weighted'])[0,1]
ssp5_pd_r_covariance_scaled = ssp5_pd_r_covariance/((pop_ssp5_hist_df['population_density'].max()-pop_ssp5_hist_df['population_density'].min())*(total_road_results_ssp5_bayesian['road_estimate_weighted'].max()-total_road_results_ssp5_bayesian['road_estimate_weighted'].min()))

ssp5_ntl_r_covariance = np.cov(df_annual_ntl_ssp5['ntl'], total_road_results_ssp5_bayesian['road_estimate_weighted'])[0,1]
ssp5_ntl_r_covariance_scaled = ssp5_ntl_r_covariance/((df_annual_ntl_ssp5['ntl'].max()-df_annual_ntl_ssp5['ntl'].min())*(total_road_results_ssp5_bayesian['road_estimate_weighted'].max()-total_road_results_ssp5_bayesian['road_estimate_weighted'].min()))

ssp5_pd_ntl_covariance = np.cov(pop_ssp5_hist_df['population_density'], df_annual_ntl_ssp5['ntl'])[0,1]
ssp5_pd_ntl_covariance_scaled = ssp5_pd_ntl_covariance/((pop_ssp5_hist_df['population_density'].max()-pop_ssp5_hist_df['population_density'].min())*( df_annual_ntl_ssp5['ntl'].max()- df_annual_ntl_ssp5['ntl'].min()))

df_annual_gdp_ssp5_var_scaled = df_annual_gdp_ssp5['var']/(df_annual_gdp_ssp5['gdp'].max()-df_annual_gdp_ssp5['gdp'].min())**2

hfi_ssp5_var = total_road_results_ssp5_bayesian_scaled_var['road_var']+df_annual_ntl_ssp5_scaled_var['var']+df_annual_gdp_ssp5_var_scaled.reset_index(drop=True)+2*ssp5_pd_r_covariance_scaled+2*ssp5_ntl_r_covariance_scaled+2*ssp5_pd_ntl_covariance_scaled
hfi_ssp5_var

hfi_ssp5['var'] = hfi_ssp5_var
hfi_ssp5['Lon'] = fpc_ssp3_df['Lon']
hfi_ssp5['Lat'] = fpc_ssp3_df['Lat']
hfi_ssp5['Year'] = fpc_ssp3_df['Year']
print(hfi_ssp5)

#### Plotting HFI
##### 2015
hfi_ssp5_2015 = hfi_ssp5[hfi_ssp5['Year']==2015]

hfi_ssp5_2015 = geopandas.GeoDataFrame(
    hfi_ssp5_2015, 
    geometry=geopandas.points_from_xy(hfi_ssp5_2015.Lon, 
                                      hfi_ssp5_2015.Lat), 
    crs="EPSG:4326"
)
pop_ssp5_2050_df

fig, ax = plt.subplots(figsize=(14, 6))
europe.plot(color='lightgrey', edgecolor='black', ax=ax)

hfi_ssp5_2015.plot(
    ax=ax,
    column='HFI',
    cmap = 'gist_earth',
    markersize = 120,
    marker='s',
    legend=True,
    legend_kwds = {
        'label': 'HFI',
        'orientation':'vertical'}
    )

ax.set_xlim(europe_boundaries[0], europe_boundaries[2])
ax.set_ylim(europe_boundaries[1], europe_boundaries[3])
ax.set_title("HFI 2015 (SSP5)", fontsize=14)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(8, 6))
europe.plot(color='lightgrey', edgecolor='black', ax=ax)

hfi_ssp5_2015.plot(
    ax=ax,
    column='var',
    cmap = 'gist_earth',
    markersize = 120,
    marker='s',
    legend=True,
    legend_kwds = {
        'label': 'Var',
        'orientation':'vertical'}
    )


ax.set_xlim(europe_boundaries[0], europe_boundaries[2])
ax.set_ylim(europe_boundaries[1], europe_boundaries[3])
ax.set_title("Var(HFI) 2015 (SSP5)", fontsize=14)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

plt.tight_layout()
plt.show()

##### 2050
hfi_ssp5_2050 = hfi_ssp5[hfi_ssp5['Year']==2050]

hfi_ssp5_2050 = geopandas.GeoDataFrame(
    hfi_ssp5_2050, 
    geometry=geopandas.points_from_xy(hfi_ssp5_2050.Lon, 
                                      hfi_ssp5_2050.Lat), 
    crs="EPSG:4326"
)

fig, ax = plt.subplots(figsize=(12,10))
europe.plot(color='lightgrey', edgecolor='black', ax=ax)

hfi_ssp5_2050.plot(
    ax=ax,
    column='HFI',
    cmap = 'gist_earth',
    marker='s',
    markersize = 120,
    legend=True,
    legend_kwds = {
        'label': 'HFI',
        'orientation':'vertical'}
    )

ax.set_xlim(europe_boundaries[0], europe_boundaries[2])
ax.set_ylim(europe_boundaries[1], europe_boundaries[3])
ax.set_title("HFI 2050 (SSP5)", fontsize=14)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(8, 6))
europe.plot(color='lightgrey', edgecolor='black', ax=ax)

hfi_ssp5_2050.plot(
    ax=ax,
    column='var',
    cmap = 'gist_earth',
    markersize = 120,
    marker='s',
    legend=True,
    legend_kwds = {
        'label': 'Var',
        'orientation':'vertical'}
    )


ax.set_xlim(europe_boundaries[0], europe_boundaries[2])
ax.set_ylim(europe_boundaries[1], europe_boundaries[3])
ax.set_title("Var(HFI) 2050 (SSP5)", fontsize=14)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

plt.tight_layout()
plt.show()

##### Difference Plot
hfi_ssp5_2015_2050 = hfi_ssp5_2050['HFI'].reset_index(drop=True)-hfi_ssp5_2015['HFI'].reset_index(drop=True)
var_hfi_ssp5_2015_2050 = hfi_ssp5_2050['var'].reset_index(drop=True)-hfi_ssp5_2015['var'].reset_index(drop=True)
hfi_ssp5_2015_2050

hfi_ssp5_2015_2050 = geopandas.GeoDataFrame(
    hfi_ssp5_2015_2050, 
    geometry=geopandas.points_from_xy(hfi_ssp5_2015.Lon, 
                                      hfi_ssp5_2015.Lat), 
    crs="EPSG:4326"
)

fig, ax = plt.subplots(figsize=(12,10))
europe.plot(color='lightgrey', edgecolor='black', ax=ax)

hfi_ssp5_2015_2050.plot(
    ax=ax,
    column='HFI',
    marker = 's',
    cmap = 'gist_earth',
    markersize = 120,
    legend=True,
    legend_kwds = {
        'label': 'HFI',
        'orientation':'vertical'}
    )

ax.set_xlim(europe_boundaries[0], europe_boundaries[2])
ax.set_ylim(europe_boundaries[1], europe_boundaries[3])
ax.set_title("HFI: 2015 vs. 2050 (SSP5)", fontsize=14)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(figsize=(8, 6))
h = axes.hist2d(hfi_ssp5_2015['HFI'], hfi_ssp5_2050['HFI'],bins=100, cmap='hot_r', range=[[0, 3.5], [0, 3.5]])
plt.colorbar(h[3], ax = axes, label='count')
plt.plot([0.5,3], [0.5,3], color='black')
axes.set_xlabel('HFI 2015')
axes.set_ylabel('HFI 2050')
axes.set_title('HFI SSP5')
plt.show()


var_hfi_ssp5_2015_2050 = geopandas.GeoDataFrame(
    var_hfi_ssp5_2015_2050, 
    geometry=geopandas.points_from_xy(hfi_ssp5_2015.Lon, 
                                      hfi_ssp5_2015.Lat), 
    crs="EPSG:4326"
)


fig, ax = plt.subplots(figsize=(12,10))
europe.plot(color='lightgrey', edgecolor='black', ax=ax)

var_hfi_ssp5_2015_2050.plot(
    ax=ax,
    column='var',
    marker = 's',
    cmap = 'gist_earth',
    markersize = 120,
    legend=True,
    legend_kwds = {
        'label': 'Var(HFI)',
        'orientation':'vertical'}
    )

ax.set_xlim(europe_boundaries[0], europe_boundaries[2])
ax.set_ylim(europe_boundaries[1], europe_boundaries[3])
ax.set_title("Var(HFI): 2015 vs. 2050 (SSP5)", fontsize=14)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

plt.tight_layout()
plt.show()

n_cells = len(hfi_ssp5_2015_2050['HFI'])

ssp5_increase = np.sum(hfi_ssp5_2015_2050['HFI'] > 0.005)
ssp5_decrease = np.sum(hfi_ssp5_2015_2050['HFI'] < -0.005)
ssp5_unchanged = np.sum(np.isclose(hfi_ssp5_2015_2050['HFI'], 0, atol=0.005))

pct_increased = (ssp5_increase / n_cells) * 100
print(f'increased: {pct_increased:.1f}%')

pct_decreased = (ssp5_decrease / n_cells) * 100
print(f'decreased: {pct_decreased:.1f}%')
pct_unchanged = (ssp5_unchanged / n_cells) * 100
print(f'unchanged: {pct_unchanged:.1f}%')

#### Sensitivity
##### EQI
eqi_ssp5_constant = pd.DataFrame(np.tile(eqi_ssp5_scaled[eqi_ssp5_scaled['Year']==2015]['EQI'], n_years), columns=['EQI_2015'])

hfi_eqi_const_ssp5 = hfi(eqi_ssp5_constant['EQI_2015'], 
                         pop_ssp5_hist_df_scaled['population_density'], df_annual_ntl_ssp5_scaled['ntl'], total_road_results_ssp5_bayesian_scaled['road'])

hfi_eqi_const_ssp5['Year'] = fpc_ssp5_df['Year']

hfi_eqi_const_ssp5_2015 = hfi_eqi_const_ssp5[hfi_eqi_const_ssp5['Year']==2015]
hfi_eqi_const_ssp5_2050 = hfi_eqi_const_ssp5[hfi_eqi_const_ssp5['Year']==2015]

##### PD
pop_ssp5_constant = pd.DataFrame(np.tile(pop_ssp5_hist_df_scaled[pop_ssp5_hist_df_scaled['Year']==2015]['population_density'], n_years), columns=['PD_2015'])

hfi_pd_const_ssp5 = hfi(eqi_ssp5_scaled['EQI'], 
                         pop_ssp5_constant['PD_2015'], df_annual_ntl_ssp5_scaled['ntl'], total_road_results_ssp5_bayesian_scaled['road'])

hfi_pd_const_ssp5['Year'] = fpc_ssp5_df['Year']

hfi_pd_const_ssp5_2015 = hfi_pd_const_ssp5[hfi_pd_const_ssp5['Year']==2015]
hfi_pd_const_ssp5_2050 = hfi_pd_const_ssp5[hfi_pd_const_ssp5['Year']==2050]

##### NTL
ntl_ssp5_constant = pd.DataFrame(np.tile(df_annual_ntl_ssp5_scaled[df_annual_ntl_ssp5_scaled['Year']==2015]['ntl'], n_years), columns=['NTL_2015'])

hfi_ntl_const_ssp5 = hfi(eqi_ssp5_scaled['EQI'], 
                         pop_ssp5_hist_df_scaled['population_density'], ntl_ssp5_constant['NTL_2015'], total_road_results_ssp5_bayesian_scaled['road'])

hfi_ntl_const_ssp5['Year'] = fpc_ssp5_df['Year']

hfi_ntl_const_ssp5_2015 = hfi_ntl_const_ssp5[hfi_ntl_const_ssp5['Year']==2015]
hfi_ntl_const_ssp5_2050 = hfi_ntl_const_ssp5[hfi_ntl_const_ssp5['Year']==2050]

##### R
road_ssp5_constant = pd.DataFrame(np.tile(total_road_results_ssp5_bayesian_scaled[total_road_results_ssp5_bayesian_scaled['Year']==2015]['road'], n_years), columns=['R_2015'])

hfi_r_const_ssp5 = hfi(eqi_ssp5_scaled['EQI'], 
                         pop_ssp5_hist_df_scaled['population_density'], df_annual_ntl_ssp5_scaled['ntl'], road_ssp5_constant['R_2015'])

hfi_r_const_ssp5['Year'] = fpc_ssp5_df['Year']

hfi_r_const_ssp5_2015=hfi_r_const_ssp5[hfi_r_const_ssp5['Year']==2015]
hfi_r_const_ssp5_2050 = hfi_r_const_ssp5[hfi_r_const_ssp5['Year']==2050]

##### Plotting
datasets = [
    (hfi_eqi_const_ssp5_2015['HFI'], hfi_eqi_const_ssp5_2050['HFI'], 'EQI'),
    (hfi_pd_const_ssp5_2015['HFI'],  hfi_pd_const_ssp5_2050['HFI'],  'PD'),
    (hfi_ntl_const_ssp5_2015['HFI'], hfi_ntl_const_ssp5_2050['HFI'], 'NTL'),
    (hfi_r_const_ssp5_2015['HFI'],   hfi_r_const_ssp5_2050['HFI'],   'R'),
]

fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharey=True, sharex=True)
axes_flatten = axes.flatten()

for ax, (x, y, label) in zip(axes_flatten, datasets):
    h = ax.hist2d(x, y, bins=100, cmap='hot_r', range=[[0.5, 3.5], [0.5, 3.5]])
    plt.colorbar(h[3], ax=ax, label='Count')
    ax.plot([0.5, 3.5], [0.5, 3.5], color='black', alpha=0.5)
    ax.set_title(label)
    ax.set_xlabel('HFI 2015')
    ax.set_ylabel('HFI 2050')
    
fig.suptitle('HFI: 2015 vs 2050 (SSP5)', fontsize=14)
plt.tight_layout()
plt.show()

## Comparing HFIs
hfi_ssp1_annual = (hfi_ssp1.groupby(['Year'],as_index=False)['HFI'].mean())
hfi_ssp1_variance = (hfi_ssp1.groupby(['Year'],as_index=False)['HFI'].var())
hfi_ssp1_std = np.sqrt(hfi_ssp1_variance['HFI'])
print(hfi_ssp1_annual)

hfi_ssp3_annual = (hfi_ssp3.groupby(['Year'],as_index=False)['HFI'].mean())
hfi_ssp3_variance = (hfi_ssp3.groupby(['Year'],as_index=False)['HFI'].var())
hfi_ssp3_std = np.sqrt(hfi_ssp3_variance['HFI'])
print(hfi_ssp3_annual)

hfi_ssp5_annual = (hfi_ssp5.groupby(['Year'],as_index=False)['HFI'].mean())
hfi_ssp5_variance = (hfi_ssp5.groupby(['Year'],as_index=False)['HFI'].var())
hfi_ssp5_std = np.sqrt(hfi_ssp5_variance['HFI'])
print(hfi_ssp5_annual)

fig, ax = plt.subplots(figsize=(10,5))

ax.fill_between(
    hfi_ssp1_annual['Year'],
    hfi_ssp1_annual['HFI']-hfi_ssp1_std,
    hfi_ssp1_annual['HFI']+hfi_ssp1_std,
    alpha=0.4,
    color='steelblue',
    label='+- SD SSP1')

ax.plot(hfi_ssp1_annual['Year'],  hfi_ssp1_annual['HFI'], marker='o', 
        linewidth=2, markersize=5, label='Mean Annual HFI SSP1')

ax.fill_between(
    hfi_ssp3_annual['Year'],
    hfi_ssp3_annual['HFI']-hfi_ssp3_std,
    hfi_ssp3_annual['HFI']+hfi_ssp3_std,
    alpha=0.4,
    color='tomato',
    label='+- SD SSP3')

ax.plot(hfi_ssp3_annual['Year'],  hfi_ssp3_annual['HFI'], marker='o', 
        linewidth=2, markersize=5, label='Mean Annual HFI SSP3', color='tomato')

ax.fill_between(
    hfi_ssp5_annual['Year'],
    hfi_ssp5_annual['HFI']-hfi_ssp5_std,
    hfi_ssp5_annual['HFI']+hfi_ssp5_std,
    alpha=0.4,
    color='green',
    label='+- SD SSP5')

ax.plot(hfi_ssp5_annual['Year'],  hfi_ssp5_annual['HFI'], marker='o', 
        linewidth=2, markersize=5, label='Mean Annual HFI SSP5', color='green')

ax.legend()
ax.grid(True,linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()

### Comparing between SSP1, 3, 5
plt.hist(hfi_ssp1['HFI'],alpha=0.5, bins=100)
plt.hist(hfi_ssp3['HFI'],alpha=0.5, bins=100)
plt.hist(hfi_ssp5['HFI'],alpha=0.5, bins=100)
plt.legend(labels=['SSP1', 'SSP3', 'SSP5'])
plt.grid(alpha=0.25)
plt.xlabel('HFI')
plt.ylabel('Count')
plt.show()

### Check Normality
#### SSP1
res_ssp1=stats.normaltest(hfi_ssp1['HFI'])
print(res_ssp1.pvalue)

qqplot(hfi_ssp1['HFI'], line='s')
plt.title('SSP1')
plt.show()

#### SSP3
res_ssp3=stats.normaltest(hfi_ssp3['HFI'])
print(res_ssp3.pvalue)

qqplot(hfi_ssp3['HFI'], line='s')
plt.title('SSP3')
plt.show()

#### SSP5
res_ssp5=stats.normaltest(hfi_ssp5['HFI'])
print(res_ssp5.pvalue)

qqplot(hfi_ssp5['HFI'], line='s')
plt.title('SSP5')
plt.show()

### Check Honogenity of Variance
stat, p = stats.levene(hfi_ssp1['HFI'], hfi_ssp3['HFI'], hfi_ssp5['HFI'])
print(p)


### Perform Kruskal
print(stats.kruskal(hfi_ssp1['HFI'], hfi_ssp3['HFI'], hfi_ssp5['HFI']))

hfi_ssp1_array = hfi_ssp1['HFI'].to_numpy()

hfi_ssp3_array = hfi_ssp3['HFI'].to_numpy()

hfi_ssp5_array = hfi_ssp5['HFI'].to_numpy()

p_values = sp.posthoc_dunn([hfi_ssp1_array, hfi_ssp3_array, hfi_ssp5_array])
p_values = p_values.rename(columns={1:'SSP1', 2:'SSP3', 3:'SSP5'}, 
                           index={1:'SSP1', 2:'SSP3', 3:'SSP5'})
print(p_values)

## Comparing other HFI with my HFI
### Human Footprint Mu et al.
hfp_directory_external = [r"C:\Data\HFI\Human Footprint"]
hfp_output_directory = r"C:\Data\HFI\Human Footprint\Output"

resample_tif_to_csv(hfp_directory_external, hfp_output_directory, 
                    europe_boundaries, 'historical', resolution, 'hfp', 
                    target_df=fpc_ssp3_df)

hfp_mu = pd.read_csv(r"C:\Data\HFI\Human Footprint\Output\df_historical_05deg.csv")

hfp_mu = hfp_mu.sort_values(by=['Lon', 'Lat']).reset_index(drop=True)
hfp_mu['hfp_scaled'] = scaler.fit_transform(hfp_mu[['hfp']])

#### Plotting
hfp_mu = geopandas.GeoDataFrame(
    hfp_mu, 
    geometry=geopandas.points_from_xy(hfp_mu.Lon, 
                                      hfp_mu.Lat), 
    crs="EPSG:4326"
)

fig, ax = plt.subplots(figsize=(12,10))
europe.plot(color='lightgrey', edgecolor='black', ax=ax)

hfp_mu.plot(
    ax=ax,
    column='hfp_scaled',
    cmap = 'gist_earth',
    markersize = 150,
    marker='s',
    legend=True,
    legend_kwds = {
        'label': 'HFP',
        'orientation':'vertical'}
    )

ax.set_xlim(europe_boundaries[0], europe_boundaries[2])
ax.set_ylim(europe_boundaries[1], europe_boundaries[3])
ax.set_title("HFP (Mu et al., 2015)", fontsize=14)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

plt.tight_layout()
plt.show()

hfi_scaled_ssp1_2015 = hfi_ssp1[hfi_ssp1['Year']==2015]['HFI_scaled']
coefficients_ssp1 = np.polyfit(hfi_scaled_ssp1_2015, hfp_mu['hfp_scaled'], 1)
p_ssp1 = np.poly1d(coefficients_ssp1)

fig, ax = plt.subplots(figsize=(8, 6))
h = ax.hist2d(hfi_scaled_ssp1_2015, hfp_mu['hfp_scaled'],bins=100, cmap='hot_r', range=[[0, 1.025], [0, 1.4]])
plt.colorbar(h[3], ax = ax, label='count')
ax.plot([0.025,1], [0.025,1], color='black', label='Identity Line')
ax.plot(hfi_scaled_ssp1_2015, p_ssp1(hfi_scaled_ssp1_2015), '-', label='Regression Line')
ax.set_xlabel('HFI SSP1')
ax.set_ylabel('HFI (Mu et al., 2022)')
ax.set_title('HFI 2015')
plt.legend()
plt.show()

rmse = sklearn.metrics.root_mean_squared_error(hfi_scaled_ssp1_2015, 
                                               hfp_mu['hfp_scaled'])
print(f'RMSE SSP1 & HFP: {rmse}')
r, p = stats.pearsonr(hfi_scaled_ssp1_2015, hfp_mu['hfp_scaled'])
print(f'r SSP1 & HFP: {r}')


hfi_scaled_ssp3_2015 = hfi_ssp3[hfi_ssp3['Year']==2015]['HFI_scaled']
coefficients_ssp3 = np.polyfit(hfi_scaled_ssp3_2015, hfp_mu['hfp_scaled'], 1)
p_ssp3 = np.poly1d(coefficients_ssp3)

fig, ax = plt.subplots(figsize=(8, 6))
h = ax.hist2d(hfi_scaled_ssp3_2015, hfp_mu['hfp_scaled'],bins=100, cmap='hot_r', range=[[0, 1.05], [0, 1.5]])
plt.colorbar(h[3], ax = ax, label='count')
ax.plot([0.025,1], [0.025,1], color='black', label='Identity Line')
ax.plot(hfi_scaled_ssp3_2015, p_ssp3(hfi_scaled_ssp3_2015), '-', label='Regression Line')
ax.set_xlabel('HFI SSP3')
ax.set_ylabel('HFP (Mu et al., 2022)')
ax.set_title('HFI 2015')
ax.legend()
plt.show()

rmse = sklearn.metrics.root_mean_squared_error(hfi_scaled_ssp3_2015, 
                                               hfp_mu['hfp_scaled'])
print(f' RMSE SSP3 & HFP: {rmse}')
r, p = stats.pearsonr(hfi_scaled_ssp3_2015, hfp_mu['hfp_scaled'])
print(f'r SSP3 & HFP: {r}')


hfi_scaled_ssp5_2015 = hfi_ssp5[hfi_ssp5['Year']==2015]['HFI_scaled']
coefficients_ssp5 = np.polyfit(hfi_scaled_ssp5_2015, hfp_mu['hfp_scaled'], 1)
p_ssp5 = np.poly1d(coefficients_ssp5)

fig, ax = plt.subplots(figsize=(8, 6))
h = ax.hist2d(hfi_scaled_ssp5_2015, hfp_mu['hfp_scaled'],bins=100, cmap='hot_r', range=[[0, 1.025], [0, 1.4]])
plt.colorbar(h[3], ax = ax, label='count')
ax.plot([0.025,1], [0.025,1], color='black', label='Identity Line')
ax.plot(hfi_scaled_ssp5_2015, p_ssp5(hfi_scaled_ssp5_2015), '-', label='Regression Line')
ax.set_xlabel('HFI SSP5')
ax.set_ylabel('HFP (Mu et al., 2022)')
ax.set_title('HFI 2015')
plt.legend()
plt.show()

rmse = sklearn.metrics.root_mean_squared_error(hfi_scaled_ssp5_2015, 
                                               hfp_mu['hfp_scaled'])
print(f'RMSE SSP5 & HFP: {rmse}')
r, p = stats.pearsonr(hfi_scaled_ssp5_2015, hfp_mu['hfp_scaled'])
print(f'r SSP5 & HFP: {r}')