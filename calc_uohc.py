#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 15:44:28 2020

@author: gliu
"""

import xarray as xr
import numpy as np
import time
import glob

def seasonal_mean(ds,mons):
    """
    Inputs
        1) ds, dataarray containing target variable
        2) mons, months to take the mean over (1-12)
    """
    season = ds.sel(time=np.in1d(ds['time.month'],mons)) # Select winter months
    dsm = season.groupby('time.year').mean('time') # take the mean    
    return dsm

def selregion_pop(bbox,invar,tlon,tlat,printfind=False):
    
    """
    IMPT: assumes input variable is of the shape [lat x lon x otherdims]
    tlon = ARRAY [lat x lon], 0 - 360 ( deg. east)
    tlat = ARRAY [lat x lon]
    """
    
    
    lonW,lonE,latS,latN = bbox # Read out bbox
    # Query Points
    if lonW > lonE: # Crossing Prime Meridian
        quer = np.where((tlon <= lonW) | (tlon >= lonE) & (tlat >= latS) & (tlat <= latN))
    else:
        quer = np.where((tlon >= lonW) & (tlon <= lonE) & (tlat >= latS) & (tlat <= latN))
    latid,lonid = quer
    
    if printfind:
        print("Closest LAT to %.1f was %s" % (latf,tlat[quer]))
        print("Closest LON to %.1f was %s" % (lonf,tlon[quer]))
        
    if (len(latid)==0) | (len(lonid)==0):
        print("Returning NaN because no points were found for LAT%.1f LON%.1f"%(latf,lonf))
        return np.nan
        exit
    
    # Locate points on variable
    if invar.shape[:2] != tlon.shape:
        print("Warning, dimensions do not line up. Make sure invar is Lat x Lon x Otherdims")
        exit
    
    return invar[latid,lonid,:],tlat[latid,lonid],tlat[latid,lonid] # Take mean along first dimension
    
def getpt_pop_array(lonf,latf,invar,tlon,tlat,searchdeg=0.75,printfind=True):
    
    """
    IMPT: assumes input variable is of the shape [lat x lon x otherdims]
    tlon = ARRAY [lat x lon]
    tlat = ARRAY [lat x lon]
    """
    
    if lonf < 0:# Convet longitude to degrees East
        lonf += 360
    
    # Query Points
    quer = np.where((lonf-searchdeg < tlon) & (tlon < lonf+searchdeg) & (latf-searchdeg < tlat) & (tlat < latf+searchdeg))
    latid,lonid = quer
    
    if printfind:
        print("Closest LAT to %.1f was %s" % (latf,tlat[quer]))
        print("Closest LON to %.1f was %s" % (lonf,tlon[quer]))
        
    if (len(latid)==0) | (len(lonid)==0):
        print("Returning NaN because no points were found for LAT%.1f LON%.1f"%(latf,lonf))
        return np.nan
        exit
    
    
    # Locate points on variable
    if invar.shape[:2] != tlon.shape:
        print("Warning, dimensions do not line up. Make sure invar is Lat x Lon x Otherdims")
        exit
    
    return invar[latid,lonid,:].mean(0) # Take mean along first dimension
    
#%% User Edits

# Variables to keep
keepdims_temp = ["TEMP","time","dz","z_w_top","z_t_150m","TLONG","TLAT"] # ['Time x z_t x nlat x nlon]
keepdims_hmxl = ["HMXL","time","TLONG","TLAT"]

# Paths
path1 = "/stormtrack/data4/share/deep_learning/data_yuchiaol/cesm_le/TEMP/" # Potential Temperature
path2 = "/vortex/jetstream/climate/data1/yokwon/CESM1_LE/downloaded/ocn/proc/tseries/monthly/HMXL/" # MLD

# Region for analysis
bbox = [260,20,25,80] # [lonW, lonE, latS, latN]

# seasons to take average over
djfm = [12,1,2,3] 

#%% Get lists of nc files

# Set up
varnames = ["TEMP","HMXL"]
paths = [path1,path2]
keepdims = [keepdims_temp,keepdims_hmxl]
nclists = []

for i in range(2):
    varname = varnames[i]
    ncpath  = paths[i]
    
    ncnames =  "b.e11.B20TRC5CNBDRD.f09_g16.*.pop.h." + varname + ".*.nc"
    globby  = ncpath+ncnames
    
    # Get list of variables for testing
    nclist = glob.glob(globby)
    nclist = [i for i in nclist if "OIC" not in i]
    nclist.sort()
    
    nclists.append(nclist)
    print("Found %i files, starting with %s"%(len(nclist),nclist[0]))


#%% # Load in the data

lonW,lonE,latS,latN = bbox 

# Preproc for TEMP  (Maybe also cut to the deepest mixed layer?)
def preprocess1(ds,varlist=keepdims[0]):
    # Drop unwanted dimension
    dsvars = list(ds.variables)
    remvar = [i for i in dsvars if i not in varlist]
    ds = ds.drop(remvar)

    
    # Correct time to start at first of each month
    if ds.time.values[0].month != 1:
        startyr = str(ds.time.values[0].year)
        correctedtime = xr.cftime_range(start=startyr,end="2005-12-31",freq="MS",calendar="noleap") 
        ds = ds.assign_coords(time=correctedtime) 
    
    # Crop to time period
    ds = ds.sel(time=slice("1920-01-01","2005-12-31"))

    
    return ds
    

# Preproc for HMXL 
def preprocess2(ds,varlist=keepdims[1]):
    # Drop unwanted dimension
    dsvars = list(ds.variables)
    remvar = [i for i in dsvars if i not in varlist]
    ds = ds.drop(remvar)

    
    # Correct time to start at first of each month
    if ds.time.values[0].month != 1:
        startyr = str(ds.time.values[0].year)
        correctedtime = xr.cftime_range(start=startyr,end="2005-12-31",freq="MS",calendar="noleap") 
        ds = ds.assign_coords(time=correctedtime) 
    
    # Crop to time period
    ds = ds.sel(time=slice("1920-01-01","2005-12-31"))
    
    return ds



#%% Preprocess the HXML First

# Read in HXML (11.40 sec)
start = time.time()
dshmxl = xr.open_mfdataset(nclists[1],concat_dim='ensemble',
                           preprocess=preprocess2)
print("Read in HXML in %.2fs"%(time.time()-start))

# Take the winter average (0.46 sec)
start = time.time()
hmxlw = seasonal_mean(dshmxl,djfm)
hmxlw = hmxlw.mean('year')
print("Took DJFM climatological mean for HXML in %.2fs"%(time.time()-start))

# Read out the data (188.15 sec)
start = time.time()
hmxlall = hmxlw.HMXL.values # [ens x tlat x tlon]
TLON    = hmxlw.TLONG.values
TLAT    = hmxlw.TLAT.values
tlon = TLON.mean(0) # remove ensemble
tlat = TLAT.mean(0) # remove ensemble
nens,nlat,nlon = hmxlall.shape
print("Loaded data for  HXML in %.2fs"%(time.time()-start))

# Find the maximum value
maxz = np.nanmax(hmxlall) 

dshmxl.close()
hmxlw.close()

# Save variable in case
outpath = "/home/glliu/01_Data/00_Scrap/"
np.save(outpath+"HMXL_DJFM_clim.npy",hmxlall) 

#%% Took 6883.39s
    
# Read in Potential Temperature
start = time.time()
dstemp = xr.open_mfdataset(nclists[0][:40],concat_dim='ensemble',
                            preprocess=preprocess1)
print("Read in TEMP in %.2fs"%(time.time()-start))

# Take winter average
start = time.time()
tempw = seasonal_mean(dstemp,djfm) # (year, ensemble, z_t, nlat, nlon)
print("Took DJFM mean for TEMP in %.2fs"%(time.time()-start))

# Get cell top values
dztop = tempw.z_w_top.values

# Read out the data
start   = time.time()
tempall = tempw.TEMP.values # [ens x year x tlat x tlon ]
dztop   = tempw.z_w_top.values
print("Loaded data for TEMP sin %.2fs"%(time.time()-start))



#%% Calculate UOHC
outpath = '/stormtrack/data3/glliu/01_Data/02_AMV_Project/02_stochmod/TEMP/'
# Load HMXL Data
hmxl = np.load("/home/glliu/01_Data/00_Scrap/"+"HMXL_DJFM_clim.npy")
dztop = np.load(outpath+"z_w_top.npy")

# Find maximum mld
zmax = np.nanmax(hmxl)
idz = np.argmax(dztop > zmax)+1 # Index of maximum MLD

nens,nlat,nlo = hmxl.shape

# Now looping for each, prepare for calculations
allstart=time.time()
uohc = np.ones((nlat,nlon,nyr,nens)) * np.nan
i = 0
for e in range(nens):
    for o in range(nlon):
        for a in range(nlat):
            
            mldpt  = hmxl[e,a,o]
            if np.any(np.isnan(mldpt)):
                i+=1
                continue # Skip point if any Nans are found
            
            # Get index for dz (first level greater than MLD)
            idz = np.argmax(dztop > mldpt)+1 
            
            
            # Get TEMP for the point and ensemble
            temppt = tempall[:,e,:,a,o]# [86 x 60] [year x depth]
            
            # Select MLD values and sum, and divide by number of levels
            tempsum = temppt[:,:idz].sum(1)/idz
            
            # Record variable
            uohc[a,o,:,e] = tempsum.copy()
            
            i += 1
            if i%1e5 ==0:
                print("Completed for %i of 4915200" % i)
                

#%% Regrid

# First get TLONG and TLAT
ds = xr.open_dataset(nclists[1][1])


# Make or get dimensions
tlon = ds.TLONG.values
tlat = ds.TLAT.values
yrs = np.arange(1920,2006,1)

# Get CESM-LE dimensions
from scipy.io import loadmat
ll = loadmat('/home/glliu/01_Data/CESM1_LATLON.mat') 
lon = ll["LON"].squeeze()
lat = ll["LAT"].squeeze()


uohc = uohc.reshape(384,320,86*40) # (combine year and dimension)

uohc_rg = np.zeros((len(lat),len(lon),len(yrs)*nens))*np.nan
i=0
for a in range(len(lat)):
    
    for o in range(len(lon)):
        
        latf = lat[a]
        lonf = lon[o]
        ptvalue = getpt_pop_array(lonf,latf,uohc,tlon,tlat,searchdeg=0.75,printfind=False)
        
        if np.any(np.isnan(ptvalue)):
           i+=1
           continue
       
        uohc_rg[a,o,:] = ptvalue.copy()
        i+= 1
        print("Completed %i of 55296"%i)


uohc_rg = uohc_rg.reshape(len(lat),len(lon),len(yrs),nens)

ds = xr.DataArray(uohc_rg,
                coords={'lat':lat,'lon':lon,"year":yrs,"ensemble":np.arange(1,41,1)},
                dims={'lat':lat,'lon':lon,"year":yrs,"ensemble":np.arange(1,41,1)},
                name="UOHC")

#ds.rename({"uohc_rg"="UOHC"})

ds = ds.transpose("ensemble","year","lat","lon",transpose_coords=True)

ds.to_netcdf(outpath+"CESM1LE_UOHC_DJFM_Regridded_1920_2005.nc",encoding={'UOHC': {'zlib': True}})   
    

#%% Something was wrong, reopen things
outpath = '/stormtrack/data3/glliu/01_Data/02_AMV_Project/02_stochmod/TEMP/'
filename = 'CESM1LE_UOHC_DJFM_Regridded_1920_2005.nc'


     


