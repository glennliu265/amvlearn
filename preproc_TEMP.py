#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 20:01:59 2020

Preprocess TEMP and save the files...

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

# Paths
path1 = "/stormtrack/data4/share/deep_learning/data_yuchiaol/cesm_le/TEMP/" # Potential Temperature

# Region for analysis
bbox = [260,20,25,80] # [lonW, lonE, latS, latN]

# seasons to take average over
djfm = [12,1,2,3] 

#%% Get lists of nc files

# Set up
keepdims = keepdims_temp

varname = "TEMP"
ncpath  = path1

ncnames =  "b.e11.B20TRC5CNBDRD.f09_g16.*.pop.h." + varname + ".*.nc"
globby  = ncpath+ncnames

# Get list of variables for testing
nclist = glob.glob(globby)
nclist = [i for i in nclist if "OIC" not in i]
nclist.sort()

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



#%%
# Load HMXL Data
hmxl = np.load("/home/glliu/01_Data/00_Scrap/"+"HMXL_DJFM_clim.npy")


# Find maximum mld
zmax = np.nanmax(hmxl)

#%%
outpath = "/stormtrack/data3/glliu/01_Data/02_AMV_Project/02_stochmod/TEMP"
allstart = time.time()
for e in range(nens):
    # Individually read in the temp data... (70 sec)
    start = time.time()
    dstemp  = xr.open_dataset(nclist[e])
    tempw  = preprocess1(dstemp,varlist=keepdims_temp) # Preprocess
    #tempw = tempw.where( ((tempw.TLONG >= lonW) | (tempw.TLONG <= lonE)) & ((tempw.TLAT >= latS) & (tempw.TLAT <= latN)),drop=True)
    if e == 0: # Find depths on first iteration
      dz = tempw.z_w_top.values   
      idz = np.argmax(dz > zmax) + 1
    # Slice to depth of deepest MLD
    tempw = tempw.sel(z_t = slice(0,idz))  
    tempw   = seasonal_mean(tempw,djfm) # (year, ensemble, z_t, nlat, nlon)
    temp = tempw.TEMP.values
    outname = "%sTEMP_DJFM_ens%02d.npy" % (outpath,e+1)
    np.save(outname,temp)
    
    print("Read in TEMP for ens%i in %.2fs"%(e+1,time.time()-start))
    
    
    
    
    



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

# Now looping for each, prepare for calculations
uohc = np.ones((nlat,nlon,nyr,nens)) * np.nan
for e in range(nens):
    
    
    # Individually read in the temp data... (83 sec)
    start = time.time()
    dstemp  = xr.open_dataset(nclists[0][e])
    dstemp  = preprocess1(dstemp,varlist=keepdims[0]) # Preprocess
    tempw   = seasonal_mean(dstemp,djfm) # (year, ensemble, z_t, nlat, nlon)
    # tempall = tempw.TEMP.values # [ens x year x tlat x tlon ]
    # dztop   = tempw.z_w_top.values
    print("Read in TEMP for ens%i in %.2fs"%(e+1,time.time()-start))
    
    
    
    
    


for e in range(nens):
    for o in range(nlon):
        for a in range(nlat):
            
            
            
            mldpt  = hmxlall[e,o,a]
            if np.any(np.isnan(mldpt)):
                continue # Skip point if any Nans are found
            
            # Get index for dz (first level greater than MLD)
            idD = np.argmax(dztop > mldpt)
            
            
            
            
            
            for y in range(nyr):
                
                temppt = tempall[e,y,:,a,o]# Get temp for that year
                
                
                
                
                
                
                
            








    
# Take DJFM Average
# Take DJFM average

ds = seasonal_mean(ds,djfm) 

temp_djfm = seasonal_mean(dstemp,djfm)
hmxl_djfm = seasonal_mean(dshmxl,djfm)


# General Procedure
# All Depths are in Centimeters, convert to meters ***??


# Load in HMXL and UOHC
dstemp = xr.open_dataset()
dshmxl = xr.open_dataset()

# Find the specified point on curvilinear grid and average values
start = time.time()
test = ds.where((ds.TLONG>=lonW) & (ds.TLONG <= lonE)
                & (ds.TLAT>=latS) & (ds.TLAT <= latN),drop=True)
print("Cut to region in  %.2fs"%(time.time()-start))



# Crop to North Atlantic (TBH might be hard with the T-grid...)

# For each ensemble member...
for e in range(nens):


        # Select values corresponding to HMXL and SUM
        
        # Divide by the HMXL
        
        # Store in shared variable
        
       
        