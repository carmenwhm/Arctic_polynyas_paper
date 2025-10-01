import os
import glob
import netCDF4 as nc 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
from skimage.segmentation import flood, flood_fill
from scipy.ndimage import label, generate_binary_structure, binary_fill_holes
from scipy.interpolate import griddata
from pyhdf.SD import *
from pyhdf.HDF import *

def nc_merge_files(data_folder,grid_folder,data_path_name,grid_path_name,nc_var_name):

    full_lat_list = []
    full_lon_list = []
    full_nc_list = []
    date_list = []

    data_path = os.path.join(data_folder,'%s'%data_path_name)
    nc_files = sorted(glob.glob(data_path))
    grid_file = os.path.join(grid_folder,'%s'%grid_path_name)

    for file in list(nc_files):
        data_f = nc.Dataset(file)
        file_var = data_f.variables['%s'%nc_var_name]
        basename = os.path.basename(file)
        date = os.path.splitext(basename)[0][:8]

        full_nc_list.append(file_var)
        date_list.append(date)

    
    grid_f = SD(grid_file, SDC.READ)
    lat = grid_f.select('Latitudes').get()
    lon = grid_f.select('Longitudes').get()

    file_num = len(glob.glob(data_path))
    full_lat_list = np.repeat(lat[np.newaxis,:,:],file_num,axis=0)
    full_lon_list = np.repeat(lon[np.newaxis,:,:],file_num,axis=0)

    full_lat_arr = np.array(full_lat_list)
    full_lon_arr = np.array(full_lon_list)
    full_nc_arr = np.array(full_nc_list)

    print(full_lat_arr.shape)
    print(full_nc_arr.shape)
    print(file_num)

    return full_lat_arr, full_lon_arr, full_nc_arr, date_list, file_num


# Function for selecting specific region for studies based on lon lat 
def select_region(full_lon,full_lat,full_var,lon_min,lon_max,lat_min,lat_max,Delete_data=None):
    z1 = (full_lon > lon_min) & (full_lon < lon_max)
    selected_lon = np.ma.masked_array(full_lon,mask=np.invert(z1))

    z2 = (full_lat > lat_min) & (full_lat < lat_max)
    selected_lat = np.ma.masked_array(full_lat,mask=np.invert(z2))

    if Delete_data == True:                                                                                                         
        region_mask = np.logical_and(z1,z2)
        selected_var = full_var*region_mask
        selected_var = np.where(region_mask,full_var,np.nan)

    else:
        region_mask = np.logical_and(z1,z2)
        selected_var = np.ma.masked_array(full_var,mask=np.invert(region_mask))

    return selected_lon,selected_lat,selected_var


def select_region_rev(full_lon,full_lat,full_var,lon_min,lon_max,lat_min,lat_max,Delete_data=None):
    z1 = (lon_min < full_lon) | (full_lon < lon_max)
    selected_lon = np.ma.masked_array(full_lon,mask=np.invert(z1))

    z2 = (full_lat > lat_min) & (full_lat < lat_max)
    selected_lat = np.ma.masked_array(full_lat,mask=np.invert(z2))

    if Delete_data == True:                                                           
        region_mask = np.logical_and(z1,z2)
        selected_var = full_var*region_mask
        selected_var = np.where(region_mask,full_var,np.nan)

    else:
        region_mask = np.logical_and(z1,z2)
        selected_var = np.ma.masked_array(full_var,mask=np.invert(region_mask))

    return selected_lon,selected_lat,selected_var


def nc_merge_files_NSIDC(data_folder,grid_folder,data_path_name,grid_name,nc_var_name):

    data_path = os.path.join(data_folder,'%s'%data_path_name)
    nc_files = sorted(glob.glob(data_path))
    grid_file = os.path.join(grid_folder,'%s'%grid_name)

    grid_f = nc.Dataset(grid_file)
    upscale_lat = grid_f.variables['latitude'][:]
    upscale_lon = grid_f.variables['longitude'][:]

    # --- filter files first ---
    selected_files = []

    for file in nc_files:
        basename = os.path.basename(file)
        date_check = basename.split('_')[4]
        if date_check[4:6] not in ("11", "04"):
            selected_files.append(file)
      
    # --- preallocate arrays with correct size ---
    n_files = len(selected_files)

    full_lat_list = np.full((n_files,upscale_lat.shape[0],upscale_lon.shape[1]),np.nan)
    full_lon_list = np.full((n_files,upscale_lat.shape[0],upscale_lon.shape[1]),np.nan)
    full_nc_list = np.full((n_files,upscale_lat.shape[0],upscale_lon.shape[1]),np.nan)
    date_list = np.full(n_files,np.nan)

    for ifile,file in enumerate(selected_files):
        try:
            data_f = nc.Dataset(file)
            file_var = data_f.variables['%s'%nc_var_name][0,:,:]
            basename = os.path.basename(file)
            date = basename.split('_')[4]

            full_nc_list[ifile] = file_var
            date_list[ifile] = date
    
        except KeyError:
            try:
                data_f = nc.Dataset(file)
                file_var = data_f.variables['F11_ICECON'][0,:,:]
                basename = os.path.basename(file)
                date = basename.split('_')[4]
    
                full_nc_list[ifile] = file_var
                date_list[ifile] = date

            except KeyError:
                try:
                    data_f = nc.Dataset(file)
                    file_var = data_f.variables['F17_ICECON'][0,:,:]
                    basename = os.path.basename(file)
                    date = basename.split('_')[4]
        
                    full_nc_list[ifile] = file_var
                    date_list[ifile] = date

                except KeyError:
                    pass
    
    file_num = len(date_list)
    full_lat_list = np.repeat(upscale_lat[np.newaxis,:,:],file_num,axis=0)
    full_lon_list = np.repeat(upscale_lon[np.newaxis,:,:],file_num,axis=0)


    print(full_lat_list.shape)
    print(full_nc_list.shape)
    print(file_num)

    return full_lat_list, full_lon_list, full_nc_list, date_list, file_num


def polynyas_loc_SIC_LR(SIC_arr,threshold):

    time_SIC = np.nan_to_num(SIC_arr,nan=-999)
    polynyas_loc = flood_fill(time_SIC,(124,50),999,tolerance=threshold)     ## This is 25 km seeding
    polynyas_loc_1 = flood_fill(polynyas_loc,(405,230),999,tolerance=threshold) 
    counts = np.logical_and(polynyas_loc_1>=0,polynyas_loc_1<=threshold)                                                                            
    anchor_arr = counts.astype(int)
    SIC_location = np.where(counts==1,1,0)

    return polynyas_loc,polynyas_loc_1,SIC_location
