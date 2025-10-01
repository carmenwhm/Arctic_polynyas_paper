import netCDF4 as nc 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
import matplotlib as mpl
from skimage.segmentation import flood, flood_fill
from scipy.ndimage import label, generate_binary_structure, binary_fill_holes
from scipy.interpolate import griddata
from pyhdf.SD import *
from pyhdf.HDF import *
from Polynyas_trend_function_SMMR import *


if __name__ == '__main__':

    # 25km SMMR SIC grid
    f_read = nc.Dataset('/proj/naiss2024-22-523/PhD_Year1/NSIDC0771_LatLon_PS_N25km_v1.0.nc')
    SIC_lat_SMMR = f_read.variables['latitude'][:]
    SIC_lon_SMMR = f_read.variables['longitude'][:]
    
    # 25km SMMR SIC grid area 
    f_area = nc.Dataset('/proj/naiss2024-22-523/PhD_Year1/NSIDC0771_CellArea_PS_N25km_v1.0.nc')
    cell_area_SMMR = f_area['cell_area'][:]

    # Year array
    yr_1978 = np.arange(1978,2024)
    
    # Concentration variation
    conc_SMMR = np.arange(0.3,0.7,0.1)
    # conc_SMMR = np.arange(0.5,0.6,0.1)

    SIC_yearly_base_loc_1978 = np.zeros((conc_SMMR.shape[0],yr_1978.shape[0],SIC_lat_SMMR.shape[0],SIC_lat_SMMR.shape[1]))
    Daily_location = np.full((SIC_lat_SMMR.shape[0],SIC_lat_SMMR.shape[1]),np.nan)
    Daily_location = np.expand_dims(Daily_location, axis=0)
    total_area_SIC = np.zeros((conc_SMMR.shape[0],yr_1978.shape[0]))
    total_num_SIC = np.zeros((conc_SMMR.shape[0],yr_1978.shape[0]))
    total_area_SMMR = np.zeros((conc_SMMR.shape[0],yr_1978.shape[0]))
    date_list = []

    # Read one winter files
    for i, year in enumerate(yr_1978):
        SIC_Data_folder = '/proj/naiss2024-22-523/PhD_Year1/1978_2024_SMMR/%s_%s/SIC/'%(year,year+1)
        Grid_folder = '/proj/naiss2024-22-523/PhD_Year1/'
        gridf_name_25 = 'NSIDC0771_LatLon_PS_N25km_v1.0.nc'
        SIC_datafile_name = '*v2.0.nc' 
    
        if year in np.arange(1978,1987):
            print(year)
            full_lat_arr, full_lon_arr, full_sic_arr, date_test, siconc_file_num = nc_merge_files_NSIDC(SIC_Data_folder,Grid_folder,\
                                                                                        SIC_datafile_name,gridf_name_25,'N07_ICECON')
            
        elif year in np.arange(1987,1992):
            print(year)
            full_lat_arr, full_lon_arr, full_sic_arr, date_test, siconc_file_num = nc_merge_files_NSIDC(SIC_Data_folder,Grid_folder,\
                                                                                        SIC_datafile_name,gridf_name_25,'F08_ICECON')
            
        elif year in np.arange(1992,1995):
            print(year)
            full_lat_arr, full_lon_arr, full_sic_arr, date_test, siconc_file_num = nc_merge_files_NSIDC(SIC_Data_folder,Grid_folder,\
                                                                                        SIC_datafile_name,gridf_name_25,'F11_ICECON')
            
        elif year in np.arange(1995,2008):
            print(year)
            full_lat_arr, full_lon_arr, full_sic_arr, date_test, siconc_file_num = nc_merge_files_NSIDC(SIC_Data_folder,Grid_folder,\
                                                                                        SIC_datafile_name,gridf_name_25,'F13_ICECON')
        else:
            print(year)
            full_lat_arr, full_lon_arr, full_sic_arr, date_test, siconc_file_num = nc_merge_files_NSIDC(SIC_Data_folder,Grid_folder,\
                                                                                        SIC_datafile_name,gridf_name_25,'F17_ICECON')
    
        # date_list.append(date_test) # For daily file
        
        SIC_location = np.zeros((siconc_file_num,SIC_lat_SMMR.shape[0],SIC_lat_SMMR.shape[1]))     
        SIC_polynyas_labeled = np.zeros((siconc_file_num,SIC_lat_SMMR.shape[0],SIC_lat_SMMR.shape[1]))
        filter_sic = np.zeros((siconc_file_num,SIC_lat_SMMR.shape[0],SIC_lat_SMMR.shape[1]))
        SIC_polynyas_num = np.zeros(siconc_file_num)
        SIC_area = []
        SIC_daily_num = []
    

        for j, concentration in enumerate(conc_SMMR):
            for k in range(siconc_file_num):

                index = 0+k
                
                masked_SIC = np.where(full_sic_arr[index]>250,-999,full_sic_arr[index])
                _, _, SIC_location[index] = polynyas_loc_SIC_LR(masked_SIC,concentration)
    
                _, _, exclude_regional_1 = select_region(full_lon=SIC_lon_SMMR,full_lat=SIC_lat_SMMR,full_var=full_sic_arr[index],\
                                                                        lon_min=70,lon_max=84,lat_min=66,lat_max=73,Delete_data=True)
        
                _, _, exclude_regional_2 = select_region(full_lon=SIC_lon_SMMR,full_lat=SIC_lat_SMMR,full_var=full_sic_arr[index],\
                                                                                lon_min=20,lon_max=25,lat_min=65,lat_max=66,Delete_data=True)
                
                _, _, exclude_regional_3 = select_region_rev(full_lon=SIC_lon_SMMR,full_lat=SIC_lat_SMMR,full_var=full_sic_arr[index],\
                                                                                lon_min=180,lon_max=-175,lat_min=65,lat_max=67,Delete_data=True)
                
                _, _, exclude_regional_4 = select_region(full_lon=SIC_lon_SMMR,full_lat=SIC_lat_SMMR,full_var=full_sic_arr[index],\
                                                                                lon_min=-68,lon_max=-64,lat_min=65,lat_max=66.5,Delete_data=True)
                
                _, _, exclude_regional_5 = select_region(full_lon=SIC_lon_SMMR,full_lat=SIC_lat_SMMR,full_var=full_sic_arr[index],\
                                                                                lon_min=-125,lon_max=-117,lat_min=65,lat_max=67,Delete_data=True)
    
                _, _, exclude_regional_6 = select_region(full_lon=SIC_lon_SMMR,full_lat=SIC_lat_SMMR,full_var=full_sic_arr[index],\
                                                                                lon_min=33,lon_max=45,lat_min=65,lat_max=67.5,Delete_data=True)
            
                all_exclude_regional = np.isfinite(exclude_regional_1).astype(int) + np.isfinite(exclude_regional_2).astype(int) + \
                                       np.isfinite(exclude_regional_3).astype(int) + np.isfinite(exclude_regional_4).astype(int) + \
                                       np.isfinite(exclude_regional_5).astype(int) + np.isfinite(exclude_regional_6).astype(int)
    
                filter_region = np.where(all_exclude_regional>0,np.nan,SIC_location[index])
                filter_masked_SIC = np.where(all_exclude_regional>0,np.nan,masked_SIC)
    
                _, _, SIC_polynyas_labeled[index] = select_region(full_lon=SIC_lon_SMMR,full_lat=SIC_lat_SMMR,full_var=filter_region,\
                                                                            lon_min=-180,lon_max=180,lat_min=65,lat_max=90,Delete_data=True)
    

                for_label_arr = np.nan_to_num(SIC_polynyas_labeled[index], nan=0)
                s = generate_binary_structure(2,2)
                daily_arr, daily_num = label(for_label_arr,structure=s)
                SIC_daily_num.append(daily_num)
                
                daily_area_sic = np.sum(cell_area_SMMR[SIC_polynyas_labeled[index]>0])/(1000**2)
                SIC_area.append(daily_area_sic)              
                
                for idx, _ in enumerate(full_sic_arr):
                    if np.any(np.isfinite(full_sic_arr[idx])==False):
                        SIC_polynyas_labeled[idx] = np.nan
            
            Daily_location = np.concatenate((Daily_location,SIC_polynyas_labeled),axis=0)

            SIC_winterly_area = np.nansum(SIC_area)
            SIC_base_loc = np.nansum(SIC_polynyas_labeled,axis=0)>0
    
            s = generate_binary_structure(2,2)
            SIC_base_loc_arr, SIC_base_num = label(SIC_base_loc,structure=s)
    
            SIC_yearly_base_loc_1978[j,i] = SIC_base_loc
            total_num_SIC[j,i] = np.sum(SIC_daily_num)
            total_area_SIC[j,i] = SIC_winterly_area
            total_area_SMMR[j,i] = np.nansum(cell_area_SMMR[SIC_base_loc_arr>0])/(1000**2)


    # Save polynya winter base map
    pathsave = '/proj/naiss2024-22-523/PhD_Year1/Revision_04092025/Results/'           # Path for saving results
    
    with nc.Dataset(f'{pathsave}SMMR_winter_basemap.nc', 'w', format='NETCDF4') as file: \
        # Create dimensions
        file.createDimension('conc', conc_SMMR.shape[0])
        file.createDimension('year', yr_1978.shape[0])
        file.createDimension('y', SIC_lat_SMMR.shape[0])
        file.createDimension('x', SIC_lat_SMMR.shape[1])
        
        time_var = file.createVariable('year', 'f4', ('year',))
        time_var[:] = yr_1978
    
        lat_var = file.createVariable('lat','f4',('y','x'))
        lat_var[:] = SIC_lat_SMMR 
        
        lon_var = file.createVariable('lon','f4',('y','x'))
        lon_var[:] = SIC_lon_SMMR
    
        SIC_winter_loc = file.createVariable('SIC_winter_loc','f4',('conc','year','y','x'))
        SIC_winter_loc[:] = SIC_yearly_base_loc_1978[:]

        
    import itertools
    flatten_date = list(itertools.chain.from_iterable(date_list))
    time_series = pd.Series(flatten_date)
    formatted_dates = pd.to_datetime(time_series.astype('Int64').astype(str),format='%Y%m%d',errors='coerce')
    formatted_dates_array = formatted_dates.dt.strftime('%Y-%m-%d').to_numpy()

    # Save daily polynya locations
    pathsave = '/proj/naiss2024-22-523/PhD_Year1/Revision_04092025/Results/'            # Path for saving results
    with nc.Dataset(f'{pathsave}daily_location_SMMR.nc', 'w', format='NETCDF4') as file: \
        # Create dimensions
        file.createDimension('time', Daily_location.shape[0] - 1)
        file.createDimension('y', Daily_location.shape[1])
        file.createDimension('x', Daily_location.shape[2])
        
        time_var = file.createVariable('time', str, ('time',))
        time_var[:] = formatted_dates_array.astype(np.bytes_)
    
        lat_var = file.createVariable('lat','f4',('y','x'))
        lat_var[:] = SIC_lat_SMMR 
        
        lon_var = file.createVariable('lon','f4',('y','x'))
        lon_var[:] = SIC_lon_SMMR 
    
        polynyas_loc_var = file.createVariable('polynyas_loc','f4',('time','y','x'))
        polynyas_loc_var[:] = Daily_location[1:,:,:] 
    
    # Save cumulative number, total area and cumulative area
    period_list = []
    
    for year in yr_1978:
        period_list.append('%s-%s'%(year,year+1))
    
    area_table = pd.DataFrame(data=total_area_SIC,index=conc_SMMR,columns=period_list)
    area_table_2 = pd.DataFrame(data=total_area_SMMR,index=conc_SMMR,columns=period_list)
    num_table = pd.DataFrame(data=total_num_SIC,index=conc_SMMR,columns=period_list)
    
    area_table.to_csv('/proj/naiss2024-22-523/PhD_Year1/Revision_04092025/Area_csv/Arctic_cumulative_area_1978_2024_SMMR.csv',sep=',')
    area_table_2.to_csv('/proj/naiss2024-22-523/PhD_Year1/Revision_04092025/Area_csv/Arctic_potential_area_1978_2024_SMMR.csv',sep=',')
    num_table.to_csv('/proj/naiss2024-22-523/PhD_Year1/Revision_04092025/Area_csv/Arctic_cumulative_num_1978_2024_SMMR.csv',sep=',')

