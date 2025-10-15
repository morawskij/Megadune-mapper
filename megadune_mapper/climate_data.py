import numpy as np
import matplotlib.pyplot as plt
from megadune_mapper.raster_data import raster_data
from megadune_mapper.file_path_fixing import my_savefig, my_open, my_rasterio_open, my_netCDF4_Dataset, my_xr_open_dataset, my_to_netcdf
import rasterio
from pyproj import Transformer
import time
from affine import Affine
import datetime as dt
import netCDF4
import numpy as np
from tqdm import tqdm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import rioxarray
import xarray as xr
import dask.array as daskar
from dask.diagnostics import ProgressBar

#### WEIRD ISSUES JUMPING BETWEEN ERA AND MAR, INVESTIGATE

def visual_debugger(data,title):
    fig,ax=plt.subplots(figsize=(12,8))
    sc=ax.imshow(data)
    ax.set_title(title)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="5%", pad=0.05)
    bar = fig.colorbar(sc,cax=cax,orientation='horizontal')

def large_array_multiply(A,B,track_progress=False):
    A_d = daskar.from_array(A, chunks=(1, 512, 512))
    B_d = daskar.from_array(B, chunks=(1, 512, 512))
    C_d = A_d * B_d
    if track_progress:
        with ProgressBar():
            return C_d.compute()
    else:
        return C_d.compute()

def generate_x_y_arrays(x0,dx,nx,y0,dy,ny):

    return np.tile((x0+dx*np.arange(nx)).reshape(nx,1),ny).T,np.tile((y0+dy*np.arange(ny)).reshape(ny,1),nx)

def coordinate_correspondence(x,x0,dx,x1,interpolate=True,cyclic=False):
    if (x-x0)*(x-x1)>=0:
        if abs(x-x0)<abs(x-x1):
            if cyclic and abs(x-x0)<abs(dx):
                pos=(x-x0)/dx
                flpos=np.floor(pos)
                frpos=pos-flpos  
                if interpolate:
                    return 0, frpos, -1
                elif frpos<=0.5:
                    return -1, 1, 0
                else:
                    return 0, 1, -1
            else:
                return 0, 1, 1
        elif cyclic and abs(x-x1)<abs(dx):
                pos=(x-x1)/dx
                flpos=np.floor(pos)
                frpos=pos-flpos     
                if interpolate:
                    return -1, 1-frpos, 0
                elif frpos<=0.5:
                    return -1, 1, 0
                else:
                    return 0, 1, -1                       
        else:
            return -1, 1, -2
    else:
        pos=(x-x0)/dx
        flpos=np.floor(pos)
        frpos=pos-flpos
        if interpolate:
            return int(flpos), 1-frpos, int(flpos)+1
        elif frpos<=0.5:
            return int(flpos), 1, int(flpos)+1
        else:
            return int(flpos), 0, int(flpos)+1

def coordinate_correspondence_ar(x,x0,dx,x1,interpolate=True,cyclic=False):
    ar_ind_1=np.zeros_like(x,dtype=np.int32)
    ar_alph=np.zeros_like(x,dtype=np.float64)
    ar_ind_2=np.zeros_like(x,dtype=np.int32)
    for idx in tqdm(np.ndindex(x.shape)):
        a, b, c = coordinate_correspondence(x[idx],x0,dx,x1,interpolate=interpolate,cyclic=cyclic)
        ar_ind_1[idx]=a
        ar_alph[idx]=b
        ar_ind_2[idx]=c
    return     ar_ind_1, np.expand_dims(ar_alph,axis=0), ar_ind_2

class climate_data:

    def __init__(self,file,values=None,time_variable_name=None,indices_of_interest=None,start_date=None,spacing_is_regular=True,time_array=None,crs="EPSG:3031",x_variable_name=None,y_variable_name=None,x0=None,dx=None,x1=None,y0=None,dy=None,y1=None,deltat=None,distrust_time=False):

        if x_variable_name is None:
            self.time_variable_name='irrelevant'
            self.start_date=None
        
        self.x_variable_name=x_variable_name
        self.y_variable_name=y_variable_name

        
        
        if values is not None:
            self.time_variable_name = time_variable_name
            self.time_array=time_array
            self.time_dim=len(self.time_array)
            self.indices=values.keys()
            self.start_date=start_date
            self.values=values
            self.spacing_is_regular=spacing_is_regular
            self.x_variable_name=x_variable_name
            self.y_variable_name=y_variable_name
            self.x0=x0
            self.dx=dx
            self.x1=x1
            self.y0=y0
            self.dy=dy
            self.y1=y1
            self.nx=int((x1-x0)/dx)+1
            self.ny=int((y1-y0)/dy)+1
            self.deltat=deltat

        else:
            file_temp = my_netCDF4_Dataset(file, 'r')
    
            if time_variable_name is None:
                for key in file_temp.variables.keys():
                    if 'time' in key.lower():
                        self.time_variable_name=key
                        break
            else:
                self.time_variable_name=time_variable_name

            if x_variable_name is None:
                for key in file_temp.variables.keys():
                    if 'x' in key.lower() or 'lat' in key.lower():
                        self.x_variable_name=key
                        break
            else:
                self.x_variable_name=x_variable_name
    
            if self.x_variable_name is None:
                self.x_variable_name='undefined'
                print('WARNING: x variable not defined and could not be identified automatically. Will not be able to generate georeferenced rasters.')

            else:
                self.x0=file_temp[self.x_variable_name][0]*(1+999*(file_temp.variables[self.x_variable_name].units=='km'))
                self.dx=file_temp[self.x_variable_name][1]*(1+999*(file_temp.variables[self.x_variable_name].units=='km'))-self.x0
                self.x1=file_temp[self.x_variable_name][-1]*(1+999*(file_temp.variables[self.x_variable_name].units=='km'))
                self.nx=len(file_temp[self.x_variable_name])
            
            if y_variable_name is None:
                for key in file_temp.variables.keys():
                    if 'y' in key.lower() or 'lon' in key.lower():
                        self.y_variable_name=key
                        break
            else:
                self.y_variable_name=y_variable_name
    
            if self.y_variable_name is None:
                self.y_variable_name='undefined'
                print('WARNING: y variable not defined and could not be identified automatically. Will not be able to generate georeferenced rasters.')
            else:
                self.y0=file_temp[self.y_variable_name][0]*(1+999*(file_temp.variables[self.y_variable_name].units=='km'))
                self.dy=file_temp[self.y_variable_name][1]*(1+999*(file_temp.variables[self.y_variable_name].units=='km'))-self.y0     
                self.y1=file_temp[self.y_variable_name][-1]*(1+999*(file_temp.variables[self.y_variable_name].units=='km'))        
                self.ny=len(file_temp[self.y_variable_name])
           
            if self.time_variable_name is None:
                self.time_variable_name='irrelevant'
                self.start_date=None
    
            else:
                nctime=file_temp[self.time_variable_name][:]
                t_cal=file_temp[self.time_variable_name].calendar
                t_unit = file_temp.variables[self.time_variable_name].units
                self.time_array=netCDF4.num2date(nctime,units = t_unit,calendar = t_cal)     
                self.time_dim=len(self.time_array)
                self.start_date=self.time_array[0]
                dt0=self.time_array[1]-self.time_array[0]
                if distrust_time:
                    self.time_array=self.start_date+dt0*np.arange(self.time_dim)
                if all(self.time_array[1:]-self.time_array[:-1] == dt0):
                    self.spacing_is_regular=True
                    self.deltat=dt0
                else:
                    self.spacing_is_regular=False
                    self.deltat=None
    
            if indices_of_interest is None:
                
                self.indices=list(file_temp.variables.keys())
                if not self.time_variable_name=='irrelevant':
                    self.indices.remove(self.time_variable_name)
            
            else:
    
                self.indices=[]
                if isinstance(indices_of_interest,str):
                    indices_of_interest=[indices_of_interest]
                for ind in indices_of_interest:
                    if ind in file_temp.variables.keys():
                        self.indices.append(ind)
                    else:
                        print("Warning: indice {} absent in the data, omitting".format(ind))
    
            self.values={}
            for ind in self.indices:
                self.values[ind]=file_temp.variables[ind][:]

        self.crs=crs

    def export_as_netcdf(self,out_path,ind=None,description='Exported NetCDF'):
        if ind is None:
            ind=self.indices
        elif isinstance(ind,str):
            ind=[ind]

        x = np.linspace(self.x0, self.x1, self.nx)
        y = np.linspace(self.y0, self.y1, self.ny)
        
        time = np.arange(self.time_dim)
        data_vars = {}
        for name in ind:
            if self.crs=='EPSG:4326':
                data_vars[name] = (("time", "x","y"), self.values[name])
            else:
                data_vars[name] = (("time", "y","x"), self.values[name])
        ds = xr.Dataset(data_vars=data_vars,
        coords={
            "x": ("x", x),
            "y": ("y", y),
            "time": ("time", time)
        },attrs={"description": description})

        if self.crs=='EPSG:4326':
            ds["x"].attrs.update({"units": "degrees_east", "axis": "X", "long_name": "longitude"})
            ds["y"].attrs.update({"units": "degrees_north", "axis": "Y", "long_name": "latitude"})
        else:
            ds["x"].attrs.update({"units": "m", "axis": "X"})
            ds["y"].attrs.update({"units": "m", "axis": "Y"})
        
        left = float(np.min(x))           
        top = float(np.max(y))
        transform = rasterio.transform.from_origin(left - self.dx / 2.0, top + self.dy / 2.0, self.dx, self.dy)

        for name in ind:
            da = ds[name]      
            da = da.rio.set_spatial_dims(x_dim="x", y_dim="y")
            da.rio.write_crs(self.crs, inplace=True)
            da.rio.write_transform(transform, inplace=True)
            ds[name] = da
        
        ds = ds.rio.write_crs(self.crs)
        my_to_netcdf(ds,out_path, engine="netcdf4")  
        """
        ds = my_xr_open_dataset(out_path, decode_coords="all",engine='scipy')
        ds = ds.rio.write_crs(self.crs, inplace=True)
        #root_grp = my_netCDF4_Dataset(out_path, 'w', format='NETCDF4')
        ds.to_netcdf(out_path, engine='scipy')
        ds.description = description

        ds.createDimension('time', self.time_dim)
        ds.createDimension('x', self.nx)
        ds.createDimension('y', self.ny)
        time = ds.createVariable('time', 'f8', ('time',))
        x = ds.createVariable('x', 'f4', ('x',))
        y = ds.createVariable('y', 'f4', ('y',))
        x[:] =  np.linspace(self.x0, self.x1, self.nx)
        y[:] =  np.linspace(self.y0, self.y1, self.ny) 
        time[:] = np.arange(self.time_dim)
        data={}
        for indd in ind:
            data[indd] = ds.createVariable(indd, 'f8', ('time', 'y', 'x',))
            data[indd][:,:,:] = self.values[indd]

        ds.close()    
"""

    def export_raster(self,ind,output_path,time_ind=None,time=None,time_axis=0):
        if time_ind is None:
            if time is not None:
                time_ind=0
                if time>=self.start_date:
                    while time>=self.time_array[time_ind+1]:
                        time_ind+=1
                        if time_ind == self.time_dim-1:
                            break
        if time_ind is None:
            for_export=raster_data(raster=self.values[ind],x0=self.x0,y0=self.y0,dx=self.dx,dy=self.dy,crs=self.crs)
        else:
            for_export=raster_data(raster=np.take(self.values[ind],time_ind,axis=time_axis),x0=self.x0,y0=self.y0,dx=self.dx,dy=self.dy,crs=self.crs)
        for_export.write_single_band_tif(output_path)
        
    def aggregate_all(self,ind=None,time_axis=0,monthly_weights=False,yearly_weights=False,custom_fun=None):
        if isinstance(ind, str):
            ind=[ind]
        elif ind is None:
            ind=self.indices
        new_values={}

        if custom_fun is not None:
            for indd in ind:
                new_values[indd]=custom_fun(self.values[indd],axis=time_axis)            
        elif self.spacing_is_regular:
            for indd in ind:
                new_values[indd]=np.mean(self.values[indd],axis=time_axis)
        else:
            weights=list(map(lambda x:x.total_seconds()/(24*3600), self.time_array[1:]-self.time_array[:-1]))
            if yearly_weights:
               if self.time_array[-1].year%4==0 and (self.time_array[-1].year%100!=0 or self.time_array[-1].year%400==0):
                   weights.append(366)
               else:
                   weights.append(365)
            elif monthly_weights:
               if self.time_array[-1].month in [1,3,5,7,8,10,12]:
                   weights.append(31)
               elif self.time_array[-1].month==2:
                   if self.time_array[-1].year%4==0 and (self.time_array[-1].year%100!=0 or self.time_array[-1].year%400==0):
                       weights.append(29)
                   else:
                       weights.append(28)
               else:
                   weights.append(30)
            else:
                weights.append(np.mean(np.array(weights)))
            for indd in ind:
                new_values[indd]=np.average(self.values[indd],axis=time_axis,weights=np.array(weights))

        return climate_data(None,values=new_values,time_variable_name='irrelevant',crs=self.crs,x_variable_name=self.x_variable_name,y_variable_name=self.y_variable_name,x0=self.x0,dx=self.dx,y0=self.y0,dy=self.dy,x1=self.x1,y1=self.y1,time_array=np.array([self.start_date]))

    def ranges_all(self,ind=None,time_axis=0):
        return self.aggregate_all(ind=ind,time_axis=time_axis,custom_fun=np.ptp)

    def std_all(self,ind=None,time_axis=0):
        return self.aggregate_all(ind=ind,time_axis=time_axis,custom_fun=np.std)
    
    def aggregate_month_by_month(self,ind=None,time_axis=0):
        if isinstance(ind, str):
            ind=[ind]
        elif ind is None:
            ind=self.indices
        new_values={}
        for indd in ind:
            new_values[indd]=[]
        new_time_array=[]
        time_ind=0
        time_ind_0=0
        print(self.time_array.shape,self.time_dim)
        while time_ind<self.time_dim:
            t0=self.time_array[time_ind_0]
            while self.time_array[time_ind].month==t0.month:
                time_ind+=1
                if time_ind>=self.time_dim:
                    break
            new_time_array.append(dt.datetime(year=t0.year,month=t0.month,day=1))
            for indd in ind:
                new_values[indd].append(np.mean(self.values[indd][time_ind_0:time_ind],axis=time_axis))
            time_ind_0=time_ind   
        for indd in ind:
            new_values[indd]=np.array(new_values[indd])
        return climate_data(None,values=new_values,time_variable_name=self.time_variable_name,crs=self.crs,x_variable_name=self.x_variable_name,y_variable_name=self.y_variable_name,x0=self.x0,dx=self.dx,y0=self.y0,dy=self.dy,x1=self.x1,y1=self.y1,start_date=new_time_array[0],time_array=new_time_array)                   
            
    def aggregate_months_all_time(self,ind=None):
        if isinstance(ind, str):
            ind=[ind]
        elif ind is None:
            ind=self.indices
        new_values={}
        for indd in ind:
            new_values[indd]=[[] for _ in range(12)]
            for i in range(self.time_dim):
                new_values[indd][self.time_array[i].month-1]+=[self.values[indd][i]]
            new_values[indd]=np.array(list(map(lambda x: np.mean(np.array(x),axis=0),new_values[indd])))
        return climate_data(None,values=new_values,time_variable_name='irrelevant',crs=self.crs,x_variable_name=self.x_variable_name,y_variable_name=self.y_variable_name,x0=self.x0,dx=self.dx,y0=self.y0,dy=self.dy,x1=self.x1,y1=self.y1,start_date=1,time_array=np.arange(1,13))      


    def extract_time_span(self,t0,t1,implicit=True):
        extractor = np.where(np.logical_and(self.time_array>=t0,self.time_array<=t1),True,False)
        new_time_array=np.extract(extractor,self.time_array)
        extractor=np.tile(extractor,np.prod(self.values[self.indices[0]].shape[1:])).reshape(self.values[self.indices[0]].shape)
        new_values={}
        for indd in self.indices:
            new_values[indd]=np.extract(extractor,self.values[indd]).reshape(np.concatenate((new_time_array.shape,self.values[indd].shape[1:])))
        if implicit:
            self.values=new_values
            self.start_date=min(new_time_array)
            self.time_array=new_time_array
            self.time_dim=len(new_time_array)
        else:
            return climate_data(None,values=new_values,time_variable_name=self.time_variable_name,crs=self.crs,x_variable_name=self.x_variable_name,y_variable_name=self.y_variable_name,x0=self.x0,dx=self.dx,y0=self.y0,dy=self.dy,x1=self.x1,y1=self.y1,start_date=min(new_time_array),time_array=new_time_array)    

    def get_x_y_arrays(self):
        return generate_x_y_arrays(self.x0,self.dx,self.nx,self.y0,self.dy,self.ny)
    
    def get_mean_values_x_y(self,ind=None,partner=None,x0=None,dx=None,nx=None,y0=None,dy=None,ny=None,partner_crs=None,interpolate=True,x_variable_name='x',y_variable_name='y'):
        if ind is None:
            ind=self.indices
        if partner is not None:
            partner_x,partner_y = partner.get_x_y_arrays()
            partner_crs = partner.crs
        else:
            partner_x,partner_y = generate_x_y_arrays(x0,dx,nx,y0,dy,ny)
        transformer=Transformer.from_crs(partner_crs, self.crs)
        my_x, my_y = transformer.transform(partner_x,partner_y)
        x_ind_1, alpha, x_ind_2 = coordinate_correspondence_ar(my_x,self.x0,self.dx,self.x1,interpolate=interpolate)
        y_ind_1, beta, y_ind_2 = coordinate_correspondence_ar(my_y,self.y0,self.dy,self.y1,interpolate=interpolate,cyclic=(self.crs=="EPSG:4326"))
        if isinstance(ind,str):
            ind=[ind]
        output={}
        for indd in ind:
            output[indd]=large_array_multiply(alpha*beta,self.values[indd][:,x_ind_1,y_ind_1])+large_array_multiply(alpha*(1-beta),self.values[indd][:,x_ind_1,y_ind_2])+large_array_multiply((1-alpha)*(1-beta),self.values[indd][:,x_ind_2,y_ind_2])+large_array_multiply((1-alpha)*beta,self.values[indd][:,x_ind_2,y_ind_1])
        if partner is None:
            return climate_data(None,values=output,time_variable_name=self.time_variable_name,crs=partner_crs,x_variable_name=x_variable_name,y_variable_name=y_variable_name,x0=x0,dx=dx,y0=y0,dy=dy,x1=x0+(nx-1)*dx,y1=y0+(ny-1)*dy,start_date=self.start_date,time_array=self.time_array)
        else:
            return output

    def adopt_from_partner(self,partner,ind=None,interpolate=True):
        if self.time_dim == partner.time_dim:
            if ind is None:
                ind=partner.indices
            self.values = {**self.values, **partner.get_mean_values_x_y(partner=self,ind=ind,interpolate=interpolate)}
            self.indices=self.values.keys()
        else:
            print("ABORTING: Inconsistent dimensions - match time arrays before reapplying adoption, for example by monthly aggregation and time span extraction")
    
    def calculate_densimetric_froude(self,dT=10,T_a=220,g=9.81):
        if dT is None:
            dT=self.values['t2m']-T_a
        self.values['Fr_d']=self.values['si10']/(self.values['HBL']*g*dT/self.values['t2m'])**0.5
        

    def rename_variable(self,old_name,new_name):
        self.indices=list(self.indices)
        self.indices.remove(old_name)
        self.indices.append(new_name)
        self.values[new_name] = self.values.pop(old_name)
        
        
  
            
            
        
        
