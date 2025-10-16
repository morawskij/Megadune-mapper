import os
import rasterio
import netCDF4
import xarray as xr
from matplotlib.backends.backend_pdf import PdfPages
from keras.models import Model

def file_name_converter(s):
    """
    
    Modifies a file path to create compatibility between different machines (Linux and Windows computers). Customize as needed by replacing in the source code:
        
        - "F-STG-25" with your MACHINE_NAME
        - 'F:' with the disk on which your project shows up
        - Add any other changes to the path as needed for your case

    Parameters:
    
        s: file path to be modified 
        
    """
    if s is not None:
        MACHINE_NAME = os.getenv("COMPUTERNAME") or os.getenv("HOSTNAME")
        if MACHINE_NAME == "F-STG-25":
            s=s.replace('/mnt','F:')
            s=s.replace('/','\\')
        elif MACHINE_NAME == "ADRIAN":
            s=s.replace('/mnt','D:')
            s=s.replace('/','\\')
    return s

def my_savefig(fig,path,*args,**kwargs):
    """
    
    Wrapper for fig.savefig() with file path compatibility between different machines

    Parameters:
    
        fig: figure to be saved 
        path: path for saving the figure, as it would be shown on the default machine of the project (internally modified using the file_name_converter function, as needed)
        *args: to be passed onto fig.savefig()
        **kwargs: to be passed onto fig.savefig()
    
    """
    fig.savefig(file_name_converter(path),*args,**kwargs)

def my_open(path,*args,**kwargs):
    """
    
    Wrapper for open() with file path compatibility between different machines
   
    Parameters:
    
        path: path of the file to be opened, as it would be shown on the default machine of the project (internally modified using the file_name_converter function, as needed)
        *args: to be passed onto open()
        **kwargs: to be passed onto open()
    
    """
    return open(file_name_converter(path),*args,**kwargs)

def my_rasterio_open(path,*args,**kwargs):
    """
    
    Wrapper for rasterio.open() with file path compatibility between different machines

    Parameters:
    
        path: path of the file to be opened, as it would be shown on the default machine of the project (internally modified using the file_name_converter function, as needed)
        *args: to be passed onto rasterio.open()
        **kwargs: to be passed onto rasterio.open()
    
    """
    return rasterio.open(file_name_converter(path),*args,**kwargs)

def my_netCDF4_Dataset(path,*args,**kwargs):
    """
    
    Wrapper for netCDF4.Dataset() with file path compatibility between different machines

    Parameters:
    
        path: path of the file to be opened, as it would be shown on the default machine of the project (internally modified using the file_name_converter function, as needed)
        *args: to be passed onto netCDF4.Dataset()
        **kwargs: to be passed onto netCDF4.Dataset()
        
    """
    return netCDF4.Dataset(file_name_converter(path),*args,**kwargs)

def my_PdfPages(path,*args,**kwargs):
    """
    
    Wrapper for PdfPages() with file path compatibility between different machines

    Parameters:
    
        path: path of the pdf file to be created, as it would be shown on the default machine of the project (internally modified using the file_name_converter function, as needed)
        *args: to be passed onto PdfPages()
        **kwargs: to be passed onto PdfPages()
    
    """
    return PdfPages(file_name_converter(path),*args,**kwargs)

def my_xr_open_dataset(path,*args,**kwargs):
    """
    
    Wrapper for PdfPages() with file path compatibility between different machines

    Parameters:
    
        path: path of the pdf file to be created, as it would be shown on the default machine of the project (internally modified using the file_name_converter function, as needed)
        *args: to be passed onto PdfPages()
        **kwargs: to be passed onto PdfPages()
    
    """
    return xr.open_dataset(file_name_converter(path),*args,**kwargs)

def my_to_netcdf(ds,path,*args,**kwargs):
    """
    
    Wrapper for to_netcdf() from xarray package, with file path compatibility between different machines

    Parameters:

        ds: xarray.Dataset object to be exported to netcdf
        path: path of the pdf file to be created, as it would be shown on the default machine of the project (internally modified using the file_name_converter function, as needed)
        *args: to be passed onto PdfPages()
        **kwargs: to be passed onto PdfPages()
    
    """
    ds.to_netcdf(file_name_converter(path),*args,**kwargs)

def my_model_load_weights(model,path,*args,**kwargs):
    """
    
    Wrapper for keras.Model.load_weights(), with file path compatibility between different machines

    Parameters:

        model: keras Model object
        path: path of the file with weights to load
        *args: to be passed onto load_weights()
        **kwargs: to be passed onto load_weights()
    
    """
    model.load_weights(file_name_converter(path),*args,**kwargs)    
