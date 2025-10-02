import os
import rasterio
import netCDF4
from matplotlib.backends.backend_pdf import PdfPages

def file_name_converter(s):
    """
    
    Modifies a file path to create compatibility between different machines (Linux and Windows computers). Customize as needed by replacing in the source code:
        
        - "F-STG-25" with your MACHINE_NAME
        - 'F:' with the disk on which your project shows up
        - Add any other changes to the path as needed for your case
    """
    if s is not None:
        MACHINE_NAME = os.getenv("COMPUTERNAME") or os.getenv("HOSTNAME")
        if MACHINE_NAME == "F-STG-25":
            s=s.replace('/mnt','F:')
            s=s.replace(r'/','\\')
    return s

def my_savefig(fig,path,*args,**kwargs):
    """
    
    Wrapper for fig.savefig with file path compatibility between different machines

    Parameters:
    
        fig: figure to be saved 
        path: path for saving the figure, as it would be shown on the default machine of the project (internally modified using the file_name_converter function, as needed)

    
    """
    fig.savefig(file_name_converter(path),*args,**kwargs)

def my_open(path,*args,**kwargs):
    """
    
    Wrapper for open() with file path compatibility between different machines
   
    Parameters:
    
        path: path of the file to be opened, as it would be shown on the default machine of the project (internally modified using the file_name_converter function, as needed)
    
    """
    return open(file_name_converter(path),*args,**kwargs)

def my_rasterio_open(path,*args,**kwargs):
    """
    
    Wrapper for rasterio.open() with file path compatibility between different machines

    Parameters:
    
        path: path of the file to be opened, as it would be shown on the default machine of the project (internally modified using the file_name_converter function, as needed)
    
    """
    return rasterio.open(file_name_converter(path),*args,**kwargs)

def my_netCDF4_Dataset(path,*args,**kwargs):
    """
    
    Wrapper for netCDF4.Dataset() with file path compatibility between different machines

    Parameters:
    
        path: path of the file to be opened, as it would be shown on the default machine of the project (internally modified using the file_name_converter function, as needed)
    
    """
    return netCDF4.Dataset(file_name_converter(path),*args,**kwargs)

def my_PdfPages(path,*args,**kwargs):
    """
    
    Wrapper for PdfPages() with file path compatibility between different machines

    Parameters:
    
        path: path of the pdf file to be created, as it would be shown on the default machine of the project (internally modified using the file_name_converter function, as needed)
    
    """
    return PdfPages(file_name_converter(path),*args,**kwargs)