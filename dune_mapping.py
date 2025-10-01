from raster_data import DEM
from raster_data import treshold_data_criterion

from mpi4py import MPI
import glob

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

inputs=glob.glob(r'./DEM_tiles_*.tif')
chunk_size = len(inputs) // size

AI_model=r'./binary_input_with_troughs_based_model_trained_on_python_computed_LoG.h5'

def worker_function(file_path,method='AI',verbose=True):

    print("Process {} running over {} files".format(rank))

    output_pred_raw=file_path[:-4]+'_{}_predictions_raw.tif'.format(method)
    output_pred=file_path[:-4]+'_{}_predictions_postcut.tif'.format(method)
    trough_skeleton=file_path[:-4]+'_{}_trough_skeleton.tif'.format(method)
    crest_skeleton=file_path[:-4]+'_{}_crest_skeleton.tif'.format(method)

    DEM_test = DEM(from_path=True,input_path=file_path,data_criterion=treshold_data_criterion(-5),name="DEM",starting_visual=False,precompute=True,verbose=verbose)

    DEM_test.write_single_band_tif(file_path[:-4]+'_smoothed.tif')
    DEM_test.VOP.write_single_band_tif(file_path[:-4]+'_VOP.tif',dtype=rasterio.float32)
    DEM_test.VON.write_single_band_tif(file_path[:-4]+'_VON.tif',dtype=rasterio.float32)
    DEM_test.RR.write_single_band_tif(file_path[:-4]+'_RR.tif')
    DEM_test.LoG.write_single_band_tif(file_path[:-4]+'_LoG.tif')
    DEM_test.slopes.write_single_band_tif(file_path[:-4]+'_slopes.tif')
    DEM_test.VO.write_single_band_tif(file_path[:-4]+'_VO.tif',dtype=rasterio.float32)
    
    prediction_raster=DEM_test.dune_predictions(method=method,size_treshold=50+150*(method=='AI'),output_to_file=output_pred,AI_raw_output=output_pred_raw,AI_model=AI_model,save_skeleton=trough_skeleton,save_crests=crest_skeleton)

local_data = inputs[rank * chunk_size: (rank + 1) * chunk_size]

local_results = [worker_function(x) for x in inputs]

all_results = comm.gather(local_results, root=0)