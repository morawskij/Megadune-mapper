import numpy as np
import rasterio
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import matplotlib.cm
from matplotlib.path import Path
from matplotlib.patches import Arc, PathPatch, Circle
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time
import scipy.ndimage as nd
from skimage import morphology
from rasterio import features
from shapely.geometry import shape
from shapely.geometry import LineString, MultiLineString, Point, MultiPoint, Polygon
import shapely.plotting as shplt
import shapely
import networkx as nx
from itertools import combinations
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pyproj import Transformer, crs
from affine import Affine
from tqdm import tqdm
from megadune_mapper.raster_data import raster_data
from sklearn.decomposition import PCA
import scipy.signal as scp
from decimal import Decimal, getcontext
from matplotlib.backends.backend_pdf import PdfPages
import datetime as dt
import netCDF4
from scipy.stats import gaussian_kde
from megadune_mapper.file_path_fixing import my_savefig, my_open, my_rasterio_open, my_netCDF4_Dataset, my_PdfPages
from rasterstats import zonal_stats


def _convert_time(Times):
    atmos_epoch = dt.datetime(1970, 1, 1, 0, 0, tzinfo=dt.timezone.utc)
    # convert array of times in hours from epoch to dates
    return np.array([atmos_epoch + dt.timedelta(seconds=i) for i in Times])

def load_netcdf(file):

    Data = {}
    file_temp = my_netCDF4_Dataset(file, 'r')
    for key in file_temp.variables.keys():
        Data[key] = file_temp.variables[key][:]
    if 'valid_time' in Data.keys():
        Data['time'] = _convert_time(Data['valid_time'].astype(np.float64))
    return Data

def string_for_label_fixer(s):
    if s[-5:]=="[km2]":
        return s[:-5]+"[km"r'$^{2}$'"]"
    elif s=="Angle with respect to wind [deg]":
        return "Wind incidence angle "+"["r'$^\circ$'"]"
    elif s[-5:]=="[deg]":
        return s[:-5]+"["r'$^\circ$'"]"
    elif s[-5:]=="m/day":
        return s[:-5]+r'$\left[\frac{m}{day}\right]$'
    elif s[-8:]=="[mm/day]":
        return s[:-8]+r'$\left[\frac{mm}{day}\right]$'
    elif s[-5:]=="speed":
        return s[:4]+" "+s[5:]+r' $\left[\frac{m}{s}\right]$'
    elif s[:6]=="Cross ":
        return "Cross-"+s[6:]
    elif s=="Froude number":
        return r'$Fr_d$'" proxy"
    else: 
        return s


def convert_ticks(x1,x2):
    xn1=int(np.ceil(x1))
    xn2=int(np.floor(x2))
    if (xn1>xn2):
        a1=10**(xn1-1)
        xn12=2*a1
        xn22=9*a1
        i1=2
        while xn12<10**x1:
            xn12+=a1
            i1+=1
        i2=9
        while xn22>10**x2:
            i2-=1
            xn22-=a1
        lbl1=r'${}\cdot10^{}{}{}$'.format(i1,"{",xn2,"}")
        lbl2=r'${}\cdot10^{}{}{}$'.format(i2,"{",xn2,"}")
        xn12=np.log10(xn12)
        xn22=np.log10(xn22)
    if (xn1==xn2):
        a1=10**(xn1-1)
        a2=10**xn1
        xn12=2*a1
        xn22=9*a2
        i1=2
        while xn12<10**x1:
            xn12+=a1
            i1+=1
        i2=9
        while xn22>10**x2:
            i2-=1
            xn22-=a2
        lbl1=r'${}\cdot10^{}{}{}$'.format(i1,"{",xn1-1,"}")
        lbl2=r'${}\cdot10^{}{}{}$'.format(i2,"{",xn1,"}")
        xn12=np.log10(xn12)
        xn22=np.log10(xn22)
    else:
        dxn=int((xn2-xn1)/3)
        xn12=xn1+dxn
        lbl1=r'$10^{}{}{}$'.format("{",xn12,"}")
        xn22=xn2-dxn
        lbl2=r'$10^{}{}{}$'.format("{",xn22,"}")        
    return [xn12,xn22], [lbl1,lbl2]

def string_for_filename(string):
    i=0
    n=len(string)
    while i<n:
        if (string[i]==" "):
            string=string[:i]+"_"+string[i+1:]
        i+=1
    return string

class wind_data_for_hex_vis:

    def __init__(self,file,crs="EPSG:3031"):
        wind_data=load_netcdf(file)
        self.u_data=wind_data["u10"]
        self.v_data=wind_data["v10"]
        self.lats=wind_data["latitude"]
        self.lons=wind_data["longitude"]
        #self.u=np.mean(wind_data["u10"],axis=0)
        #self.v=np.mean(wind_data["v10"],axis=0)
        self.transformer=Transformer.from_crs(crs, "EPSG:4326")
        self.inverse_transformer=Transformer.from_crs("EPSG:4326",crs)

    def draw_wind_rose(self,polygon,nbins_theta=72,speed_bins_spacing=3,speedbin_max=21,fig=None,ax=None,figsize=(8,8)):
        if fig is None or ax is None:
            fig,ax=plt.subplots(subplot_kw={'projection': 'polar'},figsize=figsize)
        speed_bins=np.linspace(0,speedbin_max,int(speedbin_max/speed_bins_spacing))
        l=polygon.bounds[0]
        d=polygon.bounds[1]
        r=polygon.bounds[2]
        u=polygon.bounds[3]
        coords=np.array(list(map(lambda x: self.transformer.transform(x[0],x[1]),[[l,d],[l,u],[r,d],[r,u]])))
        lat_min=np.min(coords[:,0])
        lat_max=np.max(coords[:,0])
        lon_min=np.min(coords[:,1])
        lon_max=np.max(coords[:,1])
        l_ind=int((-60-round(lat_max*4)/4)*4)
        r_ind=int((-60-round(lat_min*4)/4)*4)
        d_ind=int((180+round(lon_min*4)/4)*4)
        u_ind=int((180+round(lon_max*4)/4)*4)
        lats=self.lats[l_ind:r_ind+1]
        if u_ind-d_ind>=720:
            if u_ind==1440:
                lons=self.lons[0:d_ind]
                us=self.u_data[:,l_ind:r_ind+1,0:d_ind]
                vs=self.v_data[:,l_ind:r_ind+1,0:d_ind]
            else:
                lons=np.concatenate((self.lons[u_ind:],self.lons[0:d_ind]))
                us=np.concatenate((self.u_data[:,l_ind:r_ind+1,u_ind:],self.u_data[:,l_ind:r_ind+1,0:d_ind]),axis=2)
                vs=np.concatenate((self.v_data[:,l_ind:r_ind+1,u_ind:],self.u_data[:,l_ind:r_ind+1,0:d_ind]),axis=2)
        else:
            lons=np.concatenate((self.lons[d_ind:min(u_ind+1,1440)],self.lons[0:max(0,u_ind-1439)]))
            us=np.concatenate((self.u_data[:,l_ind:r_ind+1,d_ind:min(u_ind+1,1440)],self.u_data[:,l_ind:r_ind+1,0:max(0,u_ind-1439)]),axis=2)
            vs=np.concatenate((self.v_data[:,l_ind:r_ind+1,d_ind:min(u_ind+1,1440)],self.v_data[:,l_ind:r_ind+1,0:max(0,u_ind-1439)]),axis=2)    
        LONs, LATs = np.meshgrid(lons,lats)
        #lats=np.concatenate((self.lats[l_ind:r_ind+1,d_ind:u_ind+1],self.lats[l_ind:r_ind+1,0:u_ind2]),axis=1)
        #lons=np.concatenate((self.lons[l_ind:r_ind+1,d_ind:u_ind+1],self.lons[l_ind:r_ind+1,0:u_ind2]),axis=1)

        epsg=self.inverse_transformer.transform(LATs,LONs)#np.concatenate((LATs.reshape((LATs.shape[0],LATs.shape[1],1)),LONs.reshape((LONs.shape[0],LONs.shape[1],1))),axis=2))
        epsg=np.concatenate((epsg[0].reshape((LATs.shape[0],LATs.shape[1],1)),epsg[1].reshape((LATs.shape[0],LATs.shape[1],1))),axis=2)
        points=shapely.points(epsg[:,:,0],epsg[:,:,1])
        mask = shapely.within(points,polygon) 
        LONs_rad=np.radians(LONs)
        cosLON=np.cos(LONs_rad)
        sinLON=np.sin(LONs_rad)
        y2=vs*cosLON-us*sinLON
        x2=us*cosLON+vs*sinLON
        angles=np.arctan2(y2,x2)%(2*np.pi)
        speeds=(us**2+vs**2)**0.5
        max_speed=np.max(speeds)
        speed_ind=int(np.ceil(max_speed/speed_bins_spacing))
        theta=np.linspace(0,2*np.pi,nbins_theta+1)
        for j in range(nbins_theta):
            num,speed=np.histogram(np.extract(np.logical_and(mask,(angles-theta[j])*(angles-theta[j+1])<0),speeds),bins=speed_bins[:speed_ind])
            num_sum=0
            for k in range(len(speed)-1):
                ax.bar((theta[j]+theta[j+1])/2.0, num_sum+num[k], width=2*np.pi/nbins_theta, bottom=num_sum,color=plt.cm.plasma((k+0.5)/len(speed_bins)),label='{}{:.0f} - {:.0f} 'r'$\frac{}$'.format("_"*j,k*speed_bins_spacing,(k+1)*speed_bins_spacing,"{m}{s}"))
                num_sum+=num[k]
        ylim=ax.get_ylim()[1]
        """
        dir_mean=np.arctan2(np.mean(np.extract(mask,y2)),np.mean(np.extract(mask,x2)))%(2*np.pi)
        ax.annotate("", xy=(dir_mean,0.7*ylim), xytext=(dir_mean+np.pi,0.025*ylim),arrowprops=dict(arrowstyle="simple"))
        """
        #ax.legend()
        ax.set_yticklabels([])
        ax.set_xticklabels([])

def precise_addition(ar1,ar2,prec=50):
    getcontext().prec = prec
    if isinstance(ar1,float) and isinstance(ar2,float):
        ar1=Decimal(str(ar1))
        ar1+=Decimal(str(ar2))
        return float(ar1)
    elif isinstance(ar1,float):
        for idx in np.ndindex(ar2.shape):
            val=Decimal(str(ar2[idx]))
            val+=Decimal(str(ar1))
            ar2[idx]=float(val)
        return ar2
    elif isinstance(ar2,float):
        return precise_addition(ar2,ar1,prec=prec)
    else:
        for idx in np.ndindex(ar2.shape):
            val=Decimal(str(ar1[idx]))
            val+=Decimal(str(ar2[idx]))
            ar1[idx]=float(val)
        return ar1        
    

def crest_vertical_projection(x1,x2,x3,y1,y2,y3,z1,z2,z3):

    if z1==z2:
        z0=z1
        x0=x3
        y0=y3
        dz=0
    else:
        dz=z1-z2
        dy=y1-y2
        dx=x1-x2
        z0=(z3*dz**2+z2*(dx**2+dy**2)+(x3-x2)*dx*dz+(y3-y2)*dy*dz)/(dz**2+dx**2+dy**2)
        
        x0=precise_addition(dx*(z0-z1)/dz,x1)
        x0=np.where((x0-x1)*(x0-x2)>0,precise_addition(dx*(z0-z2)/dz,x2),x0)
            
        y0=precise_addition(dy*(z0-z1)/dz,y1)
        y0=np.where((y0-y1)*(y0-y2)>0,precise_addition(dy*(z0-z2)/dz,y2),y0)

    h=((z3-z0)**2+(x3-x0)**2+(y3-y0)**2)**0.5
    a=((z1-z0)**2+(x1-x0)**2+(y1-y0)**2)**0.5
    b=((z2-z0)**2+(x2-x0)**2+(y2-y0)**2)**0.5

    return x0,y0,z0,dz,h,a,b

class region_for_study:

    def __init__(self,name,geom,dunes):
        self.name=name
        self.geom=geom
        cropped=dune_polygons(None,None,None,None,None,for_copy=dunes)
        dunes=cropped.limit_to_ROI(self.geom)
        self.dunes=dunes
        self.num_dunes=len(self.dunes.polygons)

class regions_for_study:

    def __init__(self,path,dunes):
        if path is None:
            self.regions=[]
        else:
            file=my_open(path,'r').readlines()[1:]
            self.regions=[]
            for l in file:
                info=l.split(";")
                self.regions.append(region_for_study(info[1][:-2],shapely.from_wkt(info[0][1:-1]),dunes))
                                
class one_dune_polygon:

    def __init__(self,polygon,WOI,crestlines,troughlines,transform,DEM_path,wind_data_paths,preprocess_buffer_fac=0.05,sampling_resolution=500,all_given=None,investigate_suspicious=False,suspicious_plot_count=0,flux_data=[None,None],transformer=None,Mars=False):

        self.suspicious_plot_count=suspicious_plot_count        
        if all_given is None:
            self.morphological_indices={}
            self.sampling_resolution=sampling_resolution
            self.polygon=polygon
            self.WOI=WOI
            self.transform=transform
            self.res=self.transform[0]
            self.crestlines_present=True
            if preprocess_buffer_fac > 0:
                self.polygon=self.preprocess(fac=preprocess_buffer_fac)
            self.find_my_troughlines(troughlines)
            self.crestlines_present=self.find_my_crestlines(crestlines)
            if self.crestlines_present:
                self.find_main_crestlines(my_rasterio_open(wind_data_paths[0],mode='r+',driver='GTiff',IGNORE_COG_LAYOUT_BREAK=True),my_rasterio_open(wind_data_paths[1],mode='r+',driver='GTiff',IGNORE_COG_LAYOUT_BREAK=True),DEM_path,flux_x=flux_data[0],flux_y=flux_data[1])
                if self.main_crestlines==[]:
                    self.crestlines_present=False
                    return
                self.morphological_indices['Cummulative Crest Length']=self.crestlines.length
                self.morphological_indices['Cummulative Crest Length [km]']=self.morphological_indices['Cummulative Crest Length']/1e3              
                self.morphological_indices['Cummulative length of main crests [km]']=self.morphological_indices['Cummulative length of main crests']/1e3
                #self.find_heights_and_slopes(DEM_path,show_weird_sections=investigate_suspicious)
                self.morphological_indices['Area']=self.polygon.area
                self.morphological_indices['Perimeter']=self.polygon.length
                self.morphological_indices['Circularity']=4*np.pi*self.morphological_indices['Area']/self.morphological_indices['Perimeter']**2
                bounding_box=self.polygon.minimum_rotated_rectangle
                x, y = bounding_box.exterior.coords.xy
                box_a=LineString([[x[0],y[0]],[x[1],y[1]]]).length
                box_b=LineString([[x[2],y[2]],[x[1],y[1]]]).length
                self.morphological_indices['Bounding box length']=max(box_a,box_b)
                self.morphological_indices['Bounding box width']=min(box_a,box_b)
                self.morphological_indices['Bounding box elongation']=self.morphological_indices['Bounding box length']/self.morphological_indices['Bounding box width']
                self.morphological_indices['Area [km2]']=self.morphological_indices['Area']/1e6
                self.morphological_indices['Perimeter [km]']=self.morphological_indices['Perimeter']/1e3
                self.morphological_indices['Bounding box length [km]']=self.morphological_indices['Bounding box length']/1e3
                self.morphological_indices['Bounding box width [km]']=self.morphological_indices['Bounding box width']/1e3
                self.morphological_indices['Cross section width [km]']=self.morphological_indices['Cross section width']/1e3
                self.morphological_indices['Crest elongation']=self.morphological_indices['Cummulative length of main crests']/self.morphological_indices['Cross section width']
                self.add_wind_to_NS_and_lat(transformer,southern_hemisphere=not Mars)
        else:
            self.polygon=all_given['polygon']
            self.main_crestlines=all_given['main crestlines']
            self.crestlines=all_given['crestlines']
            self.sampling_points=all_given['sampling points']
            self.orientation_vector=all_given['orientation vector']
            self.main_crestline_sample_points=all_given['main crestline sample points']
            self.tangent_lines=all_given['tangent_lines']
            self.stoss_projection_points=all_given['stoss projection points']
            self.lee_projection_points=all_given['lee projection points']
            self.stoss_along_wind_points=all_given['stoss along wind points']
            self.lee_along_wind_points=all_given['lee along wind points']
            self.cross_section_lines=all_given['cross section lines']
            self.cross_section_widths=list(map(lambda t: t.length,self.cross_section_lines))
            self.tangent_vectors=all_given['tangent vectors']
            self.wind_direction=all_given['wind vector']
            self.local_wind_vectors=all_given['local wind vectors']
            self.local_widths=all_given['local widths']
            self.h_values_for_visualization=all_given['h values']            
            self.x_values_for_visualization=all_given['x values']
            self.z1_values_for_visualization=all_given['z1 values']            
            self.z2_values_for_visualization=all_given['z2 values']
            self.zz1_values_for_visualization=all_given['zz1 values']            
            self.zz2_values_for_visualization=all_given['zz2 values']
            self.morphological_indices=all_given['indices']
            self.sampling_point_lvl_indices=all_given['point_lvl']
            self.res=all_given['res']
            self.transform=all_given['transform']
            self.WOI=all_given['WOI']
            if investigate_suspicious:
                self.find_heights_and_slopes_old(DEM_path,show_weird_sections=True)
            self.add_wind_to_NS_and_lat(transformer,southern_hemisphere=not Mars)

    def preprocess(self,fac=0.05):
        st1=self.polygon.buffer(0) # Relict of old method which somehow makes geometry type better 
        st2=shapely.simplify(st1,tolerance=fac*self.WOI)
        return st2

    def find_my_crestlines(self,crestlines):
        crest_list=shapely.intersection(self.polygon,crestlines)
        if crest_list.is_empty or isinstance(crest_list,Point):
            return False
        else:
            if isinstance(crest_list,MultiLineString):
                self.crestlines=crest_list
            elif isinstance(crest_list,shapely.GeometryCollection):
                self.crestlines=MultiLineString([g for g in crest_list.geoms if not isinstance(g,Point)])
            elif isinstance(crest_list,LineString):
                self.crestlines=MultiLineString([crest_list])
            else:
                print("Something else: {}".format(crest_list))
            return True

    def find_my_troughlines(self,troughlines,buffer=750):
        trough_list=shapely.intersection(self.polygon.buffer(buffer),troughlines)
        if trough_list.is_empty or isinstance(trough_list,Point):
            self.troughlines=MultiLineString()
        elif isinstance(trough_list,MultiLineString):
            self.troughlines=trough_list
        elif isinstance(trough_list,shapely.GeometryCollection):
            self.troughlines=MultiLineString([g for g in trough_list.geoms if not isinstance(g,Point)])
        elif isinstance(trough_list,LineString):
            self.troughlines=MultiLineString([trough_list])
        else:
            print("Something else: {}".format(trough_list))


    
    def find_projection(self,point,normal_vect,iter_limit=100,use_troughs=True):
        condition=True
        fac_max=5
        fac_min=0
        erratic=False
        counter=0
        while condition:
            if use_troughs:
                intersection=shapely.intersection(LineString([point-normal_vect*self.WOI*fac_max,point]),self.troughlines)
            else:
                intersection=shapely.intersection(LineString([point-normal_vect*self.WOI*fac_max,point]),self.polygon.exterior)
            if counter>iter_limit:
                if use_troughs:
                    return self.find_projection(point,normal_vect,iter_limit=iter_limit,use_troughs=False)
                erratic=True
                intersection=None
                break
            if isinstance(intersection,Point):
                condition=False
            elif isinstance(intersection,MultiPoint):
                fac_max=(fac_max+fac_min)/2
            else:
                fac_min=fac_max
                fac_max=2*fac_max
            counter+=1
        return intersection, erratic
                
    
    def find_main_crestlines(self,wind_x,wind_y,DEM_path,flux_x=None,flux_y=None):
        line_segments = list(self.crestlines.geoms)
        G = nx.Graph()
        for line in line_segments: #For each split line
            coords = list(line.coords) #Extract its coordinates
            #Add an edge using the start and end coordinates as nodes, and append 
            #  line length and the line geometry as edge attributes
            G.add_edge(coords[0], coords[-1], length=line.length, geometry=line)
        
        SG = [G.subgraph(c).copy() for c in nx.connected_components(G)]
        
        self.main_crestlines=[]
        self.morphological_indices['Cummulative length of main crests']=0
        self.morphological_indices['Sinuosity']=0
        self.crest_dividers=[0]
        
        for S in SG:
            end_nodes = [node for node, d in S.degree if d==1]
            if len(end_nodes)<2:
                end_nodes = [node for node, d in S.degree if d<=2]
            if len(end_nodes)<2:
                continue
            else:
                shortest_paths = [] #A list of all shortest paths
                for n1, n2 in combinations(end_nodes, 2):
                    shortest_path = nx.single_source_dijkstra(S, source=n1, target=n2, weight="length")
                    shortest_paths.append(shortest_path)
                
                #Sort the list by length so the longest paths is last [-1]
                longest_shortest_path = sorted(shortest_paths, key=lambda x: x[0])[-1]

                if longest_shortest_path[0]<self.WOI/2:
                    continue
                else:
                    self.morphological_indices['Cummulative length of main crests']+=longest_shortest_path[0]
        
                    lines = []
                    for n1, n2 in zip(longest_shortest_path[1], longest_shortest_path[1][1:]):
                        lines.append(S.edges[n1, n2]["geometry"])

                    lines_new=[]
                    for l in lines:
                        to_be_added=list(l.coords)
                        if len(lines_new)>0:
                            if to_be_added[0]!=lines_new[-1]:
                                if to_be_added[-1]!=lines_new[-1]:
                                    lines_new.reverse()
                                if to_be_added[0]!=lines_new[-1]:
                                    to_be_added.reverse()
                        lines_new+=to_be_added
                    self.main_crestlines+=[LineString(lines_new)]
                    
                    # Weighted average sinuosity with crest length as weights
                    self.crest_dividers.append(self.crest_dividers[-1]+longest_shortest_path[0])
                    self.morphological_indices['Sinuosity']+=(longest_shortest_path[0]**2)/(LineString([Point(longest_shortest_path[1][0]),Point(longest_shortest_path[1][-1])]).length)
                
                    
        if len(self.main_crestlines)==0:
            return
        else:
            self.morphological_indices['Sinuosity']/=self.morphological_indices['Cummulative length of main crests']
            self.main_crestlines=MultiLineString(self.main_crestlines)
            #self.main_crestlines_united=shapely.union_all(self.main_crestlines.geoms)
            self.main_crestline_sample_points=[]
            self.tangent_lines=[]
            self.stoss_projection_points=[]
            self.lee_projection_points=[]
            self.cross_section_lines=[]
            self.tangent_vectors=[]
            self.local_widths=[]
            self.orientation_vector=np.array([0.0,0.0])
            self.morphological_indices['Cross section width']=0
            self.allocate_sampling_points(wind_x,wind_y,DEM_path,flux_x=flux_x,flux_y=flux_y)

    def NRSD_sensitivity_analysis_one_polygon(self,DEM_path,points_to_move=['crest'],absolute_move=False,move_values=[-10,-5,5,10]):
        slope_stoss_rotated=np.zeros((len(points_to_move),len(move_values)))
        slope_lee_rotated=np.zeros((len(points_to_move),len(move_values)))
        with my_rasterio_open(DEM_path,mode='r+',driver='GTiff',IGNORE_COG_LAYOUT_BREAK=True) as src:        
            for i in range(self.sampling_points):
                
                x1=self.stoss_projection_points[i].x*np.ones((len(points_to_move),len(move_values)))
                y1=self.stoss_projection_points[i].y*np.ones((len(points_to_move),len(move_values)))
                x2=self.lee_projection_points[i].x*np.ones((len(points_to_move),len(move_values)))
                y2=self.lee_projection_points[i].y*np.ones((len(points_to_move),len(move_values)))
                x3=self.main_crestline_sample_points[i].x*np.ones((len(points_to_move),len(move_values)))
                y3=self.main_crestline_sample_points[i].y*np.ones((len(points_to_move),len(move_values)))
                if absolute_move:
                    mvx=(x2[0,0]-x1[0,0])/((x1[0,0]-x2[0,0])**2+(y1[0,0]-y2[0,0])**2)**0.5
                    mvy=(y2[0,0]-y1[0,0])/((x1[0,0]-x2[0,0])**2+(y1[0,0]-y2[0,0])**2)**0.5
                else:
                    mvx=(x2[0,0]-x1[0,0])*0.01
                    mvy=(y2[0,0]-y1[0,0])*0.01
                for j in range(len(points_to_move)):
                    for k in range(len(move_values)):
                        if points_to_move[j]=='crest':
                                x3[j]+=move_values[k]*mvx   
                                y3[j]+=move_values[k]*mvy     
                        elif points_to_move[j]=='lee':
                                x2[j]+=move_values[k]*mvx   
                                y2[j]+=move_values[k]*mvy      
                        if points_to_move[j]=='stoss':
                                x1[j]+=move_values[k]*mvx   
                                y1[j]+=move_values[k]*mvy      
                
                z1=next(src.sample([(x1,y1)]))[0]
                z2=next(src.sample([(x2,y2)]))[0]            
                z3=next(src.sample([(x3,y3)]))[0]
                
                x0,y0,z0,dz,h,a,b=crest_vertical_projection(x1,x2,x3,y1,y2,y3,z1,z2,z3)
                
                slope_stoss_rotated+=np.arctan(h/a)
                slope_lee_rotated+=np.arctan(h/b)      

        return (slope_stoss_rotated-slope_lee_rotated)/(slope_stoss_rotated+slope_lee_rotated)
    
    def allocate_sampling_points(self,wind_x,wind_y,DEM_path,flux_x=None,flux_y=None,additional_climate_indices=[""],buf=2):
        # New version which analyses heights and slopes at the same time and rejects sampling points if whichever slope is negative
        lines_for_shares=sorted(self.main_crestlines.geoms,key=lambda x:x.length,reverse=True)
        m=len(lines_for_shares)
        self.sampling_points=0
        self.local_wind_vectors=[]
        self.local_flux_vectors=[]
        self.morphological_indices['Cross section height [m]']=0
        self.morphological_indices['Residual slope ratio']=0
        self.morphological_indices['Absolute slope ratio']=0
        self.morphological_indices['Ratio of mean slopes']=0
        self.morphological_indices['Slope in prevailing wind direction [deg]']=0
        self.morphological_indices['Slope along normal [deg]']=0
        self.sampling_point_lvl_indices={'Cross section width [km]':[],'Cross section height [m]':[],'Angle with respect to wind [deg]':[],'Aspect ratio':[],'NASD':[],'NRSD':[],'Slope in prevailing wind direction [deg]':[],'Slope along normal [deg]':[]}
        if flux_x is None or flux_y is None:
            evaluating_flux=False
        else:
            evaluating_flux=True
        if evaluating_flux:
            self.sampling_point_lvl_indices['Angle with respect to flux [deg]']=[]
        to_divide=0
        to_divide_unrotated=0
        self.lee_along_wind_points=[]
        self.stoss_along_wind_points=[]
        self.h_values_for_visualization=[]
        self.x_values_for_visualization=[]
        self.z1_values_for_visualization=[]
        self.z2_values_for_visualization=[]
        self.zz1_values_for_visualization=[]
        self.zz2_values_for_visualization=[]
        failed_some_sampling_points=False
        with my_rasterio_open(DEM_path,mode='r+',driver='GTiff',IGNORE_COG_LAYOUT_BREAK=True) as src:
            for i in range(m):
                some_erratic_along_wind_projection=False
                some_erratic_along_wind_try_except=False
                some_erratic_normal_try_except=False
                some_erratic_normal_projection=False
                some_erratic_negative_slope=False
                curr_points=self.sampling_points
                ll=lines_for_shares[i].length
                #num=np.ceil(self.sampling_points*ll/total)#self.morphological_indices['Cummulative length of main crests'])
                num=int(np.ceil(-2*buf+ll/self.sampling_resolution))
                if num<=0:
                    if i==0:
                        num=1
                        for_probing=[ll/2.0]
                    else:
                        break
                else:
                    leftover=(ll-2*buf*self.sampling_resolution)%self.sampling_resolution
                    for_probing=np.linspace(buf*self.sampling_resolution+leftover/2.0,ll-buf*self.sampling_resolution-leftover/2.0,num)
                #if num+num_already>self.sampling_points:
                #    num=self.sampling_points-num_already
                #num_already+=num
                #for j in np.arange(buf,num+buf,1):

                for j in for_probing:
                    current_sampling_point=lines_for_shares[i].interpolate(j)
                    x3=current_sampling_point.x
                    y3=current_sampling_point.y
                    
                    
                    tangent_vect_plus=lines_for_shares[i].interpolate(5+j)
                    tangent_vect_minus=lines_for_shares[i].interpolate(-5+j)
                    tangent_vect=np.array([tangent_vect_plus.x-tangent_vect_minus.x,tangent_vect_plus.y-tangent_vect_minus.y])
                    tangent_vect/=np.dot(tangent_vect,tangent_vect.T)**0.5
                    normal_vect=np.array([-tangent_vect[1],tangent_vect[0]])
                    sample_point_coords=np.array([x3,y3])
                    local_wind_vector=np.array([next(wind_x.sample([(x3,y3)]))[0],next(wind_y.sample([(x3,y3)]))[0]])

                    try:
                        if np.arccos(np.dot(local_wind_vector,normal_vect.T))<np.pi/2.0:
                            stoss_projection_point,erratic1=self.find_projection(sample_point_coords,normal_vect)
                            lee_projection_point,erratic2=self.find_projection(sample_point_coords,-normal_vect)
                        else:
                            stoss_projection_point,erratic1=self.find_projection(sample_point_coords,-normal_vect)
                            lee_projection_point,erratic2=self.find_projection(sample_point_coords,normal_vect)
                    except OverflowError:
                        some_erratic_normal_try_except=True
                        continue

                    if(erratic1 or erratic2):
                        some_erratic_normal_projection=True
                        continue                        
                        
                    x1=stoss_projection_point.x
                    y1=stoss_projection_point.y
                    z1=next(src.sample([(x1,y1)]))[0]
                    x2=lee_projection_point.x
                    y2=lee_projection_point.y
                    z2=next(src.sample([(x2,y2)]))[0]
                    z3=next(src.sample([(x3,y3)]))[0]
                    
                    x0,y0,z0,dz,h,a,b=crest_vertical_projection(x1,x2,x3,y1,y2,y3,z1,z2,z3)
                    
                    slope_stoss_rotated=np.arctan(h/a)
                    slope_lee_rotated=np.arctan(h/b)
                    slope_lee_unrotated=np.arctan((z3-z2)/(((x3-x2)**2+(y3-y2)**2)**0.5))
                    slope_stoss_unrotated=np.arctan((z3-z1)/(((x3-x1)**2+(y3-y1)**2)**0.5))
                    
                    if(min(slope_stoss_rotated,slope_stoss_unrotated,slope_lee_rotated,slope_lee_unrotated)<=0):
                        some_erratic_negative_slope=True
                        continue
                    try:
                        stoss_along_wind_point,erratic1=self.find_projection(sample_point_coords,local_wind_vector)
                        lee_along_wind_point,erratic2=self.find_projection(sample_point_coords,-local_wind_vector)                    
                    except:
                        some_erratic_along_wind_try_except=True
                        continue
                       
                    if(erratic1 or erratic2):
                        some_erratic_along_wind_projection=True
                        continue     
                    
                    zz1 = next(src.sample([(stoss_along_wind_point.x,stoss_along_wind_point.y)]))[0]
                    zz2 = next(src.sample([(lee_along_wind_point.x,lee_along_wind_point.y)]))[0]
                    slope_along_wind=np.degrees(np.arctan((zz1-zz2)/shapely.distance(stoss_along_wind_point,lee_along_wind_point)))
                    slope_along_normal=np.degrees(np.arctan((z1-z2)/shapely.distance(stoss_projection_point,lee_projection_point)))
                    
                    self.morphological_indices['Slope in prevailing wind direction [deg]']+=slope_along_wind
                    self.sampling_point_lvl_indices['Slope in prevailing wind direction [deg]'].append(slope_along_wind)
                    self.morphological_indices['Slope along normal [deg]']+=slope_along_normal
                    self.sampling_point_lvl_indices['Slope along normal [deg]'].append(slope_along_normal)
                    self.zz1_values_for_visualization.append(zz1-z3)
                    self.zz2_values_for_visualization.append(zz2-z3)
                    self.lee_along_wind_points.append(lee_along_wind_point)
                    self.stoss_along_wind_points.append(stoss_along_wind_point)
                    self.main_crestline_sample_points.append(current_sampling_point)
                    self.local_wind_vectors.append(local_wind_vector)
                    self.tangent_vectors.append(tangent_vect)
                    self.orientation_vector+=tangent_vect*int(1-2*(tangent_vect[1]<0))
                    angle=np.arccos(min(np.dot(tangent_vect,local_wind_vector.T),1))
                    self.sampling_point_lvl_indices['Angle with respect to wind [deg]'].append(np.degrees(min(angle,np.pi-angle)))
                    if evaluating_flux:
                        local_flux_vector=np.array([next(flux_x.sample([(x3,y3)]))[0],next(flux_y.sample([(x3,y3)]))[0]])
                        self.local_flux_vectors.append(local_flux_vector)
                        angle=np.arccos(min(np.dot(tangent_vect,local_flux_vector.T),1))
                        self.sampling_point_lvl_indices['Angle with respect to flux [deg]'].append(np.degrees(min(angle,np.pi-angle)))         
                    self.tangent_lines.append(LineString([sample_point_coords-tangent_vect*self.WOI/4,sample_point_coords+tangent_vect*self.WOI/4]))
                    self.stoss_projection_points.append(stoss_projection_point)
                    self.lee_projection_points.append(lee_projection_point)
                    self.cross_section_lines.append(LineString([self.stoss_projection_points[-1],self.lee_projection_points[-1]]))
                    self.local_widths.append((self.cross_section_lines[-1].length**2+dz**2)**0.5)
                    self.morphological_indices['Cross section width']+=self.local_widths[-1]
                    self.sampling_point_lvl_indices['Cross section width [km]'].append(self.local_widths[-1]/1e3)                 
                    self.h_values_for_visualization.append(h)
                    self.x_values_for_visualization.append(b)
                    self.z1_values_for_visualization.append(z1-z3)
                    self.z2_values_for_visualization.append(z2-z3)
                    self.morphological_indices['Cross section height [m]']+=h
                    self.sampling_point_lvl_indices['Cross section height [m]'].append(h) 
                    self.sampling_point_lvl_indices['Aspect ratio'].append(h/self.local_widths[-1])                      
                    self.morphological_indices['Residual slope ratio']+=slope_stoss_rotated
                    self.sampling_point_lvl_indices['NRSD'].append((slope_stoss_rotated-slope_lee_rotated)/(slope_stoss_rotated+slope_lee_rotated))                        
                    self.morphological_indices['Absolute slope ratio']+=slope_stoss_unrotated
                    self.sampling_point_lvl_indices['NASD'].append((slope_stoss_unrotated-slope_lee_unrotated)/(slope_stoss_unrotated+slope_lee_unrotated))     
                    to_divide+=slope_lee_rotated
                    to_divide_unrotated+=slope_lee_unrotated                    
                    self.sampling_points+=1
                    """                   
                if self.sampling_points>curr_points:
                    failed_some_sampling_points=True

                    if some_erratic_along_wind_projection:
                        print(ll, "Erratic along wind projection")
                    elif some_erratic_along_wind_try_except:
                        print(ll, "Erratic along wind try except")
                    elif some_erratic_normal_try_except:
                        print(ll, "Erratic normal try except")
                    elif some_erratic_normal_projection:
                        print(ll, "Erratic normal projection")   
                    elif some_erratic_negative_slope:
                        print(ll, "Negative slope")   

                    else:
                        failed_some_sampling_points=False
                    """
        if self.sampling_points==0:                 
            self.main_crestlines=[]
            return
        #elif failed_some_sampling_points:
        #    print(shapely.to_wkt(self.polygon))
        self.orientation_vector/=np.dot(self.orientation_vector,self.orientation_vector.T)**0.5
        self.morphological_indices['Slope in prevailing wind direction [deg]']/=self.sampling_points
        self.morphological_indices['Slope along normal [deg]']/=self.sampling_points
        self.wind_direction=np.mean(np.array(self.local_wind_vectors),axis=0)
        self.wind_direction/=np.dot(self.wind_direction,self.wind_direction.T)**0.5
        angle=np.arccos(min(np.dot(self.orientation_vector,self.wind_direction.T),1))
        self.morphological_indices['Angle with respect to wind [deg]']=np.degrees(min(angle,np.pi-angle))
        if evaluating_flux:
            self.flux_direction=np.mean(np.array(self.local_flux_vectors),axis=0)
            self.flux_direction/=np.dot(self.flux_direction,self.flux_direction.T)**0.5
            angle=np.arccos(np.dot(self.orientation_vector,self.flux_direction.T))
            if np.isnan(angle):
                print(np.dot(self.orientation_vector,self.flux_direction.T))
            self.morphological_indices['Angle with respect to flux [deg]']=np.degrees(min(angle,np.pi-angle))
        self.morphological_indices['Cross section width']/=self.sampling_points
        self.morphological_indices['Cross section height [m]']/=self.sampling_points
        self.morphological_indices['Aspect ratio']=self.morphological_indices['Cross section height [m]']/self.morphological_indices['Cross section width']
        self.morphological_indices['NRSD']=(self.morphological_indices['Residual slope ratio']-to_divide)/(self.morphological_indices['Residual slope ratio']+to_divide)
        self.morphological_indices['NASD']=(self.morphological_indices['Absolute slope ratio']-to_divide_unrotated)/(self.morphological_indices['Absolute slope ratio']+to_divide_unrotated)
        self.morphological_indices['Residual slope ratio']/=to_divide
        self.morphological_indices['Absolute slope ratio']/=to_divide_unrotated
        return 
    
    def greater_scale_SPWD(self,DEM_path,length=10,length_in_km=True):
        length*=(1+999*int(length_in_km))
        half_length=length/2.0
        spwd=0
        with my_rasterio_open(DEM_path,mode='r+',driver='GTiff',IGNORE_COG_LAYOUT_BREAK=True) as src:
            for i in range(self.sampling_points):
                x_l=self.lee_along_wind_points[i].x
                y_l=self.lee_along_wind_points[i].y
                x_s=self.stoss_along_wind_points[i].x
                y_s=self.stoss_along_wind_points[i].y
                x_c=self.main_crestline_sample_points[i].x
                y_c=self.main_crestline_sample_points[i].y
                d_l=((x_l-x_c)**2+(y_l-y_c)**2)**0.5
                d_s=((x_s-x_c)**2+(y_s-y_c)**2)**0.5
                spwd+=np.degrees(np.arctan2(next(src.sample([(x_c+half_length*(x_s-x_c)/d_s,y_c+half_length*(y_s-y_c)/d_s)]))[0]-next(src.sample([(x_c+half_length*(x_l-x_c)/d_l,y_c+half_length*(y_l-y_c)/d_l)]))[0],length))
        return spwd/self.sampling_points
                
    def visualize(self,fig=None,ax=None,ax2=None,ax3=None, ax4=None, figsize=(5,5), raster=None, cmap=plt.cm.gray,alpha=1,all_crests=True, main_crests=True,points_along=True,tangent_lines=True,cross_sections=True,report=True,air=1,angle_marker_width=100,angle_marker_height=1,move_text=False,return_vis_transform=False,vis_transform=None,main_crest_color='yellow',return_artists=False,alpha_vec=1,clip_patch=None,fill=False,height_dependent_fill=None,thick_crests=False,thick_crest_thickness=2,suspicious_only=False,show_unrotated_too=False,show_along_wind_sections=False,titles_fontsize=13,angle_symbol_fontsize=10,angle_annotation_shifter=1.1,point_marker_size=13,scale_bar=False,scale_bar_val_km=1,each_to_its_own=None):
        if vis_transform is None:
            vis_transform=lambda x: (x+[-self.transform[2],-self.transform[5]])*np.array([1/self.transform[0],-1/self.transform[0]])
        
        if fig is None or ax is None:
            if report and each_to_its_own is None:
                ncls=2+int(show_unrotated_too)+int(show_along_wind_sections)
                fig,axs=plt.subplots(figsize=(ncls*figsize[0],figsize[1]),ncols=ncls)
                ax=axs[0]
                ax2=axs[1]
                if show_unrotated_too:
                    ax3=axs[2]
                    if show_along_wind_sections:
                        ax4=axs[3]
                    elif show_along_wind_sections:
                        ax4=axs[2]
            else:
                fig,ax=plt.subplots(figsize=figsize)
                
        if each_to_its_own is not None:
            fig,ax=plt.subplots(figsize=figsize)  
            fig2,ax2=plt.subplots(figsize=figsize)
            fig3,ax3=plt.subplots(figsize=figsize)
            fig4,ax4=plt.subplots(figsize=figsize)
        
        if raster is not None:
            l=self.polygon.bounds[0]   
            d=self.polygon.bounds[1]   
            r=self.polygon.bounds[2]   
            u=self.polygon.bounds[3]   
            l_raster=raster.bounds[0]   
            d_raster=raster.bounds[1]   
            r_raster=raster.bounds[2]   
            u_raster=raster.bounds[3]
            delta_horizontal=r-l
            delta_vertical=u-d
            if delta_horizontal > delta_vertical:
                mid_vertical=(u+d)/2
                u=mid_vertical+delta_horizontal/2
                d=mid_vertical-delta_horizontal/2
            else:
                mid_horizontal=(l+r)/2
                r=mid_horizontal+delta_vertical/2
                l=mid_horizontal-delta_vertical/2
            l-=5*self.res
            d-=5*self.res
            u+=5*self.res
            r+=5*self.res
        
            crop_y=[min(1,max(0,(l-l_raster)/(r_raster-l_raster))),min(1,max(0,(r-l_raster)/(r_raster-l_raster)))]
            crop_x=[min(1,max(0,(u-u_raster)/(d_raster-u_raster))),min(1,max(0,(d-u_raster)/(d_raster-u_raster)))]
            
            raster.visualize(ax=ax,fig=fig,cmap=cmap,crop_x=crop_x,crop_y=crop_y,title_fontsize=titles_fontsize,title="Dune morphological analysis",cbar=False,alpha=alpha)

            vis_transform=lambda x: (x+[-self.transform[2]-max(0,l-l_raster),-self.transform[5]+max(0,u_raster-u)])*np.array([1/self.transform[0],-1/self.transform[0]])

            if move_text is None:
                lr=(r-l)/2
                u2=(u+d)/2
                move_text=shapely.intersection(self.polygon,shapely.Polygon(zip([l,l+lr,l+lr,l,l],[d,d,u2,u2,d]))).area>shapely.intersection(self.polygon,shapely.Polygon(zip([r,r-lr,r-lr,r,r],[d,d,u2,u2,d]))).area
            move_polar=shapely.intersection(self.polygon,shapely.Polygon(zip([l,l+lr,l+lr,l,l],[u,u,u2,u2,u]))).area>shapely.intersection(self.polygon,shapely.Polygon(zip([r,r-lr,r-lr,r,r],[u,u,u2,u2,u]))).area            
        
        x,y = shapely.transform(self.polygon.exterior,vis_transform).xy
        if fill:
            if height_dependent_fill is None:
                c='steelblue'
            else:
                c=plt.cm.Blues((self.morphological_indices['Angle with respect to wind [deg]']-height_dependent_fill[0])/(height_dependent_fill[1]-height_dependent_fill[0]))
            arts=[ax.fill(x,y,lw=1,c=c,alpha=0.35*alpha_vec)[0],ax.plot(x,y,lw=0.25,c='black',alpha=alpha_vec)[0]]
        else:
            arts=[ax.plot(x,y,lw=1,c='indigo',alpha=alpha_vec)[0]]
        if all_crests:
            shplt.plot_line(shapely.transform(self.crestlines,vis_transform),ax=ax,add_points=False,linewidth=1.2,color='darkgreen',alpha=alpha_vec,label='Crest skeleton')
        if scale_bar:
            fct_v=0.06
            if move_text:
                fct=0.1
                shf=1
            else:
                fct=0.9
                shf=-1
            x_bar_a,y_bar_a=vis_transform(np.array([(1-fct)*l+fct*r,(1-fct_v)*d+fct_v*u]))
            x_bar_a1,y_bar_a1=vis_transform(np.array([(1-fct)*l+fct*r,(1-fct_v)*d+fct_v*u+50*scale_bar_val_km]))
            x_bar_a2,y_bar_a2=vis_transform(np.array([(1-fct)*l+fct*r,(1-fct_v)*d+fct_v*u-50*scale_bar_val_km]))
            x_bar_b,y_bar_b=vis_transform(np.array([(1-fct)*l+fct*r+shf*1000*scale_bar_val_km,(1-fct_v)*d+fct_v*u]))
            x_bar_b1,y_bar_b1=vis_transform(np.array([(1-fct)*l+fct*r+shf*1000*scale_bar_val_km,(1-fct_v)*d+fct_v*u+50*scale_bar_val_km]))
            x_bar_b2,y_bar_b2=vis_transform(np.array([(1-fct)*l+fct*r+shf*1000*scale_bar_val_km,(1-fct_v)*d+fct_v*u-50*scale_bar_val_km]))
            ax.plot([x_bar_a,x_bar_b],[y_bar_a,y_bar_b],c='black',lw=1)
            ax.plot([x_bar_a1,x_bar_a2],[y_bar_a1,y_bar_a2],c='black',lw=1)
            ax.plot([x_bar_b1,x_bar_b2],[y_bar_b1,y_bar_b2],c='black',lw=1)
            fcan=0.75
            fcan=fcan*int(not move_text)+(1-fcan)*int(move_text)
            ax.annotate("{} km".format(scale_bar_val_km),xy=(x_bar_a*(1-fcan)+x_bar_b*fcan,y_bar_a1))                                    
            
        if main_crests:
            arts.append(shplt.plot_line(shapely.transform(self.main_crestlines,vis_transform),ax=ax,add_points=False,linewidth=0.6+thick_crest_thickness*int(thick_crests),color=main_crest_color,alpha=alpha_vec*(1-0.5*int(fill)),label='Main crestlines'))
        if clip_patch is not None:
            for art in arts:
                art.set_clip_path(clip_patch)
        if suspicious_only:
            for_iterations=self.suspicious_sections
        else:
            for_iterations=sorted(range(self.sampling_points),key=lambda ind: self.main_crestline_sample_points[ind].y)
        j=0
        for i in for_iterations:
            if points_along:
                vis_pt=shapely.transform(self.main_crestline_sample_points[i],vis_transform)
                ax.scatter(vis_pt.x,vis_pt.y,color='darkcyan',s=point_marker_size,marker='o')
            if tangent_lines:
                shplt.plot_line(shapely.transform(self.cross_section_lines[i],vis_transform),ax=ax,add_points=False,linewidth=0.5,color='darkslategrey')
            if show_along_wind_sections:
                shplt.plot_line(shapely.transform(LineString([self.stoss_along_wind_points[i],self.lee_along_wind_points[i]]),vis_transform),ax=ax,add_points=False,linewidth=0.5,linestyle="dashed",color=plt.cm.copper(0.5*(1+j/self.sampling_points)))
                vis_pt=shapely.transform(self.stoss_along_wind_points[i],vis_transform)
                ax.scatter(vis_pt.x,vis_pt.y,color='maroon',s=point_marker_size,marker='o')       
                vis_pt=shapely.transform(self.lee_along_wind_points[i],vis_transform)
                ax.scatter(vis_pt.x,vis_pt.y,color='seagreen',s=point_marker_size,marker='o')    
            if cross_sections:
                shplt.plot_line (shapely.transform(self.cross_section_lines[i],vis_transform),ax=ax,add_points=False,linewidth=0.5,color=plt.cm.PuRd(0.5*(1+j/self.sampling_points)))
                vis_pt=shapely.transform(self.stoss_projection_points[i],vis_transform)
                ax.scatter(vis_pt.x,vis_pt.y,color='orange',s=point_marker_size,marker='o')       
                vis_pt=shapely.transform(self.lee_projection_points[i],vis_transform)
                ax.scatter(vis_pt.x,vis_pt.y,color='lime',s=point_marker_size,marker='o')  
              
            j+=1
        if report:
            #xtick=ax.get_xticks()[-1]
            #ytick=ax.get_yticks()[-1]
            WOI_modified=self.WOI/self.res
            sub_ax = ax.inset_axes([0.05+0.7*int(move_polar),0.7,0.15,0.15],polar=True)            
            #sub_ax = inset_axes()
            orient=np.arctan2(self.orientation_vector[1],self.orientation_vector[0])
            sub_ax.annotate('', xy=(orient,1), xytext=(orient+np.pi,1), arrowprops=dict(arrowstyle='<->',color='black'))
            wind_orient=np.arctan2(self.wind_direction[1],self.wind_direction[0])
            sub_ax.annotate('', xy=(wind_orient,1), xytext=(wind_orient+180,0.1), arrowprops=dict(arrowstyle='->',color='blue'))
            sub_ax.set_title('{:.2f}°'.format(np.degrees(orient%np.pi)),fontsize=10)
            sub_ax.set_yticklabels([])
            sub_ax.set_xticklabels([])
            sub_ax.patch.set_alpha(0.01)
            ax.text(0.05+0.62*int(move_text),0.05,"Crest: {:.1f} km\nWidth: {:.1f} km\nHeight: {:.2f} m\n"r'$\angle$'" to wind: {:.2f}°\nNRSD: {:.2f}".format(self.morphological_indices['Cummulative length of main crests [km]'],self.morphological_indices['Cross section width']/1000,self.morphological_indices['Cross section height [m]'],self.morphological_indices['Angle with respect to wind [deg]'],self.morphological_indices['NRSD'])+int(show_unrotated_too)*"\nNASD: {:.2f}\nSAN:"r'${:.2f}^\circ$'.format(self.morphological_indices['NASD'],self.morphological_indices['Slope along normal [deg]'])+int(show_along_wind_sections)*"\nSPWD: "r'${:.2f}^\circ$'.format(self.morphological_indices['Slope in prevailing wind direction [deg]']),bbox={'ec':'black','capstyle':'round','fc':'navajowhite','alpha':0.8},fontsize=8,transform=ax.transAxes)
            h=0
            j=0
            max_width=np.max(np.array(self.local_widths))
            air*=np.sum(np.array(self.h_values_for_visualization))/10
            for i in for_iterations:
                vis_pt_1=shapely.transform(self.main_crestline_sample_points[i],vis_transform)
                local_orient=np.degrees(np.arctan2(self.tangent_vectors[i][1],self.tangent_vectors[i][0]))
                ax.annotate("{:.2f}°".format(local_orient%180),xy=(vis_pt_1.x+air,vis_pt_1.y+air),fontsize=8)
                ax2.plot([0,self.x_values_for_visualization[i],self.cross_section_widths[i]],np.array([0,self.h_values_for_visualization[i],0])+h,color=plt.cm.PuRd(0.5*(1+j/self.sampling_points)))
                ax2.plot([0,self.cross_section_widths[i]],[h,h],lw=0.4,ls='dashed',c='black',alpha=0.6)
                ax2.vlines(self.x_values_for_visualization[i],h,h+self.h_values_for_visualization[i],color='black',lw=0.9)
                corr=min(1,self.h_values_for_visualization[i]/(2*angle_marker_height))
                ax2.add_patch(Arc([self.x_values_for_visualization[i],h],angle_marker_width*corr*(12-4*min(2.75,self.sampling_points)),angle_marker_height*corr,theta1=90,theta2=180,color='black',lw=0.5))
                ax2.scatter(self.x_values_for_visualization[i]-angle_marker_width*(12-4*min(2.75,self.sampling_points))*corr/5,h+angle_marker_height*corr/5,color='black',s=corr*0.65)
                ax2.annotate("{:.2f} m".format(self.h_values_for_visualization[i]),xy=(self.x_values_for_visualization[i]+angle_marker_width/5,h+self.h_values_for_visualization[i]/5),fontsize=9)
                ax2.annotate("{:.0f} m".format(self.local_widths[i]),xy=(0.9*self.local_widths[i],h-air*0.9),annotation_clip=False,fontsize=10)
                ax2.scatter(0,h,color='lime',s=2*point_marker_size,marker='o')
                ax2.scatter(self.local_widths[i],h,color='orange',s=2*point_marker_size,marker='o')
                alph=np.degrees(np.arctan(self.h_values_for_visualization[i]/self.x_values_for_visualization[i]))
                beta=np.degrees(np.arctan(self.h_values_for_visualization[i]/(self.local_widths[i]-self.x_values_for_visualization[i])))                
                ax2.scatter(self.x_values_for_visualization[i],h+self.h_values_for_visualization[i],color='darkcyan',s=2*point_marker_size,marker='o')
                ax2.add_patch(Arc([0,h],angle_marker_width*4,angle_marker_height*4,theta1=0,theta2=alph,color='black',lw=0.5))
                ax2.add_patch(Arc([self.local_widths[i],h],angle_marker_width*4,angle_marker_height*4,theta1=180-beta,theta2=180,color='black',lw=0.5))
                ax2.annotate(r'$\alpha$'.format(alph),xy=(self.x_values_for_visualization[i]/10,h+self.h_values_for_visualization[i]/3),fontsize=angle_symbol_fontsize)
                ax2.annotate(r'$\beta$'.format(beta),xy=(self.local_widths[i]*0.95,h+self.h_values_for_visualization[i]/3),fontsize=angle_symbol_fontsize)

                ax2.annotate(r'${}={:.2f}$'.format(r'\frac{\beta-\alpha}{\beta+\alpha}',(beta-alph)/(beta+alph)),xy=(max_width*angle_annotation_shifter,h+self.h_values_for_visualization[i]/3),annotation_clip=False,fontsize=angle_symbol_fontsize)
                j+=1
                h+=self.h_values_for_visualization[i]+air
            ax2.set_title('Sampled cross sections (base flattened)',fontsize=titles_fontsize)
            ax2.axis('off')                
            if show_unrotated_too:
                h=0
                j=0
                for i in for_iterations:    
                    ax3.plot([0,shapely.distance(self.lee_projection_points[i],self.main_crestline_sample_points[i]),self.cross_section_lines[i].length],np.array([self.z2_values_for_visualization[i],0,self.z1_values_for_visualization[i]])+h,color=plt.cm.PuRd(0.5*(1+j/self.sampling_points)))            
                    #ax3.plot([0,self.cross_section_lines[i].length],[h+self.z1_values_for_visualization[i],h+self.z2_values_for_visualization[i]],lw=0.4,ls='dashed',c='black',alpha=0.3)
                    ax3.plot([0,self.cross_section_lines[i].length],[h+self.z2_values_for_visualization[i],h+self.z2_values_for_visualization[i]],lw=0.4,ls='dotted',c='black',alpha=0.6)
                    ax3.plot([0,self.cross_section_lines[i].length],[h+self.z1_values_for_visualization[i],h+self.z1_values_for_visualization[i]],lw=0.4,ls='dotted',c='black',alpha=0.6)
                    ax3.scatter(0,h+self.z2_values_for_visualization[i],color='lime',s=2*point_marker_size,marker='o')
                    ax3.plot([0,self.cross_section_lines[i].length],[h+self.z2_values_for_visualization[i],h+self.z1_values_for_visualization[i]],lw=0.4,ls='dashed',c='black',alpha=0.6)
                    ax3.plot([0,self.cross_section_lines[i].length],[h+self.z2_values_for_visualization[i],h+self.z2_values_for_visualization[i]],lw=0.4,ls='dotted',c='black',alpha=0.6)
                    ax3.scatter(self.cross_section_lines[i].length,h+self.z1_values_for_visualization[i],color='orange',s=2*point_marker_size,marker='o')            
                    ax3.scatter(shapely.distance(self.lee_projection_points[i],self.main_crestline_sample_points[i]),h,color='darkcyan',s=2*point_marker_size,marker='o')
                    gamma=np.degrees(np.arctan(-self.z2_values_for_visualization[i]/shapely.distance(self.lee_projection_points[i],self.main_crestline_sample_points[i])))
                    delta=np.degrees(np.arctan(-self.z1_values_for_visualization[i]/shapely.distance(self.stoss_projection_points[i],self.main_crestline_sample_points[i])))
                    if gamma<0:
                        th1=gamma
                        th2=0
                    else:
                        th1=0
                        th2=gamma
                    ax3.add_patch (Arc([0,h+self.z2_values_for_visualization[i]],angle_marker_width*4,angle_marker_height*4,theta1=th1,theta2=th2,color='black',lw=0.5))
                    if delta<0:
                        th1=delta
                        th2=0
                    else:
                        th1=0
                        th2=delta    
                    ax3.add_patch (Arc([self.cross_section_lines[i].length,h+self.z1_values_for_visualization[i]],angle_marker_width*4,angle_marker_height*4,theta2=180-th1,theta1=180-th2,color='black',lw=0.5))
                    ax3.annotate(r'$\gamma$',xy=(self.x_values_for_visualization[i]/10,h+0.7*self.z2_values_for_visualization[i]),fontsize=angle_symbol_fontsize)
                    ax3.annotate(r'$\delta$',xy=(self.local_widths[i]*0.95,h+0.7*self.z1_values_for_visualization[i]),fontsize=angle_symbol_fontsize)
                    thh=np.degrees(np.arctan((self.z1_values_for_visualization[i]-self.z2_values_for_visualization[i])/self.cross_section_lines[i].length))
                    ax3.annotate(r'${}={:.2f}$'", "r'$\theta^*={:.2f}^\circ$'.format(r'\frac{\delta-\gamma}{\delta+\gamma}',(delta-gamma)/(delta+gamma),thh),xy=(max_width*angle_annotation_shifter,h+0.35*(self.z1_values_for_visualization[i]+self.z2_values_for_visualization[i])),annotation_clip=False,fontsize=angle_symbol_fontsize)
                    if thh<0:
                        th1=thh
                        th2=0
                    else:
                        th1=0
                        th2=thh
                    ax3.add_patch (Arc([0,h+self.z2_values_for_visualization[i]],angle_marker_width*8,angle_marker_height*8,theta1=th1,theta2=th2,color='black',lw=0.5))

                    ax3.annotate(r'$\theta^*$',xy=(self.x_values_for_visualization[i]/2,h+self.z2_values_for_visualization[i]*(1+0.2*(1-1.5*int(self.z2_values_for_visualization[i]<self.z1_values_for_visualization[i])))),fontsize=angle_symbol_fontsize,annotation_clip=False)
                    j+=1
                    h+=self.h_values_for_visualization[i]+air
                ax3.set_title('Sampled cross sections (as in DEM)',fontsize=titles_fontsize)
                ax3.axis('off') 
            if show_along_wind_sections:
                max_width=0
                for i in for_iterations:    
                    wind_sec_len=shapely.distance(self.lee_along_wind_points[i],self.stoss_along_wind_points[i])
                    if wind_sec_len>max_width:
                        max_width=wind_sec_len
                h=0
                j=0
                for i in for_iterations:    
                    wind_sec_len=shapely.distance(self.lee_along_wind_points[i],self.stoss_along_wind_points[i])
                    wind_x_len=shapely.distance(self.lee_along_wind_points[i],self.main_crestline_sample_points[i])    
                    ax4.plot([0,wind_x_len,wind_sec_len],np.array([self.zz2_values_for_visualization[i],0,self.zz1_values_for_visualization[i]])+h,linestyle='dashed',color=plt.cm.copper(0.5*(1+j/self.sampling_points)))            
                    ax4.plot([0,wind_sec_len],[h+self.zz2_values_for_visualization[i],h+self.zz1_values_for_visualization[i]],lw=0.4,ls='dashed',c='black',alpha=0.6)
                    ax4.plot([0,wind_sec_len],[h+self.zz2_values_for_visualization[i],h+self.zz2_values_for_visualization[i]],lw=0.4,ls='dotted',c='black',alpha=0.6)
                    ax4.scatter(0,h+self.zz2_values_for_visualization[i],color='seagreen',s=2*point_marker_size,marker='o')
                    ax4.scatter(wind_sec_len,h+self.zz1_values_for_visualization[i],color='maroon',s=2*point_marker_size,marker='o')            
                    ax4.scatter(wind_x_len,h,color='darkcyan',s=2*point_marker_size,marker='o')
                    thh=np.degrees(np.arctan((self.zz1_values_for_visualization[i]-self.zz2_values_for_visualization[i])/wind_sec_len))
                    if thh<0:
                        th1=thh
                        th2=0
                    else:
                        th1=0
                        th2=thh
                    ax4.add_patch (Arc([0,h+self.zz2_values_for_visualization[i]],angle_marker_width*4,angle_marker_height*4,theta1=th1,theta2=th2,color='black',lw=0.5))
                    ax4.annotate(r'$\theta$',xy=(self.x_values_for_visualization[i]/10,h+0.7*self.zz2_values_for_visualization[i]),fontsize=angle_symbol_fontsize)
                    ax4.annotate(r'$\theta={:.2f}^\circ$'.format(thh),xy=(max_width*angle_annotation_shifter,h+0.35*(self.zz1_values_for_visualization[i]+self.zz2_values_for_visualization[i])),annotation_clip=False,fontsize=angle_symbol_fontsize)
                    j+=1
                    h+=self.h_values_for_visualization[i]+air
                ax4.set_title('Wind parallel cross sections',fontsize=titles_fontsize)
                ax4.axis('off') 
        if each_to_its_own is not None:
            for aaa in [ax,ax2,ax3,ax4]:
                aaa.set_title('')
            for adj,fg in zip([1,0.5,0.5,0.5],[fig,fig2,fig3,fig4]):
                fg.subplots_adjust(right=adj, wspace=0, hspace=0)
            for desc,fg in zip(['_polygon.jpg','_base_flat.jpg','_normal_sec.jpg','wind_sec.jpg'],[fig,fig2,fig3,fig4]):
                fg.savefig(each_to_its_own+desc)
        if return_vis_transform:
            return vis_transform
        elif return_artists:
            return arts

    def add_wind_to_NS_and_lat(self,transformer,southern_hemisphere=True):
        self.morphological_indices["Latitude [deg]"]=self.get_coordinate(transformer)
        self.morphological_indices["Longitude [deg]"]=self.get_coordinate(transformer,latitude=False)
        self.morphological_indices["Wind angle to NS [deg]"]=self.get_wind_angle_to_NS(transformer,southern_hemisphere=southern_hemisphere)
        
    def get_coordinate(self,transformer,latitude=True):
        cent = shapely.centroid(self.polygon)
        lat,lon = transformer.transform(cent.x,cent.y)
        if latitude:
            return lat
        else:
            return lon
        
    def get_wind_angle_to_NS(self,transformer,southern_hemisphere=True):
        lon=np.radians(self.get_coordinate(transformer,latitude=False))
        NS_vect=np.array([np.sin(lon),np.cos(lon)])
        if not southern_hemisphere:
            NS_vect[1]=-NS_vect[1]
        prod=np.dot(NS_vect,self.wind_direction.T)
        if prod>1:
            prod=1
        elif prod<-1:
            prod=-1
        return np.degrees(np.arccos(prod))
    
        
class dune_polygons:

    def __init__(self,raster,crestlines,troughlines,DEM_data,DEM_path,wind_data_paths,all_given=None,precomputed=None,verbose=True,connectivity=4,preprocess_buffer_fac=0.1,map_base_path=r'.\parameter_map_',create_maps=False,indices=['Cummulative Crest Length [km]','Area [km2]','Perimeter [km]','Circularity','Cummulative length of main crests [km]','Cross section width [km]','Crest elongation','Cross section height [m]','Bounding box length [km]','Bounding box width [km]','Bounding box elongation','Sinuosity','Angle with respect to wind [deg]','Aspect ratio','NASD','NRSD','Slope in prevailing wind direction [deg]','Slope along normal [deg]'],point_lvl_indices=['Cross section width [km]','Cross section height [m]','Angle with respect to wind [deg]','Aspect ratio','NASD','NRSD','ASSI','RSSI','Slope in prevailing wind direction [deg]','Slope along normal [deg]'],for_copy=None,slope_ratio_filter=True,record_suspicious=None,investigate_suspicious=False,limit_to=None,climate_indices=["surface_pressure","temperature","total_precipitation","wind_speed","wind_constancy","froude_number"],climate_indice_custom_names=["Surface pressure [Pa]","Temperature 2m above surface [K]","Total precipitation m/day","Wind_speed","Wind constancy","Froude number"],climate_param_base_path=None,QGIS_write=None,Mars=False,transformer=None,dT_Froude=10,L_Froude=200):

        if transformer is None:
            if Mars:
                transformer=Transformer.from_crs(crs.CRS.from_wkt('PROJCS["MarsNP_3376",GEOGCS["GCS_Unknown",DATUM["D_Unknown",SPHEROID["S_Unknown",3376200,0]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]]],PROJECTION["Polar_Stereographic"],PARAMETER["latitude_of_origin",90],PARAMETER["central_meridian",0],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1],AXIS["Easting",SOUTH],AXIS["Northing",SOUTH]]'),"ESRI:104971")
            else:
                transformer=Transformer.from_crs("EPSG:3031","EPSG:4326")
        if for_copy is not None:
            self.indices=for_copy.indices
            self.point_lvl_indices=for_copy.point_lvl_indices
            self.n_par=len(self.indices)
            self.polygons=for_copy.polygons
            self.connectivity=connectivity
            self.transform=for_copy.transform
            self.res=for_copy.res
            self.WOI=for_copy.WOI
            self.num_polygons=for_copy.num_polygons
        
        elif all_given is None:
            if verbose:
                tck=time.time()
                print("Starting polygonizing")
            if climate_param_base_path is None:
                climate_indices=[]
                climate_indice_custom_names=None
            if climate_indice_custom_names is None:
                climate_indice_custom_names = climate_indices
            self.indices=indices+climate_indice_custom_names
            self.point_lvl_indices=point_lvl_indices
            self.n_par=len(self.indices)
            self.polygons=[]
            self.connectivity=connectivity
            self.transform=raster.transform
            self.res=raster.res
            self.WOI=raster.WOI
            suspicious_plot_count=0
            if precomputed is None:
                for coords, value in tqdm(features.shapes(raster.data,transform=self.transform,connectivity=self.connectivity)):
                    # ignore polygons corresponding to nodata
                    if value != 0:
                        # convert geojson to shapel1y geometry
                        geom = shape(coords)
                        new_poly=one_dune_polygon(geom,self.WOI,crestlines,troughlines,self.transform,DEM_path,wind_data_paths,preprocess_buffer_fac=preprocess_buffer_fac,suspicious_plot_count=suspicious_plot_count,investigate_suspicious=investigate_suspicious,transformer=transformer,Mars=Mars)
                        suspicious_plot_count=new_poly.suspicious_plot_count
                        if suspicious_plot_count>=20:
                            break
                    
                        if new_poly.crestlines_present:
                            self.polygons.append(new_poly)
            else:
                lines=my_open(precomputed).readlines()
                for l in lines[1:]:
                    new_poly=one_dune_polygon(shapely.from_wkt(l[1:-2]),self.WOI,crestlines,troughlines,self.transform,DEM_path,wind_data_paths,preprocess_buffer_fac=preprocess_buffer_fac,investigate_suspicious=investigate_suspicious,suspicious_plot_count=suspicious_plot_count,transformer=transformer,Mars=Mars)
                    suspicious_plot_count=new_poly.suspicious_plot_count
                    if suspicious_plot_count>=20:
                        break
                    if new_poly.crestlines_present:
                        self.polygons.append(new_poly)
            if verbose:
                print("Entering height calculation:")
            self.num_polygons=len(self.polygons)
            self.calculate_heights(DEM_data)
            for i in range(len(climate_indices)):
                self.calculate_climate_param(climate_param_base_path+"_"+climate_indices[i]+".tif",climate_indice_custom_names[i])
            if verbose:
                print("Polygonizing performed in {:.2f} s".format(time.time()-tck))
        else:
            self.point_lvl_indices=point_lvl_indices        
            self.read_from_file(all_given,DEM_path,indices_reset=indices,investigate_suspicious=investigate_suspicious,limit_to=limit_to,transformer=transformer)
        if slope_ratio_filter:
            if Mars:
                self.param_filter("Area [km2]",threshold=2000)
            else:
                self.param_filter()
                self.param_filter("Slope in prevailing wind direction [deg]",threshold=-1,lower=True)
        if create_maps:
            self.create_parameter_maps((raster.x,raster.y),raster.profile,base_path=map_base_path)
        if record_suspicious is not None:
            limited=dune_polygons(None,None,None,None,None,for_copy=self)
            limited.polygons=[pol for pol in limited.polygons if len(pol.suspicious_sections)>0]
            limited.num_polygons=len(limited.polygons)
            limited.export_to_file(record_suspicious)
        if all_given is None:
            self.nan_filter()
        if QGIS_write is not None:
            self.write_for_QGIS(QGIS_write)
        if not Mars and for_copy is None:
            for pol in self.polygons:
                pol.morphological_indices["Total precipitation [mm/day]"]=1000*pol.morphological_indices["Total precipitation m/day"]

    def NRSD_sensitivity_analysis(self,DEM_path,points_to_move=['crest'],absolute_move=False,move_values=[-10,-5,5,10],figsize=(15,10),save_to=None,colors=['indigo','blue','black','red','darkred'],sizes=[0.01,0.02,0.01,0.02,0.01],markers=['^','^','.','v','v']):
        if absolute_move:
            label_text="m"
        else:
            label_text="%"
        values=np.zeros((self.num_polygons,len(points_to_move),len(move_values)+1))
        move_values=sorted(move_values)
        zero_ind=0
        while zero_ind<len(move_values):
            if move_values[zero_ind]<0:
                zero_ind+=1
            else:
                break
        for i in tqdm(range(self.num_polygons)):
            values[i,:,zero_ind]=self.polygons[i].morphological_indices['NRSD']
            moved_vals=self.polygons[i].NRSD_sensitivity_analysis_one_polygon(DEM_path,points_to_move=points_to_move,absolute_move=absolute_move,move_values=move_values)
            values[i,:,:zero_ind]=moved_vals[:,:zero_ind]
            values[i,:,zero_ind+1:]=moved_vals[:,zero_ind:]
        if isinstance(save_to,str) or save_to is None:
            save_to=[save_to]
        save_to+=[None]*(len(points_to_move)-len(save_to))
        for j in range(len(points_to_move)):
            fig,ax=plt.subplots(figsize=figsize)
            for it in range(max(zero_ind,len(move_values)-zero_ind)):
                if it<zero_ind:
                    ax.scatter(values[:,j,zero_ind],values[:,j,it],marker=markers[it],s=sizes[it],c=colors[it],label="{}{} stoss-ward".format(abs(move_values[it]),label_text))
                if it<len(move_values)-zero_ind:
                    ax.scatter(values[:,j,zero_ind],values[:,j,len(move_values)-it],marker=markers[len(move_values)-it],s=sizes[len(move_values)-it],c=colors[len(move_values)-it],label="{}{} lee-ward".format(abs(move_values[len(move_values)-it-1]),label_text))
            ax.scatter(values[:,j,zero_ind],values[:,j,zero_ind],marker=markers[zero_ind],s=sizes[zero_ind],c=colors[zero_ind])
            ax.legend()
            ax.set_title("Analysis of NRSD sensitivity towards moving the {} point".format(points_to_move[j]))
            if save_to[j] is not None:
                my_savefig(fig,save_to[j])
            
    def add_symmetry_indices(self):
        for pol in self.polygons:
            pol.morphological_indices['RSSI']=abs(pol.morphological_indices['NRSD'])
            pol.morphological_indices['ASSI']=abs(pol.morphological_indices['NASD'])
            pol.sampling_point_lvl_indices['RSSI']=[abs(x) for x in pol.sampling_point_lvl_indices['NRSD']]
            pol.sampling_point_lvl_indices['ASSI']=[abs(x) for x in pol.sampling_point_lvl_indices['NASD']]
        if not 'RSSI' in self.point_lvl_indices:
            self.point_lvl_indices.append('RSSI')
        if not 'ASSI' in self.point_lvl_indices:
            self.point_lvl_indices.append('ASSI')
            
    def param_filter(self,par='Residual slope ratio',threshold=14,lower=False):
        self.polygons=[poly for poly in self.polygons if lower == (poly.morphological_indices[par]>threshold)]
        self.num_polygons=len(self.polygons)
        
    def nan_filter(self,nodata_val=None):
        for par in self.indices:
            new_polygons=[]
            for poly in self.polygons:
                if poly.morphological_indices[par] is None:
                    if nodata_val is not None:
                        poly.morphological_indices[par]=nodata_val
                        new_polygons.append(poly)
                    continue
                elif not np.isnan(poly.morphological_indices[par]):
                    new_polygons.append(poly)# for poly in self.polygons if not np.isnan(poly.morphological_indices[par])]
        self.polygons=new_polygons
        self.num_polygons=len(self.polygons)
            
    def calculate_heights(self,data):
        stats=zonal_stats([x.polygon for x in self.polygons], data, stats="min max",affine=self.transform)
        for i in tqdm(range(self.num_polygons)):
            self.polygons[i].morphological_indices['Height [m]']=stats[i]['max']-stats[i]['min']
         
    def SPWD_mustache(self,DEM_path,lengths=[1,10,25,50,100],fig=None,ax=None,figsize=(10,10),mustache_width=0.5,not_precomputed=True,box_color="darkolivegreen",bbox_to_anchor=None,legend_fontsize=15,title_fontsize=20,y=1.02,zero_line=True,zero_line_color="darkred",zero_line_style="dashed",frezzotti_range_color="lightseagreen",correlations_color="darkblue",zero_line_thickness=1.5,title="B-SPWD behavior dependent on baseline",outliers=False,export_to=None,cor_marker="X",cor_marker_size=5,whis=(1,99),**kwargs):
        
        if not_precomputed:
            self.spwds=list(map(lambda l: np.array([pol.greater_scale_SPWD(DEM_path,length=l) for pol in self.polygons]),tqdm(lengths)))
        hghts=np.array([pol.morphological_indices["Cross section height [m]"] for pol in self.polygons]).T
        Sh=np.sum(hghts)
        cor=[]
        for ls in self.spwds:
            Sspwd=np.sum(ls)
            cor.append((self.num_polygons*np.dot(ls,hghts)-Sh*Sspwd)/((self.num_polygons*np.dot(ls,ls.T)-Sspwd*Sspwd)*(self.num_polygons*np.dot(hghts.T,hghts)-Sh*Sh))**0.5)
        if fig is None or ax is None:
            fig,ax=plt.subplots(figsize=figsize)
        if zero_line:
            ax.axhline(ls=zero_line_style,c=zero_line_color,lw=zero_line_thickness)
        bplot=ax.boxplot(self.spwds,tick_labels=lengths,widths=mustache_width,showfliers=outliers,whis=whis,patch_artist=True)
        for patch in bplot['boxes']:
            patch.set(facecolor=box_color)       
        
        npt=len(lengths)
        ax.fill_between(y1=np.degrees(np.arctan(0.001)),y2=np.degrees(np.arctan(0.0015)),x=np.linspace(0.5,0.5+npt),alpha=0.25,label="Range reported in Frezzotti et al. 2002",color=frezzotti_range_color)
        ax.scatter(range(1,1+npt),cor,marker=cor_marker,s=cor_marker_size,color=correlations_color,label="Correlation with CSH",zorder=3,**kwargs)
        ax.legend(bbox_to_anchor=bbox_to_anchor,fontsize=legend_fontsize)
        ax.set_ylabel("B-SPWD ["r'$^\circ$'"] / correlation",fontsize=legend_fontsize)
        ax.set_xlabel("Baseline [km]",fontsize=legend_fontsize)
        ax.set_title(title,fontsize=title_fontsize,y=y)
        ax.tick_params(axis='both',which='both',labelsize=legend_fontsize)
        if export_to is not None:
            my_savefig(fig,export_to)
        
        
    def calculate_climate_param(self,data,ind_name):
        stats=zonal_stats([x.polygon for x in self.polygons], data, stats="mean",affine=self.transform)
        for i in tqdm(range(self.num_polygons)):
            self.polygons[i].morphological_indices[ind_name]=stats[i]['mean']

            
    def latitude_mustache(self,transformer,parameter="Angle with respect to wind [deg]",lat_min=None,lat_max=None,w=2.5,southern_hemisphere=True,fig=None,ax=None,figsize=(10,10),histogram=True,histogram_scale_factor=40.0,hist_bin_width_factor=1,mustache_width_factor=0.5,bar_alpha=0.5):
        if parameter == "Wind angle to NS":
            vals=np.array([pol.get_wind_angle_to_NS(transformer,southern_hemisphere=southern_hemisphere) for pol in self.polygons])
        else:
            vals=np.array([pol.morphological_indices[parameter] for pol in self.polygons])
        lats=np.array(list(map(lambda x: x.get_coordinate(transformer),self.polygons)))*(1-2*int(southern_hemisphere))
        if lat_min is None:
            lat_min=90-np.ceil((90-min(lats))/w)*w*1.1
        if lat_max is None:
            lat_max=90-np.floor((90-max(lats))/w)*w
        lat_windows=np.arange(lat_max-w,lat_min-w*1.1*int(lat_min>min(lats)),-2*w)
        binned=list(map(lambda l: np.extract(abs(lats-l)<w,vals),lat_windows))
        nums=np.array(list(map(lambda x: float(len(x)),binned)))
        nums/=np.sum(nums)/histogram_scale_factor
        if fig is None or ax is None:
            fig,ax=plt.subplots(figsize=figsize)
        if histogram:
            ax.bar(lat_windows*(1-2*int(southern_hemisphere)),nums,alpha=bar_alpha,width=hist_bin_width_factor*2*w)
        ax.boxplot(binned,positions=lat_windows*(1-2*int(southern_hemisphere)),tick_labels=lat_windows*(1-2*int(southern_hemisphere)),widths=mustache_width_factor*2*w)
        if not southern_hemisphere:
            ax.xaxis.set_inverted(True)
        ax.set_xlabel("Latitude ["r'$^\circ$'"]")
        ax.set_ylabel(string_for_label_fixer(parameter))
        
            
    def create_parameter_maps(self,shape,profile,base_path=r'.\parameter_map_'):
        for par in self.indices:
            geom_value = ((geom.polygon,geom.morphological_indices[par]) for geom in  self.polygons)
            rasterized = raster_data(raster=features.rasterize(geom_value, out_shape = shape, transform = self.transform, fill = -5, dtype = rasterio.float32),nodata_value=-5,profile=profile)
            rasterized.write_single_band_tif(base_path+'{}.tif'.format(string_for_filename(par)),dtype=rasterio.float32)

    def pdf_with_sections(self,lines_file,output_path,RR,DEM_path,hillshade,wind_speed,hillshade_all,fig=None,ax=None,params=['Cross section height [m]','Absolute slope ratio','Residual slope ratio','Slope in prevailing wind direction [deg]'],figsize=(15,3),breathe_y=0.2,breathe_x=0.03,line_extr_frac=0.05,cross_sec_space=0.5,cmap=plt.cm.brg_r,cor_space=0.2,cor_space_saver=0.7, wind_RR_cor_space=0,fourier_space=0.3,plot_edges=False,peak_prominence=0.01,fourier_semilog=True,fourier_peaks_only_above=10,position_reference_space=0.2,early_stop=None):
        for path in lines_file, output_path, RR, DEM_path, hillshade, wind_speed:
            path=fine_name_converter(path)
        lines=[]
        regions=[]
        for l in my_open(lines_file).readlines()[1:]:
            spl=l.split(";")
            lines.append(shapely.from_wkt(spl[0][1:-1]))
            regions.append(spl[1][:-1])
        pp = my_PdfPages(output_path)
        if early_stop is None:
            early_stop=len(lines)
        for it in tqdm(range(early_stop)):
            first = Point(lines[it].geoms[0].coords[0])
            last = Point(lines[it].geoms[0].coords[-1])
            x1=first.x
            y1=first.y
            x2=last.x
            y2=last.y
            with my_rasterio_open(DEM_path,mode='r+',driver='GTiff',IGNORE_COG_LAYOUT_BREAK=True) as src:
                if next(src.sample([(x1,y1)]))[0]<next(src.sample([(x2,y2)]))[0]: 
                    fig,ax = self.probe_section(y2,x2,y1,x1,RR,DEM_path,hillshade,wind_speed,hillshade_all,fig=None,ax=None,params=params,figsize=figsize,breathe_y=breathe_y,breathe_x=breathe_x,line_extr_frac=line_extr_frac,cross_sec_space=cross_sec_space,cmap=cmap,cor_space=cor_space,cor_space_saver=cor_space_saver, wind_RR_cor_space=wind_RR_cor_space,fourier_space=fourier_space,plot_edges=plot_edges,peak_prominence=peak_prominence,fourier_semilog=fourier_semilog,fourier_peaks_only_above=fourier_peaks_only_above,position_reference_space=position_reference_space,region_name=regions[it]+", section {}".format(it%10+1))
                else:
                    fig,ax = self.probe_section(y1,x1,y2,x2,RR,DEM_path,hillshade,wind_speed,hillshade_all,fig=None,ax=None,params=params,figsize=figsize,breathe_y=breathe_y,breathe_x=breathe_x,line_extr_frac=line_extr_frac,cross_sec_space=cross_sec_space,cmap=cmap,cor_space=cor_space,cor_space_saver=cor_space_saver, wind_RR_cor_space=wind_RR_cor_space,fourier_space=fourier_space,plot_edges=plot_edges,peak_prominence=peak_prominence,fourier_semilog=fourier_semilog,fourier_peaks_only_above=fourier_peaks_only_above,position_reference_space=position_reference_space,region_name=regions[it]+", section {}".format(it%10+1))
            pp.savefig(fig)
            plt.close()
            
        pp.close()
            
    def probe_section(self,y1,x1,y2,x2,RR,DEM_path,hillshade,wind_speed,hillshade_all,fig=None,ax=None,params=None,figsize=(15,3),breathe_y=0.2,breathe_x=0.02,line_extr_frac=0.05,cross_sec_space=0.8,save_to=None,cmap=plt.cm.brg_r,cor_space=0.2,cor_space_saver=0.7, wind_RR_cor_space=0,fourier_space=0.3,plot_edges=False,peak_prominence=0.005,fourier_semilog=True,fourier_peaks_only_above=10,fourier_peaks_only_below=100,position_reference_space=0.1,region_name='',quick_fix=[1,2],horizontal_line_for_WIA=None):
        for path in RR, DEM_path, hillshade, wind_speed, save_to:
            path=fine_name_converter(path)
        if y1>y2:
            from_up_down=True
        else:
            from_up_down=False
        if params is None:
            params=self.indices
        num_subax=len(params)+3
        P1=Point(x1,y1)
        P2=Point(x2,y2)
        line=LineString([P1,P2])
        dunes_of_interest=sorted([poly for poly in self.polygons if shapely.intersects(line,poly.polygon)],key=lambda p: shapely.intersection(line,p.polygon).centroid.y*(1-2*int(from_up_down)))
        probing_points=np.array([shapely.distance(shapely.intersection(line,p.polygon).centroid,P1) for p in dunes_of_interest])/1000
        if fig is None or ax is None:
            fig,ax=plt.subplots(figsize=(figsize[0],figsize[1]*num_subax/(1-wind_RR_cor_space-cor_space)))
        dh=(1-cor_space-wind_RR_cor_space)/num_subax
        h0=dh*breathe_y+wind_RR_cor_space
        plot_h=dh*(1-2*breathe_y)
        for_linspace=int(line.length/100)
        xx=np.linspace(x1,x2,for_linspace) 
        yy=y1+(y2-y1)*(xx-x1)/(x2-x1)
        dist=((xx-x1)**2+(yy-y1)**2)**0.5

        
        sub_ax=ax.inset_axes([0,h0,cross_sec_space,plot_h])
      
            ##### WIND SPEED #######
        
        wind_for_plot=[]
        with my_rasterio_open(wind_speed,mode='r+',driver='GTiff',IGNORE_COG_LAYOUT_BREAK=True) as src:
            for i in range(for_linspace):
                wind_for_plot.append(next(src.sample([(xx[i],yy[i])]))[0])
            sub_ax.plot(dist/1000,wind_for_plot,color=cmap(h0))
            sub_ax.set_xlabel("Distance along cross-section [km]")
            sub_ax.set_ylabel("Average wind speed "r'$\left[\frac{m}{s}\right]$')   
        h0+=dh        
        sub_ax=ax.inset_axes([0,h0,cross_sec_space,plot_h])   
        
        ###### ELEVATION #########

        DEM_for_plot=[]
        with my_rasterio_open(DEM_path,mode='r+',driver='GTiff',IGNORE_COG_LAYOUT_BREAK=True) as src:
            for i in range(for_linspace):
                DEM_for_plot.append(next(src.sample([(xx[i],yy[i])]))[0])
            sub_ax.plot(dist/1000,DEM_for_plot,color=cmap(h0))
            sub_ax.set_xlabel("Distance along cross-section [km]")
            sub_ax.set_ylabel("Elevation [m]")     
        h0+=dh        
        sub_ax=ax.inset_axes([0,h0,cross_sec_space,plot_h])
        
        ###### RESIDUAL RELIEF  #########
        
        RR_for_plot=[]
        with my_rasterio_open(RR,mode='r+',driver='GTiff',IGNORE_COG_LAYOUT_BREAK=True) as src:
            for i in range(for_linspace):
                RR_for_plot.append(next(src.sample([(xx[i],yy[i])]))[0])
            sub_ax.plot(dist/1000,RR_for_plot,color=cmap(h0))
            sub_ax.set_xlabel("Distance along cross-section [km]")
            sub_ax.set_ylabel("Residual Relief [m]")
            if plot_edges:
                for dune in dunes_of_interest:
                    I=shapely.intersection(line,dune.polygon.exterior)
                    if not isinstance(I,Point):
                        I=I.geoms
                        I1=I[0]
                        I2=I[1]
                        sub_ax.plot([((I1.x-x1)**2+(I1.y-y1)**2)**0.5/1000,((I2.x-x1)**2+(I2.y-y1)**2)**0.5/1000],[next(src.sample([(I1.x,I1.y)]))[0],next(src.sample([(I2.x,I2.y)]))[0]],c='black')
                        I3=shapely.intersection(line,dune.main_crestlines)
                        if isinstance(I3,Point):
                            sub_ax.plot([((I1.x-x1)**2+(I1.y-y1)**2)**0.5/1000,((I3.x-x1)**2+(I3.y-y1)**2)**0.5/1000],[next(src.sample([(I1.x,I1.y)]))[0],next(src.sample([(I3.x,I3.y)]))[0]],c='darkred')
                            sub_ax.plot([((I2.x-x1)**2+(I2.y-y1)**2)**0.5/1000,((I3.x-x1)**2+(I3.y-y1)**2)**0.5/1000],[next(src.sample([(I2.x,I2.y)]))[0],next(src.sample([(I3.x,I3.y)]))[0]],c='darkred')
                            sub_ax.scatter(((I3.x-x1)**2+(I3.y-y1)**2)**0.5/1000,next(src.sample([(I3.x,I3.y)]))[0],c='darkred')
        

            
        ###### PARAMETER VALUES ALONG SECTION #######
            
        for ind in params:
            h0+=dh
            sub_ax=ax.inset_axes([0,h0,cross_sec_space,plot_h])
            sub_ax.plot(probing_points,list(map(lambda d: d.morphological_indices[ind],dunes_of_interest)),marker='o',ls='-',color=cmap(h0))
            sub_ax.set_xlabel("Distance along cross-section [km]")
            sub_ax.set_ylabel(string_for_label_fixer(ind))
            if ind=="Angle with respect to wind [deg]" and horizontal_line_for_WIA is not None:
                sub_ax.axhline(horizontal_line_for_WIA,ls="dashed",c="darkred")
           
        ##### FOURIER ##########
        
        vis_space=1-cross_sec_space-2*breathe_x
        sub_ax=ax.inset_axes([cross_sec_space+breathe_x*3,wind_RR_cor_space*(1+breathe_y)+breathe_y*fourier_space*(1-position_reference_space-cor_space-wind_RR_cor_space),vis_space-2*breathe_x,fourier_space*(1-cor_space-position_reference_space-wind_RR_cor_space)*(1-breathe_y)])
        fourier = np.fft.fft(RR_for_plot)
        frequencies = np.fft.fftfreq(for_linspace,np.mean(dist[1:]-dist[:-1]))
        trunc = int(np.ceil(for_linspace/2))
        fourier=fourier[:trunc]
        frequencies=frequencies[:trunc]
        amp=np.absolute(fourier)/for_linspace
        sub_ax.plot(frequencies,amp)
        if fourier_semilog:
            sub_ax.semilogy()
            sub_ax.semilogx()
        pks,_=scp.find_peaks(amp,prominence=peak_prominence)
        sub_ax.set_xlim(left=0.00001,right=0.001)
        sub_ax.set_xlabel("Frequency [m"r'$^{-1}$'"]",fontsize=12)
        sub_ax.set_ylabel("Normalized amplitude",fontsize=12)
        sub_ax.set_title("Residual relief Fourier spectrum",fontsize=15)
        txt="Marked peaks:\n"
        i=0
        for pk in pks[::-1]:
            wl=1/(1000*frequencies[pk])
            if (wl-fourier_peaks_only_above)*(wl-fourier_peaks_only_below)<0:
                txt+="{:.2f} km\n".format(1/(1000*frequencies[pk]))
                sub_ax.scatter(frequencies[pk],amp[pk],c='black',s=20)
                i+=1
        if i>0:
            ylim=sub_ax.get_ylim()
            sub_ax.text(0.000016,2*ylim[0],txt[:-1],bbox={'ec':'black','fc':'white','alpha':0.7})
        
        ###### SECTION VISUAL ############
        
        section_vis_v=1.5*breathe_y+(1-fourier_space)*(1-position_reference_space-cor_space-wind_RR_cor_space)
        sub_ax=ax.inset_axes([cross_sec_space+breathe_x,quick_fix[0]*h0+wind_RR_cor_space*(1+breathe_y)+(1)*fourier_space*(1-position_reference_space-cor_space-wind_RR_cor_space)-breathe_y/2,vis_space,section_vis_v-quick_fix[1]*h0])
        hill_raster=raster_data(from_path=True,input_path=hillshade)
        
        if abs(y1-y2)/abs(x1-x2)>section_vis_v/vis_space:
            if from_up_down:
                u=(1+line_extr_frac)*y1-line_extr_frac*y2   
                d=(1+line_extr_frac)*y2-line_extr_frac*y1   
            else:
                u=(1+line_extr_frac)*y2-line_extr_frac*y1   
                d=(1+line_extr_frac)*y1-line_extr_frac*y2 
            rml=(x1+x2)/2
            w=(u-d)*vis_space/(2*section_vis_v)
            r=rml+w
            l=rml-w
        else:
            if x1>x2:
                r=(1+line_extr_frac)*x1-line_extr_frac*x2   
                l=(1+line_extr_frac)*x2-line_extr_frac*x1   
            else:
                r=(1+line_extr_frac)*x2-line_extr_frac*x1   
                l=(1+line_extr_frac)*x1-line_extr_frac*x2 
            rml=(y1+y2)/2
            w=(r-l)*section_vis_v/(2*vis_space)
            u=rml+w
            d=rml-w
        l_raster=hill_raster.bounds[0]   
        d_raster=hill_raster.bounds[1]   
        r_raster=hill_raster.bounds[2]   
        u_raster=hill_raster.bounds[3]
        crop_y=[min(1,max(0,(l-l_raster)/(r_raster-l_raster))),min(1,max(0,(r-l_raster)/(r_raster-l_raster)))]
        crop_x=[min(1,max(0,(u-u_raster)/(d_raster-u_raster))),min(1,max(0,(d-u_raster)/(d_raster-u_raster)))]
        #print(l,r,u,d,l_raster,r_raster,u_raster,d_raster,crop_x,crop_y)
        hill_raster.crop(crop_x,crop_y).visualize(fig=fig,ax=sub_ax,cbar=False,cmap=plt.cm.gray)
        vis_transform=lambda x: (x+[-hill_raster.transform[2]-max(0,l-l_raster),-hill_raster.transform[5]+max(0,u_raster-u)])*np.array([1/hill_raster.transform[0],-1/hill_raster.transform[0]])
        shplt.plot_line(shapely.transform(line,vis_transform),lw=2,color="black",ax=sub_ax)
        sub_ax.set_title('')
        sub_ax.annotate("A", fontsize=15,xy=(vis_transform(np.array([(1+line_extr_frac/2)*x1-line_extr_frac*x2/2,(1+line_extr_frac/2)*y1-line_extr_frac*y2/2]))))
        sub_ax.annotate("B", fontsize=15,xy=(vis_transform(np.array([(1+line_extr_frac)*x2-line_extr_frac*x1,(1+line_extr_frac)*y2-line_extr_frac*y1]))))
        ax.text(0,wind_RR_cor_space-0.01,"A",fontsize=18,transform=ax.transAxes)
        ax.text(cross_sec_space,wind_RR_cor_space-0.01,"B",fontsize=18,transform=ax.transAxes)
        ax.axis('off')
        vis_space=cor_space-breathe_x
        cropped=dune_polygons(None,None,None,None,None,for_copy=self)
        cropped.polygons=dunes_of_interest
        cropped.num_polygons=len(cropped.polygons)
        l_trans,d_trans=vis_transform(np.array([l,d]))
        r_trans,u_trans=vis_transform(np.array([r,u]))
        pth=Path(list(zip([l_trans,r_trans,r_trans,l_trans,l_trans],[d_trans,d_trans,u_trans,u_trans,d_trans])))
        patch = PathPatch(pth, transform=sub_ax.transData, alpha=0.5)   
        cropped.visualize(fig=fig, ax=sub_ax,main_crests=True,vis_transform=vis_transform,main_crest_color='darkred',fill=True,clip_patch=patch)    
        sub_ax.set_xlim(l_trans,r_trans)
        sub_ax.set_ylim(d_trans,u_trans)
        
        ##### WIND RR Cross-correlation (legacy) ########
        
        if(wind_RR_cor_space>0):
            sub_ax=ax.inset_axes([0,wind_RR_cor_space*breathe_y,1,wind_RR_cor_space*(1-3*breathe_y)])
            corr = scp.correlate(RR_for_plot,wind_for_plot, mode='same') / (scp.correlate(RR_for_plot,RR_for_plot, mode='valid')*scp.correlate(wind_for_plot,wind_for_plot, mode='valid'))**0.5
            sub_ax.plot(dist/1000-max(dist)/2000,corr,color='navy')
            sub_ax.set_xlabel("Shift [m]")
            sub_ax.set_ylabel("Cross-corelation")
            sub_ax.set_title("Cross-corelation between Residual Relief and wind speed")
            
        ###### CORRELATIONS OF PARAMETERS ALONG SECTION #######
            
        if cor_space>0:
            sub_ax=ax.inset_axes([0,1-cor_space*(1-breathe_y),cor_space_saver,vis_space*cor_space_saver])
            cropped.compute_correlation_matrix(params=params,fig=fig,ax=sub_ax)
        
        #### POSITION ON THE MAP ######
        
        newax = ax.inset_axes([0.5*(cross_sec_space+breathe_x+1-position_reference_space),1-cor_space*(1-breathe_y)-position_reference_space,position_reference_space,position_reference_space])
        hillshade_all.visualize(fig=fig,ax=newax,cbar=False,cmap=plt.cm.gray,title='')
        #newax.scatter(((x1+x2)/2-hillshade_all.transform[2])/hillshade_all.res,-((y1+y2)/2-hillshade_all.transform[5])/hillshade_all.res,c='red',s=100,marker="*")
        xs=list(map(lambda x: (x-hillshade_all.transform[2])/hillshade_all.res,[l,r,r,l,l,x1,x2]))
        ys=list(map(lambda x: -(x-hillshade_all.transform[5])/hillshade_all.res,[d,d,u,u,d,y1,y2]))
        newax.plot(xs[:-2],ys[:-2],c='red',lw=0.5)
        newax.plot(xs[-2:],ys[-2:],c='black',lw=0.5)
        newax.set_title(region_name)
        newax.axis('off')
        #shplt.plot_line(shapely.transform(line,hillshade_all.transform),lw=10,color="black",ax=newax)
        if save_to is not None:
            my_savefig(fig,save_to)
        return fig, ax
    
    def write_for_QGIS(self,out_path,params=None):

        if params is None:
            params=self.indices
        num_params=len(params)   
        out=my_open(out_path,'w')
        out.write("WKT; ")
        for par in params:
            out.write("{}; ".format(par))
        out.write("\n")
        for poly in tqdm(self.polygons):
            out.write("{}; ".format(shapely.to_wkt(poly.polygon)))
            for i in range(num_params):
                out.write("{:f}; ".format(poly.morphological_indices[params[i]]))
            out.write("\n")
        out.close()
    
    def parameter_table(self,region_file,global_too=True,params=None,number_of_dunes=True,NNI=True,medians=True):

        if params is None:
            params=self.indices
        regions=regions_for_study(region_file,self)
        num_params=len(params)   
        string="Parameter & "+"Global & "*int(global_too)
        for region in regions.regions:
            string+="{} & ".format(region.name)
        string=string[:-2]
        string+='\\ '
        string+='\hline '
        if number_of_dunes:
            string+="Number of dunes & "
            if global_too:
                string+="{} & ".format(self.num_polygons)
            for region in regions.regions:
                string+="{} & ".format(region.num_dunes)
        string=string[:-2]
        string+='\\ '
        string+='\hline '
        for i in tqdm(range(num_params)):
            string+="{} & ".format(params[i])
            if medians:
                fun = np.nanmedian
            else:
                fun = np.mean
            if global_too:
                string+="${:.{}f}$ & ".format(fun(np.array([x.morphological_indices[params[i]] for x in self.polygons])),1+3*int(params[i]=='Aspect ratio')+int(params[i]=='Circularity')+int(params[i]=='Sinuosity')+int(params[i]=='Slope in prevailing wind direction [deg]')+int(params[i]=='Cross section width')+int(params[i]=='NRSD'))
            for region in regions.regions:
                polys=self.limit_to_ROI(region.geom).polygons
                string+="${:.{}f}$ & ".format(fun(np.array([x.morphological_indices[params[i]] for x in polys])),1+3*int(params[i]=='Aspect ratio')+int(params[i]=='Circularity')+int(params[i]=='Absolute slope ratio')+int(params[i]=='Residual slope ratio')+int(params[i]=='Sinuosity')+int(params[i]=='Slope in prevailing wind direction [deg]')+int(params[i]=='Cross section width')+int(params[i]=='NRSD'))
            string=string[:-2]
            string+='\\ '
            string+='\\hline '
        if NNI:
            string+="NNI & "
            for region in regions.regions:
                polys=self.limit_to_ROI(region.geom).polygons
                centers=[]
                for x in polys:
                    centers+=x.main_crestline_sample_points
                multipoint_centers=shapely.MultiPoint(centers)
                distances=np.array([shapely.distance(p,shapely.ops.nearest_points(p,shapely.difference(multipoint_centers,p))[1]) for p in tqdm(centers)])
                A=multipoint_centers.minimum_rotated_rectangle.area
                string+="${:.3f}$ & ".format(np.mean(distances)*2*(len(centers)/A)**0.5)                

        return string
            
    
    def parameter_maps_hexagonal_grid(self,hexagons,out_path=r'.\hexagonal_parameter_map.csv',params=None,number_of_dunes=True):

        if params is None:
            params=self.indices
        num_params=len(params)   
        out=my_open(out_path,'w')
        out.write("WKT; ")
        for par in params:
            out.write("{}; ".format(par))
        if number_of_dunes:
            out.write("Number of dunes\n")
        else:
            out.write("\n")
        for hexagon in tqdm(hexagons):
            polys=self.limit_to_ROI(hexagon).polygons
            if len(polys)>0:
                #for l in geom_values:
                   # l.append((hexagon,-5))
                out.write("{}; ".format(shapely.to_wkt(hexagon)))
                for i in range(num_params):
                    #geom_values[i].append((hexagon,np.mean(np.array([x.morphological_indices[params[i]] for x in polys]))))
                    out.write("{}; ".format(np.mean(np.array([x.morphological_indices[params[i]] for x in polys]))))
                if number_of_dunes:
                    out.write("{}\n".format(len(polys)))
                    #geom_values[-1].append((hexagon,len(polys)))
                else:
                    out.write("\n")
            else:
                out.write("{}; ".format(shapely.to_wkt(hexagon))+"-1; "*num_params+"0;"*int(number_of_dunes)+"\n")
        out.close()
    
    def visualize(self,fig=None,ax=None,figsize=(15,10),all_crests=False, main_crests=False,points_along=False,tangent_lines=False,cross_sections=False,vis_transform=None,main_crest_color='yellow',return_artists=False,alpha_vec=1,clip_patch=None,fill=False,height_dependent_fill=None,save_to=None,thick_crests=False,thick_crest_thickness=2,show_along_wind_sections=False):

        if fig is None or ax is None:
            fig,ax=plt.subplots(figsize=figsize)

        if return_artists:
            art=[]
            for geom in self.polygons:
                art+=geom.visualize(fig=fig,ax=ax,all_crests=all_crests, main_crests=main_crests,points_along=points_along,tangent_lines=tangent_lines,cross_sections=cross_sections,report=False,vis_transform=vis_transform,main_crest_color=main_crest_color,return_artists=True,alpha_vec=alpha_vec,clip_patch=clip_patch,fill=fill,height_dependent_fill=height_dependent_fill,thick_crests=thick_crests,thick_crest_thickness=thick_crest_thickness,show_along_wind_sections=show_along_wind_sections)   
            return art
        else:
            for geom in self.polygons:
                geom.visualize(fig=fig,ax=ax,all_crests=all_crests, main_crests=main_crests,points_along=points_along,tangent_lines=tangent_lines,cross_sections=cross_sections,report=False,vis_transform=vis_transform,main_crest_color=main_crest_color,alpha_vec=alpha_vec,clip_patch=clip_patch,fill=fill,thick_crests=thick_crests,thick_crest_thickness=thick_crest_thickness,show_along_wind_sections=show_along_wind_sections)
            if save_to is not None:
                fig.savefig(save_to)
    
    def plot_histogram(self,parameter,nbins=100,fig=None,ax=None,figsize=(6,5),reject_highest=[0],actually_lowest=False,mark_missing=False,title_fontweight='normal',mark_median=False,color="royalblue",planet_code=r'$\oplus$',median_mark_shift_factor=0.1,median_mark_vertical_pos=0.5,title_fontsize=16,median_text_fontsize=15,tick_label_fontsize=15,legend_text_fontsize=15,y=1.01,Earth_with_Mars=None,color_Mars='maroon',planet_code_Mars=r'$\mars$',nbins_Mars=10,reverse_order=False,extra_median_horizontal_shift_factors=[1,1],extra_median_vertical_shift_factors=[1,1],ax2=None,make_it_two=False,subfigure_tag=None,subfigure_tag_left=True,export_to=None,bbox_to_anchor=(0.6,0.7),actually_lowest_Mars=False,actually_a_bit_of_both_on_Mars=False,fig_suptitle=True):

        if fig is None or ax is None:
            if make_it_two:
                fig, (ax, ax2) = plt.subplots(1, 2,sharey=True,figsize=figsize)
            else:
                fig, ax = plt.subplots(figsize=figsize)
        
        objs_for_plotting=[self]
        if Earth_with_Mars is not None:
            objs_for_plotting.append(Earth_with_Mars)
        
        colors=[color,color_Mars]
        planet_codes=[planet_code,planet_code_Mars]
        nbinss=[nbins,nbins_Mars]
        actually_lowestt=[actually_lowest,actually_lowest_Mars]
        if ax2 is None:
            axs=[ax,ax]
        else:
            axs=[ax,ax2]
        if reverse_order:
            for lst in objs_for_plotting,colors,planet_codes,nbinss,reject_highest,actually_lowestt:
                lst=lst.reverse()
        it=0
        tit='{}'.format(string_for_label_fixer(parameter))
        add_for_tit=""
        meds=[]
        stds=[]
        xlims=[]
        ylims=[]
        for obj in objs_for_plotting:
            if parameter=="Precipitation / ice condensation "r'$\left[\frac{mm}{day/sol}\right]$':
                if it==int(reverse_order):
                    vals = np.array([pol.morphological_indices["Total precipitation m/day"]*1000 for pol in obj.polygons])
                else:
                    vals = np.array([pol.morphological_indices["Water ice condensation [mm/sol]"] for pol in obj.polygons])
            elif parameter=="Area [km2]":
                vals = np.array([pol.morphological_indices[parameter]**0.5 for pol in obj.polygons])    
            else:
                vals = np.array([pol.morphological_indices[parameter] for pol in obj.polygons])
            vals=sorted(vals,key=lambda v: (1-2*int(actually_lowestt[it]))*v)
            meds.append(np.nanmedian(np.array(vals)))
            if parameter=="Area [km2]":
                stds.append(np.nanstd(np.array(vals)**2))
            else:    
                stds.append(np.nanstd(np.array(vals)))
            if reject_highest[it]>0: 
                true_max=vals[-1]
                if actually_a_bit_of_both_on_Mars:
                    vals=vals[int(len(vals)*reject_highest[it]/2):int(-len(vals)*reject_highest[it]/2)]
                else:
                    vals=vals[:int(-len(vals)*reject_highest[it])]
            add_for_tit+="{}{}: "r'${}{}{}$'.format(int(it==1)*", ",planet_codes[it],int(reject_highest[it]>0)*("\\overline{"*int(not actually_lowestt[it] or (actually_a_bit_of_both_on_Mars and it==int(not reverse_order)))+"\\underline{"*int(actually_lowestt[it]) or  (actually_a_bit_of_both_on_Mars and it==int(not reverse_order))),int(100*(1-reject_highest[it])),"\%"+"}"*(int(reject_highest[it]>0)+int(it==1 and actually_a_bit_of_both_on_Mars)))
            axs[it].hist(vals,bins=nbinss[it],alpha=0.5,color=colors[it],label="{}, {} bins".format(planet_codes[it],nbinss[it]))
            axs[it].hist(vals,bins=nbinss[it],histtype='step',color=colors[it])
            if make_it_two:
                if actually_lowestt[it]:
                    axs[it].set_xlim(vals[-1],vals[0])
                else:
                    axs[it].set_xlim(vals[0],vals[-1])
            xlims.append([vals[0],vals[-1]])
            ylims.append(ax.get_ylim())
            if parameter=="Area [km2]":
                tcks=ax.get_xticks()
                ax.set_xticks(tcks,labels=list(map(lambda x:"{:.0f}".format(x**2),tcks)))

            it+=1
        if Earth_with_Mars is None:
            xlims.append(xlims[-1])
        if parameter == "Area [km2]":
            add_for_tit+=", bins in "r'$\sqrt{}$'
        if not make_it_two:
            if actually_lowest:
                ax.set_xlim(min(xlims[0][1],xlims[1][1]),max(xlims[0][0],xlims[1][0]))
            else:
                ax.set_xlim(min(xlims[0][0],xlims[1][0]),max(xlims[0][1],xlims[1][1]))
        """
        if reject_highest>0 and mark_missing:

            ax.annotate ("Max ({})".format(planet_code)*int(not actually_lowest)+"Min ({})".format(planet_code)*int(actually_lowest)+": {:.1f}".format(true_max),xy=((int(actually_lowest)+(1-2*int(actually_lowest))*0.95)*xlim[1]+(int(actually_lowest)+(1-2*int(actually_lowest))*0.05)*xlim[0],0.8*ylim[1]+0.2*ylim[0]),xytext=((int(actually_lowest)+(1-2*int(actually_lowest))*0.76)*xlim[1]+(int(actually_lowest)+(1-2*int(actually_lowest))*0.24)*xlim[0],0.9*ylim[1]+0.1*ylim[0]),horizontalalignment=int(actually_lowest)*'right'+int(not actually_lowest)*'left',arrowprops={"arrowstyle":"->"},transform=ax.transAxes)
        """
        if mark_median:
            it=0
            for obj in objs_for_plotting:
                axs[it].axvline(meds[it],ls='dashed',color="black")
                precis=1+3*int(parameter=='Aspect ratio')+int(parameter=='Wind constancy')+int(parameter=='NASD')+int(parameter=='NRSD')+int(parameter=='Circularity')+int(parameter=='Sinuosity')+int(parameter=='Slope in prevailing wind direction [deg]')+int(parameter=='Slope along normal [deg]')+int(parameter=='Cross section width')-int(parameter=='Temperature 2m above surface [K]')-int(parameter=='Surface pressure [Pa]')+2*int(parameter=="Precipitation / ice condensation "r'$\left[\frac{mm}{day/sol}\right]$')
                axs[it].text(meds[it]+extra_median_horizontal_shift_factors[it]*median_mark_shift_factor*(xlims[it][1]-xlims[it][0]),extra_median_vertical_shift_factors[it]*median_mark_vertical_pos*(ylims[it][0]+ylims[it][1]),"{} median: {:.{}f}\n{} st dev: {:.{}f}".format(planet_codes[it],meds[it]**(1+int(parameter=="Area [km2]")),precis,planet_codes[it],stds[it],precis),bbox={'ec':'black','fc':'white','alpha':0.7},ha="left",fontsize=median_text_fontsize)
                it+=1
        if make_it_two:
            ax.spines['right'].set_visible(False)
            ax2.spines['left'].set_visible(False)
            ax.yaxis.tick_left()
            ax.tick_params(labelright='off')
            ax2.yaxis.tick_right()
            d = .015  # how big to make the diagonal lines in axes coordinates
            # arguments to pass plot, just so we don't keep repeating them
            kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
            ax.plot((1-d, 1+d), (-d, +d), **kwargs)
            ax.plot((1-d, 1+d), (1-d, 1+d), **kwargs)
            kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
            ax2.plot((-d, +d), (1-d, 1+d), **kwargs)
            ax2.plot((-d, +d), (-d, +d), **kwargs)
            fig.subplots_adjust(wspace=0.045)
        if subfigure_tag is not None:
            fc=0.1
            fcy=0.55
            fcx=0.5
            x1,x2=axs[int(not subfigure_tag_left)].get_xlim()
            y1,y2=axs[int(not subfigure_tag_left)].get_ylim()
            if subfigure_tag_left:
                print(x1,x1*(1-fc*(1+int(make_it_two)))+fc*(1+int(make_it_two))*x2,x2)
                axs[0].fill_between(np.linspace(x1,x1*(1-fc*(1+int(make_it_two)))+fc*(1+int(make_it_two))*x2),y1=(1-fc)*y2+fc*y1,y2=y2,color="black")
                axs[0].text(x1*(1-fc*(1+int(make_it_two))*fcx)+fc*(1+int(make_it_two))*x2*fcx,(1-fc*fcy)*y2+fc*y1*fcy,subfigure_tag,c="white",fontsize=14,ha="center",va="center",fontweight='bold')
            else:
                axs[1].fill_between(np.linspace(x2,x2*(1-fc*(1+int(make_it_two)))+fc*(1+int(make_it_two))*x1),y1=(1-fc)*y2+fc*y1,y2=y2,color="black")
                axs[1].text(x2*(1-fc*(1+int(make_it_two))*fcx)+fc*(1+int(make_it_two))*x1*fcx,(1-fc*fcy)*y2+fc*y1*fcy,subfigure_tag,c="white",fontsize=16,ha="center",va="center",fontweight='bold')
            axs[int(not subfigure_tag_left)].set_xlim(x1,x2)
            axs[int(not subfigure_tag_left)].set_ylim(y1,y2)
        if fig_suptitle:
            fig.legend(bbox_to_anchor=bbox_to_anchor,fontsize=legend_text_fontsize)
            fig.suptitle(tit+r'{}{}{}'.format(" (",add_for_tit,")"),fontweight=title_fontweight,fontsize=title_fontsize,y=y)
            for axx in axs:
                axx.tick_params(axis='x',which='both',labelsize=tick_label_fontsize)
            ax.tick_params(axis='y',which='both',labelsize=tick_label_fontsize)
        if not make_it_two:
            ax.yaxis.set_ticks_position('both')
        if export_to is not None:
            my_savefig(fig,export_to)


   
    def hexagon_quality_filter(self,hexagons,info_path,level=1,output_hexagon_file_path=None,output_dune_file_path=None):
        quality_file=my_open(info_path,'r').readlines()
        if output_hexagon_file_path is not None:
            output_hexagon_file=my_open(output_hexagon_file_path,'w')
            output_hexagon_file.write("WKT; QUALITY\n")
        cropped=dune_polygons(None,None,None,None,None,for_copy=self)
        i=0
        quality_list=[]
        surviving_hexagons=[]
        for hexagon in hexagons:
            hex_q=int(quality_file[i].split(" ")[1])
            if output_hexagon_file_path is not None:
                output_hexagon_file.write("{}; {}\n".format(shapely.to_wkt(hexagon),hex_q))
            if hex_q>level:
                quality_list.append(hex_q)
                surviving_hexagons.append(hexagon)
            else:
                cropped=cropped.limit_to_ROI(hexagon,inverse=True)
            i+=1
        if output_dune_file_path is not None:
            cropped.export_to_file(output_dune_file_path)
        return cropped, surviving_hexagons, quality_list

    def limit_to_ROI(self,ROI,inverse=False):
        cropped=dune_polygons(None,None,None,None,None,for_copy=self)
        if inverse:
            cropped.polygons=[poly for poly in cropped.polygons if not shapely.within(poly.polygon,ROI)]
        else:
            try:
                cropped.polygons=[poly for poly in cropped.polygons if shapely.intersects(ROI,poly.polygon)]
            except:
                print(ROI)
        cropped.num_polygons=len(cropped.polygons)
        return cropped

    def plot_parameter_evolution(self,hexagons,info_path,params=None,fig=None,ax=None,figsize=(6,5),ncols=4,save_to=None,y=0.91,suptitle_fontsize=18,vertical_breathe=4,horizontal_breathe=4):
        if params is None:
            params=self.indices
        num_params=len(params)
        vals=[]
        errs=[]
        for i in range(num_params):
            anal=np.array([dune.morphological_indices[params[i]] for dune in self.polygons])
            vals.append([np.mean(anal)])
            errs.append([np.std(anal)])
        ls=np.arange(1,6)
        for l in tqdm(ls):
            cropped,surviving_hexagons,quality_list = self.hexagon_quality_filter(hexagons,info_path,level=l)
            for i in range(num_params):
                anal=np.array([dune.morphological_indices[params[i]] for dune in cropped.polygons])
                vals[i].append(np.mean(anal))
                errs[i].append(np.std(anal))
                
        nrows=int(np.ceil(num_params/ncols))
        
        if fig is None or ax is None:
            fig, ax = plt.subplots(nrows=nrows,ncols=ncols,figsize=(figsize[0]*ncols+horizontal_breathe,figsize[1]*nrows+vertical_breathe))  

        for i in range(num_params):
            ax[i//ncols][i%ncols].plot(range(6),vals[i],'o',linestyle='-')
            #ax[i//ncols][i%ncols].errorbar(range(6),vals[i],fmt='o',linestyle='-',yerr=errs[i])
            ax[i//ncols][i%ncols].set_title(params[i])
            ax[i//ncols][i%ncols].set_xlabel("Rejection level")
            ax[i//ncols][i%ncols].set_ylabel("Mean value")

        fig.suptitle("Evolution of morphological parameter mean values upon rejection of poorly mapped hexagons",y=y,fontsize=suptitle_fontsize)
        if save_to is not None:
            my_savefig(fig,save_to)
        
    
    def plot_all_histograms(self,fig=None,ax=None,figsize=(6,5),nbins=100,ncols=4,save_to=None,params=None,reject_highest=[0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.05,0,0.01,0.01,0,0,0,0,0,-0.01],fontweights=[],mark_medians=True,color="royalblue",planet_code="Global",median_mark_shift_factor=0.1):
        if params is None:
            params=self.indices
            n_par=self.n_par
        else:
            n_par=len(params)
        if reject_highest is None:
            reject_highest=[0]*n_par
        fontweights+=['normal']*(n_par-len(fontweights))
        nrows=int(np.ceil(n_par/ncols))
        
        if fig is None or ax is None:
            fig, ax = plt.subplots(nrows=nrows,ncols=ncols,figsize=(figsize[0]*ncols,figsize[1]*nrows))

        for i in tqdm(range(n_par)):
            self.plot_histogram(params[i],nbins=nbins,fig=fig,ax=ax[i//ncols][i%ncols],reject_highest=abs(reject_highest[i]),actually_lowest=int(reject_highest[i]<0),title_fontweight=fontweights[i],mark_median=mark_medians,color=color,planet_code=planet_code,median_mark_shift_factor=median_mark_shift_factor)

        if n_par%ncols!=0:
            for i in range(ncols-n_par%ncols):
                ax[-1][-1-i].axis('off')
            
        if save_to is not None:
            my_savefig(fig,save_to)

    def PCA(self,out_path,params=None,n_components=3,correlations_with_PCA_save_to=None,reject_highest=[0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.05,0,0.01,0.01,0,0,0,0,0,-0.01],cor_precision=3,cor_lower_part=True,cor_fontsize=12,chessboard_coords=False):
        vals, params, num_params=self.matrix_vals(params=params,normalize=True)
        #print(vals.shape)
        pca = PCA(n_components=n_components)
        vals_new=pca.fit_transform(vals.T)
        print(pca.explained_variance_ratio_)
        j=0
        for dune in self.polygons:
            #pca_params=pca.transform(np.array(list(map(lambda x: dune.morphological_indices[x],params))).reshape(-1,1))
            for i in range(n_components):
                dune.morphological_indices["PCA{}".format(i+1)]=vals_new[j][i]
            j+=1
        contributions=pca.inverse_transform(np.eye(n_components))
        """
        print(contributions.shape)
        for j in range(n_components):
            print("PCA{}:\n".format(j+1))
            for i in range(len(params)):
                print("     {}: {}\n".format(params[i],contributions[j][i]))
        """
        self.write_for_QGIS(out_path,params=["PCA{}".format(x) for x in range(1,n_components+1)])
        self.compute_correlation_matrix(params=self.indices+["PCA{}".format(x) for x in range(1,n_components+1)],precision=cor_precision,include_lower_part=cor_lower_part,save_to=correlations_with_PCA_save_to,reject_highest=reject_highest,num_fontsize=cor_fontsize,chessboard_coords=chessboard_coords)
            
    def pdf_with_three_parameter_plots(self,out_file,regions,Martian,params=[],reject=[],cmaps=[],splitter=140):
        npar=len(params)
        cmaps+=[plt.cm.Reds]*(npar-len(cmaps))
        reject+=[[0.1,0.1]]*(npar-len(reject))
        reject=np.array(reject)
        vals=self.matrix_vals(params=params)[0]
        sorted_vals=np.array(list(map(lambda i: sorted(vals[i]),range(len(params)))))
        ranges=list(map(lambda i: [sorted_vals[i,int(reject[i,0]*self.num_polygons)],sorted_vals[i,int(-reject[i,1]*self.num_polygons)]],range(npar)))
        page_it=0
        part_it=1
        for cmb in tqdm(list(combinations(range(npar),2))):
            for j in range(npar):
                if not j in cmb:
                    if page_it%splitter==0:
                        pp = my_PdfPages(out_file+'_part_{}.pdf'.format(part_it))                    
                    fig,ax=plt.subplots(figsize=(20,10))
                    sub_ax=ax.inset_axes([-0.05,0,0.35,1])
                    self.three_param_plot(fig=fig,ax=sub_ax,x_param=params[cmb[0]],y_param=params[cmb[1]],z_param=params[j],vals=vals[np.array([cmb[0],cmb[1],j])],cbar_pad=0.5,dot_size=0.5,x_range=ranges[cmb[0]],y_range=ranges[cmb[1]],z_range=ranges[j],cmap=cmaps[j],cbar_labelsize=12)
                    sub_ax.set_title("Antarctica (all)")
                    it=0
                    for region in regions:
                        sub_ax=ax.inset_axes([0.45+0.3*int(it>1),0.5*(it%2),0.2,0.4])
                        self.limit_to_ROI(region.geom).three_param_plot(fig=fig,ax=sub_ax,x_param=params[cmb[0]],y_param=params[cmb[1]],z_param=params[j],vals=None,cbar_pad=0.5,dot_size=0.5,x_range=ranges[cmb[0]],y_range=ranges[cmb[1]],z_range=ranges[j],cmap=cmaps[cmb[2]],cbar_labelsize=12)
                        sub_ax.set_title(region.name)
                        it+=1
                    sub_ax=ax.inset_axes([0.5+0.25*int(it>1),0.5*(it%2),0.15,0.4]) 
                    if "Total precipitation m/day" in [params[cmb[0]],params[cmb[1]],params[j]]:
                        sub_ax.annotate("N/A",xy=(0.5,0.5),transform=ax.transAxes,fontsize=20)
                        sub_ax.axis('off')
                    else:
                        Martian.three_param_plot(fig=fig,ax=sub_ax,x_param=params[cmb[0]],y_param=params[cmb[1]],z_param=params[j],vals=None,cbar_pad=0.5,dot_size=1,x_range=None,y_range=None,z_range=None,cmap=cmaps[j],cbar_labelsize=12)
                    sub_ax.set_title("MNPC")
                    ax.axis('off')
                    pp.savefig(fig)
                    plt.close()
                    page_it+=1
                    if page_it%splitter==0:
                        pp.close()
                        part_it+=1
        pp.close()

                

    def three_param_plot(self,x_param="NRSD",y_param="Cross section height [m]",z_param="Angle with respect to wind [deg]",permutate=[0,1,2],point_lvl=False,fig=None,ax=None,figsize=(10,10),cmap=plt.cm.Reds,cbar=True,cbar_labelsize=18,cbar_pad=0.05,dot_size=1,labelsize=20,vals=None,x_range=None,y_range=None,z_range=None,cbar_below=False,x_reject=[0.1,0.1],y_reject=[0.1,0.1],z_reject=[0.1,0.1]):
        raw_params=np.array([x_param,y_param,z_param])
        params=raw_params[np.array(permutate)]
        if vals is None:
            vals=self.matrix_vals(params=params,point_lvl=point_lvl)[0]
            if x_range is None or y_range is None or z_range is None:
                sorted_vals=np.array(list(map(lambda i: sorted(vals[i]),range(len(params)))))
                if x_range is None:
                    x_range=[sorted_vals[0,int(x_reject[0]*self.num_polygons)],sorted_vals[0,int(-x_reject[1]*self.num_polygons)]]
                if y_range is None:
                    y_range=[sorted_vals[1,int(y_reject[0]*self.num_polygons)],sorted_vals[1,int(-y_reject[1]*self.num_polygons)]]
                if z_range is None:
                    z_range=[sorted_vals[2,int(z_reject[0]*self.num_polygons)],sorted_vals[2,int(-z_reject[1]*self.num_polygons)]]
                ranges=np.array([x_range,y_range,z_range])
            else:
                raw_ranges=np.array([x_range,y_range,z_range])
                ranges=raw_ranges[np.array(permutate)]
        else:
            vals=vals[np.array(permutate)]
            raw_ranges=np.array([x_range,y_range,z_range])
            ranges=raw_ranges[np.array(permutate)]
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        sc=ax.scatter(np.where((vals[2]-ranges[2][0])*(vals[2]-ranges[2][1])<0,vals[0],np.nan),vals[1],c=vals[2],cmap=cmap,s=dot_size)
        if cbar:
            divider = make_axes_locatable(ax)
            if cbar_below:
                cax = divider.append_axes("bottom", size="5%", pad=cbar_pad)
                bar = fig.colorbar(sc,cax=cax,orientation='horizontal')
                cax.set_xlabel(params[2])
            else:
                cax = divider.append_axes("right", size="5%", pad=cbar_pad)
                bar = fig.colorbar(sc,cax=cax,orientation='vertical')
                cax.set_ylabel(string_for_label_fixer(params[2]),fontsize=cbar_labelsize)
            bar.ax.tick_params(labelsize=cbar_labelsize)

        ax.set_xlim(ranges[0])
        ax.set_ylim(ranges[1])
        ax.set_xlabel(string_for_label_fixer(params[0]),fontsize=labelsize)
        ax.set_ylabel(string_for_label_fixer(params[1]),fontsize=labelsize)
            
    def matrix_vals(self,params=None,normalize=False,point_lvl=False):
        if point_lvl:
            if params is None:
                params=self.point_lvl_indices
            vals=[]
            for x in params:
                vals.append([])
                itt=0
                for pol in self.polygons:
                    itt+=1
                    vals[-1]+=pol.sampling_point_lvl_indices[x]
            vals=np.array(vals)
            num_obj=vals.shape[1]
        else:
            if params is None:
                params=self.indices
            vals=np.array(list(map(lambda x: [pol.morphological_indices[x] for pol in self.polygons],params)))
            num_obj=self.num_polygons
        num_params=len(params)
        if normalize:
            print(vals.shape)
            for i in range(vals.shape[0]):
                vals[i,:]-=np.mean(vals[i,:])
                vals[i,:]/=np.std(vals[i,:])
        return np.where(np.isnan(vals),0,vals),params,num_params,num_obj       
            
    def compute_correlation_matrix(self,params=None,visualize=True,figsize=(16,16),cmap=plt.cm.coolwarm,fig=None,ax=None,save_to=None,precision=3,reject_highest=[],point_lvl=True,include_lower_part=True,num_fontsize=12,labelfontsize=15,chessboard_coords=False):
        vals, params,num_params,num_obj=self.matrix_vals(params=params,point_lvl=point_lvl)
        sorted_vals=np.array(list(map(lambda i: sorted(vals[i]),range(num_params))))
        sums=np.sum(vals,axis=1).reshape(1,num_params)  
        cor=num_obj*np.dot(vals,vals.T)-np.dot(sums.T,sums)
        auto=np.diag(cor).reshape(num_params,1)
        cor/=(np.repeat(auto,num_params,axis=1)*np.repeat(auto.T,num_params,axis=0))**0.5
        reject_highest+=(num_params-len(reject_highest))*[0]
        extr_inf_for_params=list(map(lambda x: int(x!=0)*r' ${}{}{}$'.format("("+"\\overline{"*int(x>0)+"\\underline{"*int(x<0),100*(1-abs(x)),"}\%)"),reject_highest))
        actually_lowest=np.where(np.array(reject_highest)>=0,False,True)
        reject_highest=np.abs(np.array(reject_highest))
        if visualize:
            if fig is None or ax is None:
                fig,ax = plt.subplots(figsize=figsize)
            for i in range(num_params):
                ax.vlines(i,(num_params-i)*int(not include_lower_part),num_params,lw=0.5,color='black')
                ax.hlines(i,(num_params-i)*int(not include_lower_part),num_params,lw=0.5,color='black')
                if include_lower_part:
                    sub_ax=ax.inset_axes([i/num_params,(num_params-i-1)/num_params,1.0/num_params,1.0/num_params])
                    self.plot_histogram(params[i],fig=fig,ax=sub_ax,reject_highest=[reject_highest[i]],actually_lowest=actually_lowest[i],mark_missing=False,fig_suptitle=False)
                    sub_ax.set_title('')
                    sub_ax.tick_params(axis='x',labelrotation=90,labelsize=labelfontsize)
                    if actually_lowest[i]:
                        sub_ax.set_xlim(sorted_vals[i,int(self.num_polygons*reject_highest[i])],sorted_vals[i,-1])
                    else:
                        sub_ax.set_xlim(sorted_vals[i,0],sorted_vals[i,int(-self.num_polygons*reject_highest[i])-1])
                    if i+1<num_params:
                        sub_ax.axis('off')
                    else:
                        sub_ax.set_yticklabels([])
                for j in range(i+1,num_params):
                    ax.fill_between(np.linspace(j,j+1),num_params-i,num_params-i-1,color=cmap((cor[i,j]+1)/2),alpha=0.5)
                    ax.annotate('{:.{}f}'.format(cor[i,j],precision),xy=(j+0.5,num_params-(i+0.5)),ha='center',transform=ax.transAxes,fontsize=num_fontsize)
                    if include_lower_part:
                        sub_ax=ax.inset_axes([i/num_params,(num_params-j-1)/num_params,1.0/num_params,1.0/num_params])
                        sub_ax.scatter(vals[i],vals[j],s=1,color='black')
                        if actually_lowest[i]:
                            sub_ax.set_xlim(sorted_vals[i,int(self.num_polygons*reject_highest[i])],sorted_vals[i,-1])
                        else:
                            sub_ax.set_xlim(sorted_vals[i,0],sorted_vals[i,int(-self.num_polygons*reject_highest[i])-1])
                        if actually_lowest[j]:
                            sub_ax.set_ylim(sorted_vals[j,int(self.num_polygons*reject_highest[j])],sorted_vals[j,-1])
                        else:
                            sub_ax.set_ylim(sorted_vals[j,0],sorted_vals[j,int(-self.num_polygons*reject_highest[j])-1])
                        if i>0:                    
                            sub_ax.set_yticklabels([])
                        else:
                            sub_ax.tick_params(axis='y',labelsize=labelfontsize)
                        if j+1<num_params:
                            sub_ax.set_xticklabels([])
                        else:
                            sub_ax.tick_params(axis='x',labelrotation=90,labelsize=labelfontsize)
            ax.set_xlim(0,num_params)
            ax.set_ylim(0,num_params)
            ax.set_xticks(0.5+np.arange(num_params))
            ax.set_yticks(0.5+np.arange(num_params))
            params_for_labels=list(map(string_for_label_fixer,params))
            lbl=[params_for_labels[i]+extr_inf_for_params[i] for i in range(num_params)]
            ax.set_xticklabels(lbl, rotation=90,fontsize=labelfontsize)
            ax.set_yticklabels(lbl[::-1], rotation=0,fontsize=labelfontsize)
            ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
            ax.yaxis.tick_right()
            if chessboard_coords:
                for i in range(num_params):
                    for j in range(i+1,num_params):
                        ax.text(j+0.75,num_params-(i+0.85),'{}{}'.format(chr(ord('A') + i),j+1),ha='center',color='dimgrey', transform=ax.transData,fontsize=num_fontsize-4,fontweight='bold')
            if save_to is not None:
                my_savefig(fig,save_to,bbox_inches='tight')
        return cor

    def export_to_file(self,path):
        file=my_open(path,'w')
        for i in range(6):
            file.write('{}\n'.format(self.transform[i]))
        file.write('{}\n'.format(self.res))
        file.write('{}\n'.format(self.WOI))
        inds_for_out=self.polygons[0].morphological_indices.keys()
        file.write('{}\n'.format(len(inds_for_out)))
        nlpi=len(self.point_lvl_indices)
        file.write('{}\n'.format(nlpi))
        for i in inds_for_out:
            file.write('{}\n'.format(i))
        for i in self.point_lvl_indices:
            file.write('{}\n'.format(i))
        file.write('{}\n'.format(self.num_polygons))
        for i in range(self.num_polygons):
            file.write('{}\n'.format(shapely.to_wkt(self.polygons[i].polygon)))
            file.write('{}\n'.format(shapely.to_wkt(self.polygons[i].main_crestlines)))
            file.write('{}\n'.format(shapely.to_wkt(self.polygons[i].crestlines)))
            file.write('{}\n'.format(self.polygons[i].sampling_points))
            file.write('{}\n'.format(self.polygons[i].orientation_vector[0]))
            file.write('{}\n'.format(self.polygons[i].orientation_vector[1]))
            file.write('{}\n'.format(self.polygons[i].wind_direction[0]))
            file.write('{}\n'.format(self.polygons[i].wind_direction[1]))
            for j in range(self.polygons[i].sampling_points):
                file.write('{}\n'.format(shapely.to_wkt(self.polygons[i].main_crestline_sample_points[j])))
                file.write('{}\n'.format(shapely.to_wkt(self.polygons[i].tangent_lines[j])))
                file.write('{}\n'.format(shapely.to_wkt(self.polygons[i].stoss_projection_points[j])))
                file.write('{}\n'.format(shapely.to_wkt(self.polygons[i].lee_projection_points[j])))
                file.write('{}\n'.format(shapely.to_wkt(self.polygons[i].cross_section_lines[j])))
                file.write('{}\n'.format(shapely.to_wkt(self.polygons[i].stoss_along_wind_points[j])))
                file.write('{}\n'.format(shapely.to_wkt(self.polygons[i].lee_along_wind_points[j])))
                file.write('{}\n'.format(self.polygons[i].tangent_vectors[j][0]))
                file.write('{}\n'.format(self.polygons[i].tangent_vectors[j][1]))
                file.write('{}\n'.format(self.polygons[i].local_wind_vectors[j][0]))
                file.write('{}\n'.format(self.polygons[i].local_wind_vectors[j][1]))
                file.write('{}\n'.format(self.polygons[i].local_widths[j]))
                file.write('{}\n'.format(self.polygons[i].h_values_for_visualization[j]))
                file.write('{}\n'.format(self.polygons[i].x_values_for_visualization[j]))
                file.write('{}\n'.format(self.polygons[i].z1_values_for_visualization[j]))
                file.write('{}\n'.format(self.polygons[i].z2_values_for_visualization[j]))
                file.write('{}\n'.format(self.polygons[i].zz1_values_for_visualization[j]))
                file.write('{}\n'.format(self.polygons[i].zz2_values_for_visualization[j]))
                for k in range(nlpi):
                    file.write('{}\n'.format(self.polygons[i].sampling_point_lvl_indices[self.point_lvl_indices[k]][j]))
            for ind in self.polygons[0].morphological_indices.keys():
                file.write('{}\n'.format(self.polygons[i].morphological_indices[ind]))
        file.close()
        return

    def read_from_file(self,path,DEM_path,indices_reset=['Cummulative Crest Length [km]','Area [km²]','Perimeter [km]','Circularity','Cummulative length of main crests [km]','Cross section width [km]','Crest elongation','Cross section height [m]','Bounding box length [km]','Bounding box width [km]','Bounding box elongation','Height [m]','Sinuosity','Residual slope ratio','Angle with respect to wind [°]','Aspect ratio','Slope in prevailing wind direction','Absolute slope ratio'],investigate_suspicious=False,limit_to=None,transformer=None):
        file=my_open(path,'r').readlines()
        self.transform=Affine(float(file[0]),float(file[1]),float(file[2]),float(file[3]),float(file[4]),float(file[5]))
        self.res=float(file[6])
        self.WOI=float(file[7])
        n=int(file[8])
        npli=int(file[9])
        npli18=18+npli
        self.indices=[]
        self.point_lvl_indices=[]
        current_line=9
        for i in range(n):
            current_line+=1
            self.indices.append(file[current_line][:-1])
        for i in range(npli):
            current_line+=1
            self.point_lvl_indices.append(file[current_line][:-1])
        current_line+=1
        if limit_to is None:
            n=int(file[current_line])
        else:
            n=limit_to
        self.polygons=[]
        for i in tqdm(range(n)):
            nj=int(file[current_line+4])
            point_lvl_dict={}
            for k in range(npli):
                point_lvl_dict[self.point_lvl_indices[k]]=[float(o) for o in file[current_line+27+k:current_line+27+k+nj*npli18:npli18]]   
            self.polygons.append(one_dune_polygon(None,None,None,None,None,DEM_path,None,None,transformer=transformer,investigate_suspicious=investigate_suspicious,all_given={'polygon':shapely.from_wkt(file[current_line+1]),
            'main crestlines':shapely.from_wkt(file[current_line+2]),
            'crestlines':shapely.from_wkt(file[current_line+3]),
            'sampling points':nj,
            'orientation vector': np.array([float(file[current_line+5]),float(file[current_line+6])]),
            'wind vector': np.array([float(file[current_line+7]),float(file[current_line+8])]),                                                                            
            'main crestline sample points': [shapely.from_wkt(file[current_line+9+j*npli18]) for j in range(nj)],
            'tangent_lines': [shapely.from_wkt(file[current_line+10+j*npli18]) for j in range(nj)],
            'stoss projection points': [shapely.from_wkt(file[current_line+11+j*npli18]) for j in range(nj)],
            'lee projection points': [shapely.from_wkt(file[current_line+12+j*npli18]) for j in range(nj)],
            'cross section lines': [shapely.from_wkt(file[current_line+13+j*npli18]) for j in range(nj)],
                                                                                                                                                     'stoss along wind points': [shapely.from_wkt(file[current_line+14+j*npli18]) for j in range(nj)],
            'lee along wind points': [shapely.from_wkt(file[current_line+15+j*npli18]) for j in range(nj)],                                                                                                                                         

            'tangent vectors': [np.array([float(file[current_line+16+j*npli18]),float(file[current_line+17+j*npli18])]) for j in range(nj)],
            'local wind vectors': [np.array([float(file[current_line+18+j*npli18]),float(file[current_line+19+j*npli18])]) for j in range(nj)],
            'local widths': [float(file[current_line+20+j*npli18]) for j in range(nj)],
            'h values': [float(file[current_line+21+j*npli18]) for j in range(nj)],
            'x values': [float(file[current_line+22+j*npli18]) for j in range(nj)],
            'z1 values': [float(file[current_line+23+j*npli18]) for j in range(nj)],
            'z2 values': [float(file[current_line+24+j*npli18]) for j in range(nj)],     
            'zz1 values': [float(file[current_line+25+j*npli18]) for j in range(nj)],
            'zz2 values': [float(file[current_line+26+j*npli18]) for j in range(nj)],                                                                                                                                                    
            'indices': dict(zip(self.indices,list(map(float,file[current_line+9+nj*npli18:current_line+9+nj*npli18+len(self.indices)])))),'point_lvl':point_lvl_dict,'res':self.res,'transform':self.transform, 'WOI':self.WOI
            }))
            current_line=current_line+9+nj*npli18+len(self.indices)-1
        if indices_reset is not None:
            self.indices=indices_reset
        self.n_par=len(self.indices)
        self.num_polygons=len(self.polygons)
        return
            
    def visualize_dunes(self,raster,ind_start=0,jumping=10,nrows=4,ncols=2,figside=5,tighter=0,row_detighten=5,save_to=None,bbox_to_anchor=(0.97,0.98),y=1,suptitle_fontsize=14,report=True,angle_marker_widths=None,angle_marker_heights=None,move_text=None,early_break=None,show_unrotated_too=True,show_along_wind_sections=True,angle_symbol_fontsize=10,angle_annotation_shifter=1.1,point_marker_size=13,extra_shifts=None,scale_bar=False,scale_bar_val_km=1,each_to_its_own=None):
        n_for_loop=ncols*nrows
        if extra_shifts is None:
            extra_shifts = [0]*n_for_loop
        if angle_marker_widths is None:
            angle_marker_widths=[100]*n_for_loop
        if angle_marker_heights is None:
            angle_marker_heights=[1]*n_for_loop
        if move_text is None:
            move_text=[None]*n_for_loop
        if early_break is None:
            early_break=n_for_loop
        col_fac=2+int(show_unrotated_too)+int(show_along_wind_sections)
        ncols=col_fac*ncols
        fig,axs = plt.subplots(figsize=(ncols*figside-tighter+row_detighten,nrows*figside-tighter),nrows=nrows,ncols=ncols)
        if nrows==1:
            axs=[axs]
        for i in np.arange(n_for_loop):

            if i>=early_break:
                axs[(col_fac*i)//ncols][(col_fac*i)%ncols].axis('off')
                axs[(col_fac*i+1)//ncols][(col_fac*i+1)%ncols].axis('off')
            else:
                #axs[(col_fac*i)//ncols][(col_fac*i)%ncols].text(0.96,0.96,"{}".format(1+ind_start+i*jumping), verticalalignment="top",horizontalalignment="right",bbox={'ec':'black','fc':'white','alpha':1},fontsize=10,transform=axs[(col_fac*i)//ncols][(col_fac*i)%ncols].transAxes)
                self.polygons[ind_start+i*jumping].transform=raster.transform
                if show_unrotated_too:
                    ax3=axs[(col_fac*i+2)//ncols][(col_fac*i+2)%ncols]
                else:
                    ax3=None
                if show_along_wind_sections:
                    ax4=axs[(col_fac*i+2+int(show_unrotated_too))//ncols][(col_fac*i+2+int(show_unrotated_too))%ncols]
                else:
                    ax4=None
                if each_to_its_own is not None:
                    each_to_its_own_temp=each_to_its_own+'_{}'.format(i)
                self.polygons[ind_start+i*jumping+np.sum(np.array(extra_shifts[:i+1]))].visualize(fig=fig,ax=axs[(col_fac*i)//ncols][(col_fac*i)%ncols],ax2=axs[(col_fac*i+1)//ncols][(col_fac*i+1)%ncols],ax3=ax3,ax4=ax4,raster=raster,alpha=0.5,report=report,angle_marker_width=angle_marker_widths[i],angle_marker_height=angle_marker_heights[i],move_text=move_text[i],show_unrotated_too=show_unrotated_too,show_along_wind_sections=show_along_wind_sections,angle_symbol_fontsize=angle_symbol_fontsize,angle_annotation_shifter=angle_annotation_shifter,point_marker_size=point_marker_size,scale_bar=scale_bar,scale_bar_val_km=scale_bar_val_km,each_to_its_own=each_to_its_own_temp,figsize=(5*(1+int(each_to_its_own is not None)),5))
        fig.suptitle("Main crest and cross-section analysis for some dune examples",fontsize=suptitle_fontsize,y=y)
        handles = [ plt.Line2D([], [], color='indigo', linewidth=1, label='Dune outlines'),plt.Line2D([], [], color='darkgreen', linewidth=1, label='Crest skeleton'),plt.Line2D([], [], color='yellow', linewidth=1, label='Main crests'), plt.Line2D([], [], color='darkcyan', marker='o',linewidth=0, label='Sampling points'), plt.Line2D([], [], color=plt.cm.PuRd(0.75), linewidth=1, label='Cross section lines'),plt.Line2D([], [], color=plt.cm.copper(0.75), linewidth=1, linestyle='dashed',label='Wind parallel cross section lines'), plt.Line2D([], [], color='lime', marker='o',linewidth=0, label='Lee side projections'), plt.Line2D([], [], color='orange', marker='o',linewidth=0, label='Stoss side projections'),plt.Line2D([], [], color='seagreen', marker='o',linewidth=0, label='Lee side wind parallel projections'),plt.Line2D([], [], color='maroon', marker='o',linewidth=0, label='Stoss side wind parallel projections'),plt.Line2D([], [], color='black', linewidth=1, label='Mean dune directions'),plt.Line2D([], [], color='blue', linewidth=1, label='Mean wind directions')]

        fig.legend(handles=handles,bbox_to_anchor=bbox_to_anchor,title='Legend',title_fontsize=int(2*suptitle_fontsize/3),fontsize=int(suptitle_fontsize/2),ncols=6)
        if save_to is not None:
            my_savefig(fig,save_to)
        return fig, axs        
    
    def precompute_ensemble(self,nbins = 300,alt_indices=['Cross section width [km]','Angle with respect to wind [deg]','Area [km2]','Absolute slope ratio','Residual slope ratio','Cummulative length of main crests [km]','Circularity','Crest elongation','Aspect ratio','Sinuosity','Temperature 2m above surface [K]','Cross section height [m]']):
        precomputed_ensemble_indices=np.array([np.array([dune.morphological_indices[ind] for dune in self.polygons]) for ind in alt_indices])
        precomputed_ensemble_hist=[]
        for i in tqdm(range(int(len(alt_indices)/2))):
            x=np.log10(precomputed_ensemble_indices[2*i])
            y=np.log10(precomputed_ensemble_indices[2*i+1])
            k = gaussian_kde([x,y])
            xi, yi = np.mgrid[
               x.min():x.max():nbins*1j,
               y.min():y.max():nbins*1j
            ]
            zi = k(np.vstack([
               xi.flatten(),
               yi.flatten()
            ])).reshape(xi.shape)
            xx,lbl_xx=convert_ticks(np.min(x),np.max(x))
            yy,lbl_yy=convert_ticks(np.min(y),np.max(y))
            precomputed_ensemble_hist.append([alt_indices[2*i],alt_indices[2*i+1],xi,yi,zi,xx,lbl_xx,yy,lbl_yy])
        return precomputed_ensemble_hist

    def visualization_pdf(self,hillshade,hillshade_all,path,hexagons,wind_data_path,circle_radius=125000,nbins=72,param_plots_breathe=0.1,quality=None,nbins_ensemble = 300,alt_indices=['Cross section width [km]','Angle with respect to wind [deg]','Area [km2]','Absolute slope ratio','Residual slope ratio','Cummulative length of main crests [km]','Circularity','Crest elongation','Aspect ratio','Sinuosity','Temperature 2m above surface [K]','Cross section height [m]']):
        transformer=Transformer.from_crs("EPSG:3031", "EPSG:4326")
        wind_data=wind_data_for_hex_vis(wind_data_path)
        precomputed_ensemble_hist=self.precompute_ensemble(nbins=nbins_ensemble,alt_indices=alt_indices)
        pp = my_PdfPages(path)
        poly_anchor_l=hillshade.bounds[0]
        poly_anchor_r=hillshade.bounds[2]
        poly_anchor_rl=poly_anchor_r-poly_anchor_l
        poly_anchor_d=hillshade.bounds[1]
        poly_anchor_u=hillshade.bounds[3]
        poly_anchor_du=poly_anchor_d-poly_anchor_u
        i=0
        n=len(hexagons)-1
        if quality is None:
            quality=['Not set']*(n+1)
        neighbour_list=[[hexagon2 for hexagon2 in hexagons if shapely.touches(hexagon,hexagon2)] for hexagon in hexagons]
        for hexagon in tqdm(hexagons):
                fig,ax = plt.subplots(figsize=(15,10))
                l=hexagon.bounds[0]
                d=hexagon.bounds[1]
                r=hexagon.bounds[2]
                u=hexagon.bounds[3]
                center_x=(l+r)/2
                center_y=(u+d)/2    
                l=center_x-circle_radius
                r=center_x+circle_radius
                d=center_y-circle_radius
                u=center_y+circle_radius        
                cropped_hill=hillshade.crop([(u-poly_anchor_u)/poly_anchor_du,(d-poly_anchor_u)/poly_anchor_du],[(l-poly_anchor_l)/poly_anchor_rl,(r-poly_anchor_l)/poly_anchor_rl])
                extra_shift=[-(l-poly_anchor_l),-(u-poly_anchor_u)]
                vis_transform=lambda x: (x+[-hillshade.transform[2],-hillshade.transform[5]]+extra_shift)*np.array([1/hillshade.transform[0],-1/hillshade.transform[0]])
                circle = Circle((cropped_hill.x/2,cropped_hill.y/2), circle_radius/hillshade.transform[0], transform=ax.transData,alpha=0.35)
                cropped_detections=self.limit_to_ROI(hexagon)
                x,y=shapely.transform(hexagon.geoms[0],vis_transform).exterior.xy
                ax.plot(x,y,color='black',lw=0.5)
                for hexagon2 in neighbour_list[i]:            
                    x,y=shapely.transform(hexagon2.geoms[0],vis_transform).exterior.xy
                    arrrr=ax.plot(x,y,color='black',lw=0.5,alpha=0.3)[0]
                    arrrr.set_clip_path(circle)
                hill_im=cropped_hill.visualize(fig=fig,ax=ax,cmap=plt.cm.gray,cbar=False,return_im=True)
                hill_im2=cropped_hill.visualize(fig=fig,ax=ax,cmap=plt.cm.gray,cbar=False,return_im=True)
                cropped_detections.visualize(fig=fig,ax=ax,main_crests=True,vis_transform=vis_transform,main_crest_color='darkred',alpha_vec=0.5,clip_patch=circle,fill=True)
                detect_im2=cropped_detections.visualize(fig=fig,ax=ax,main_crests=True,vis_transform=vis_transform,main_crest_color='darkred',return_artists=True,fill=True)
                l2,d2=vis_transform(np.array([l,d]))
                r2,u2=vis_transform(np.array([r,u]))
                ax.set_xlim(l2,r2)
                ax.set_ylim(d2,u2)            
                newax = ax.inset_axes([-0.15,0.75,0.25,0.25])
                hillshade_all.visualize(fig=fig,ax=newax,cbar=False,cmap=plt.cm.gray)
                x,y=shapely.transform(hexagon.geoms[0],vis_transform).exterior.xy
                pth=Path(list(zip(x,y)))
                patch = PathPatch(pth, transform=ax.transData, alpha=0.5)
                hill_im2.set_clip_path(patch)
                r=min(cropped_hill.x,cropped_hill.y)
                r2=max(cropped_hill.x,cropped_hill.y)
                opacity=((np.repeat(np.linspace(-cropped_hill.x/2,cropped_hill.x/2,cropped_hill.x),cropped_hill.y).reshape((cropped_hill.x,cropped_hill.y))**2+np.repeat(np.linspace(-cropped_hill.y/2,cropped_hill.y/2,cropped_hill.y).reshape(cropped_hill.y,1),cropped_hill.x,axis=1).T**2)/((cropped_hill.y/2)**2+(cropped_hill.x/2)**2))
                opacity=(1-opacity)**4
                hill_im.set_clip_path(circle)
                hill_im.set_alpha(opacity)
                for art in detect_im2:
                    art.set_clip_path(patch)
                j=0
                for hehehe in hexagons:
                    x,y=hehehe.geoms[0].exterior.xy
                    newax.plot(-0.5+(np.array(x)-hillshade_all.transform[2])/hillshade_all.res,-0.5+(-np.array(y)+hillshade_all.transform[5])/hillshade_all.res,lw=1,color='black')            
                    j+=1
                if i<n:
                    x,y=hexagons[i].geoms[0].exterior.xy
                    newax.plot(-0.5+(np.array(x)-hillshade_all.transform[2])/hillshade_all.res,-0.5+(-np.array(y)+hillshade_all.transform[5])/hillshade_all.res,lw=1,color='indianred')  
                x,y=hexagon.geoms[0].exterior.xy
                newax.plot(-0.5+(np.array(x)-hillshade_all.transform[2])/hillshade_all.res,-0.5+(-np.array(y)+hillshade_all.transform[5])/hillshade_all.res,lw=1,color='red')         
                lat,lon=transformer.transform(center_x,center_y)
                ax.text(0.39,0.95,'{:.1f}°S {:.1f}°{}'.format(-lat,abs(lon),"E"*int(lon>=0)+"W"*int(lon<0)),fontsize=17, bbox={'ec':'black','boxstyle':'round','fc':'white','alpha':0.95}, transform=ax.transAxes)
                ax.text(0.12,0.13,'# Dunes: {}'.format(cropped_detections.num_polygons),fontsize=11, bbox={'ec':'black','fc':'white','alpha':1}, ha='right',transform=ax.transAxes)  
                newax.set_title('')
                pol_ax = ax.inset_axes([0.92,0.75,0.2,0.2],polar=True)
                wind_data.draw_wind_rose(hexagon,fig=fig,ax=pol_ax)
                ylim=pol_ax.get_ylim()[1]
                num,theta=np.histogram(np.array([np.arctan2(poly.orientation_vector[1]*int(1-2*(poly.orientation_vector[1]<0)),poly.orientation_vector[0]*int(1-2*(poly.orientation_vector[1]<0))) for poly in cropped_detections.polygons]),bins=np.linspace(0,2*np.pi,nbins+1))            
                scl=ylim/max(num)
                for j in range(len(theta)-1):
                    pol_ax.bar((theta[j]+theta[j+1])/2.0, num[j]*scl, width=2*np.pi/nbins, bottom=0,color='steelblue',alpha=0.6)
                    pol_ax.bar(np.pi+(theta[j]+theta[j+1])/2.0, num[j]*scl, width=2*np.pi/nbins, bottom=0,color='steelblue',alpha=0.6)
                nn=len(precomputed_ensemble_hist)/2
                for j in range(int(2*nn)):
                    par_ax = ax.inset_axes([1.25+0.42*(j%2==1),0.1+np.floor(j/2)/nn,0.3,0.3])
                    par_ax.pcolormesh(precomputed_ensemble_hist[j][2],precomputed_ensemble_hist[j][3],precomputed_ensemble_hist[j][4],cmap=plt.cm.GnBu)
                    par_ax.scatter(np.log10(np.array([dune.morphological_indices[precomputed_ensemble_hist[j][0]] for dune in cropped_detections.polygons])),np.log10(np.array([dune.morphological_indices[precomputed_ensemble_hist[j][1]] for dune in cropped_detections.polygons])),s=0.5,color='black',alpha=0.5)
                    par_ax.set_xlabel(precomputed_ensemble_hist[j][0],fontsize=8)   
                    par_ax.set_ylabel(precomputed_ensemble_hist[j][1],fontsize=8)     
                    par_ax.set_xticks(precomputed_ensemble_hist[j][5],precomputed_ensemble_hist[j][6])
                    par_ax.set_yticks(precomputed_ensemble_hist[j][7],precomputed_ensemble_hist[j][8])
                #ax.text(0.05,0.1,i,bbox={'ec':'black','fc':'white','alpha':1}, ha='right',transform=ax.transAxes)
                j=0
                ax.text(0.85,-0.05,"Mean values",fontsize=12,transform=ax.transAxes) 
                for ind in self.indices:
                    ax.text(0.3+0.5*(j%4),-0.05*np.ceil((j+4.5)/4),"{}:".format(ind),fontsize=11, ha='right',transform=ax.transAxes) 
                    ax.text(0.32+0.5*(j%4),-0.05*np.ceil((j+4.5)/4),"{:.{}f}".format(np.mean(np.array([dune.morphological_indices[ind] for dune in cropped_detections.polygons])),1+2*int(ind=='Aspect ratio')),fontsize=11, ha='left',transform=ax.transAxes) 
                    j+=1
                ax.set_title('')
                ax.axis('off')
                newax.axis('off')             
                quality_flag_l=l2*0.05+r2*0.95
                quality_flag_u=u2*0.15+d2*0.85      
                quality_flag_r=r2
                quality_flag_d=u2*0.1+d2*0.9
                if quality[i]=='Not set':
                    ax.fill_between([quality_flag_l,quality_flag_r],quality_flag_d,quality_flag_u,color="gray")
                else:
                    ax.fill_between([quality_flag_l,quality_flag_r],quality_flag_d,quality_flag_u,color=plt.cm.RdYlGn(0.2*(int(quality[i])-1)))
                ax.plot([quality_flag_l,quality_flag_r,quality_flag_r,quality_flag_l,quality_flag_l],[quality_flag_d,quality_flag_d,quality_flag_u,quality_flag_u,quality_flag_d],lw=0.5,color='black')
                fig.subplots_adjust(left=0.1, bottom=0.2, right=0.54, top=0.95, wspace=0, hspace=0)
                pp.savefig(fig)
                i+=1
                plt.close()
        pp.close()
        
class crestlines:

    def __init__(self,input_file,transform):
        crestlines=[]
        self.transform=transform
        lines=my_open(input_file).readlines()
        for l in lines[1:]:
            next_line_candidate=shapely.from_wkt(l[1:-2])
            if isinstance(next_line_candidate,LineString):
                crestlines.append(next_line_candidate)
            else:
                for lll in next_line_candidate.geoms:
                    crestlines.append(lll)
        self.lines=MultiLineString(crestlines)

    def visualize(self,limits=None,poly_anchor_l=0,poly_anchor_u=0,fig=None,ax=None,figsize=(15,10),lw=0.5,c='darkred'):

        if fig is None or ax is None:
            fig,ax=plt.subplot(figsize=figsize)
        if limits is None:
            extra_shift=[0,0]
        else:
            extra_shift=[-(limits.bounds[0]-poly_anchor_l),-(limits.bounds[3]-poly_anchor_u)]



        vis_transform=lambda x: (x+[-self.transform[2],-self.transform[5]]+extra_shift)*np.array([1/self.transform[0],-1/self.transform[0]])

        to_plot=self.lines
        if limits is not None:
            to_plot=shapely.intersection(to_plot,limits)

        for geom in to_plot.geoms:
            shplt.plot_line(shapely.transform(geom,vis_transform),ax=ax,add_points=False,linewidth=lw,color=c)