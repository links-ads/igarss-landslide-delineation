import re
import os
import shutil
import numpy as np
import scipy as sp
from datetime import datetime, timedelta
from calendar import isleap
from haversine import inverse_haversine, Direction, Unit
from pyproj import Geod
import shapely
import geopandas as gpd
from glob import glob
import xarray as xr



def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)



def db_scale(x):
    return 10 * np.log10(x)



def find_buffer_values(area_of_interest, meters):
    """
    Returns the buffer value (in degrees) needed to buffer the area of interest of the given meters.
    NB: area_of_interest must be a polygon in EPSG:4326.
    """
    
    # NB: the GeoJSON specification strongly suggests splitting geometries so that neither of their parts cross the antimeridian,
    # i.e. the 180th meridian. This is because many libraries used to render maps do not correctly handle geometries that cross the antimeridian.
    # https://tools.ietf.org/html/rfc7946#section-3.1.9

    # NB: haversine expects (lat, lon) instead of (lon, lat), i.e. (y, x) instead of (x, y)
    
    minx, miny, maxx, maxy = area_of_interest.bounds
    buffer_x = abs(minx - inverse_haversine((miny, minx), meters, Direction.WEST, unit=Unit.METERS)[1])    
    buffer_y = abs(miny - inverse_haversine((miny, minx), meters, Direction.SOUTH, unit=Unit.METERS)[0])
    return buffer_x, buffer_y



def compute_area_m2(input):
    geod = Geod(ellps="WGS84")
    if isinstance(input, gpd.GeoDataFrame):
        input = input['geometry']
    if isinstance(input, gpd.GeoSeries):
        return input.to_crs(epsg=4326).apply(lambda g: abs(geod.geometry_area_perimeter(g)[0]))
    if isinstance(input, shapely.Polygon):
        return abs(geod.geometry_area_perimeter(input)[0])
    
    # alternative way to compute area: https://gis.stackexchange.com/a/166421
    # (i.e. Albers Equal Area Conic projection)
    # or: https://gis.stackexchange.com/a/339545



def split_item_collection_by_date(item_collection):
    acq_dates = [item.datetime for item in item_collection]

    rng = [[0]]
    for i in range(1,len(acq_dates)):
        d_prev = acq_dates[i-1]
        d_curr = acq_dates[i]
        if d_curr == d_prev:
            rng[-1].append(i)
        else:
            rng.append([i])
    
    return [item_collection[r[0]:r[-1]+1] for r in rng]



def harmonize_to_old_processing_baseline(s):
    # harmonize s2 data post 25 jan 2022 (processing baseline >=04.00) with the old processing
    # see https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a#Baseline-Change
    # NB: the baseline is reported in s.coords['s2:processing_baseline']

    cutoff = datetime(2022, 1, 25)
    offset = 1000   # new reflectance offset introduced in processing baseline 4.0
    bands_to_process = ["B01","B02","B03","B04","B05","B06","B07","B08","B8A","B09","B10","B11","B12"]
    bands_to_process = list(set(bands_to_process) & set(s.band.data.tolist()))

    old = s.sel(time=slice(cutoff))
    new = s.sel(time=slice(cutoff, None)).drop_sel(band=bands_to_process)

    new_harmonized = s.sel(time=slice(cutoff, None), band=bands_to_process).clip(min=offset)
    new_harmonized -= offset

    new = xr.concat([new, new_harmonized], "band").sel(band=s.band.data.tolist())
    return xr.concat([old, new], dim="time")



def remove_empty_folders(path):
    walk = list(os.walk(path, topdown=False))
    for subpath, _, _ in walk:
        if len(os.listdir(subpath)) == 0:
            os.rmdir(subpath)



def copy_and_overwrite(from_path, to_path):
    if os.path.exists(to_path):
        shutil.rmtree(to_path)
    shutil.copytree(from_path, to_path)



def check_equal_basenames(l1, l2):
    l1 = [os.path.basename(l) for l in l1]
    l2 = [os.path.basename(l) for l in l2]
    return set(l1) == set(l2)



def nglob(path):
    return natural_sort(glob(path))



def interpolate_2d(arr, method='linear', nodata=None):
    """ Fill nan values in 2D array (numpy/torch) by interpolating from valid values. """
    shape = arr.shape
    h, w = shape[-2:]
    x = np.arange(0, h)
    y = np.arange(0, w)

    if len(arr.shape) > 2:
        # put together the first dimensions
        arr = arr.reshape(-1, h, w)
        # interpolate each band separately
        arr = np.array([interpolate_2d(band, method, nodata) for band in arr])
        # reshape back to original shape
        return arr.reshape(shape)
        
    #mask invalid values
    arr = arr.astype(float)
    if nodata is not None:
        arr[arr == nodata] = np.nan

    # convert arr to masked array
    arr = np.ma.masked_invalid(arr)
    
    #get only the valid values
    xx, yy = np.meshgrid(x, y)
    x_valid = xx[~arr.mask]
    x_invalid = xx[arr.mask]
    y_valid = yy[~arr.mask]
    y_invalid = yy[arr.mask]
    valid_arr = arr[~arr.mask]

    # NB: interpolator is very slow to build when x_valid and y_valid are many...
    if method == 'linear':
        interp = sp.interpolate.LinearNDInterpolator(list(zip(x_valid, y_valid)), valid_arr.ravel())
    elif method == 'cubic':
        interp = sp.interpolate.CloughTocher2DInterpolator(list(zip(x_valid, y_valid)), valid_arr.ravel())
    elif method == 'nearest':
        interp = sp.interpolate.NearestNDInterpolator(list(zip(x_valid, y_valid)), valid_arr.ravel())
    else:
        raise ValueError(f"Method {method} not supported")
    new_valid_values = interp(x_invalid, y_invalid)
    new_arr = arr.filled()
    new_arr[arr.mask] = new_valid_values

    return new_arr