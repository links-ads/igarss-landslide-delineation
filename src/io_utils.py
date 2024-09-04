import os
from pathlib import Path
import uuid
import numpy as np
import rasterio as rio
import rioxarray as rxr



def read_raster(
    path: Path,
    bands: list[int] = None,
    window: tuple[int, int, int, int] = None,
    return_profile: bool = False,
) -> np.ndarray:
    """Read a raster file as a CHW numpy array using rasterio.

    Args:
        path (Path): Path to the raster file.
        bands (list[int], optional): List of bands to read. Defaults to None.
        window (tuple[int, int, int, int], optional): Window to read. Defaults to None.

    Returns:
        np.ndarray: Raster data.
    """

    # open whole raster (lazily) as a xarray.DataArray
    raster = rxr.open_rasterio(path)

    # assign band names (if given)
    if hasattr(raster, 'long_name'):
        if isinstance(raster.long_name, str):
            long_name = [raster.long_name]  # from string to list
        else:
            long_name = list(raster.long_name)  # from tuple to list
        raster = raster.assign_coords({'band' : long_name})

    # select window
    if window is not None:
        h_start, h_end, w_start, w_end = window
        raster = raster.isel(y=slice(h_start, h_end), x=slice(w_start, w_end))
    
    # select bands
    if bands is not None:
        raster = raster.sel(band=bands)
    
    return raster.values



def write_image(img, path, georef_like=None, crs=None, transform=None, nodata=None, compress=None, descriptions=None):
    """
    Write an image (as a numpy array) to disk as a tiff file.
    The image is of shape CHW.
    Optionally, georeference information can be copied from another image.
    """
    assert isinstance(img, np.ndarray), f"Input image must be a numpy array. Got type {type(img)}"
    assert path.endswith('.tif'), f"Save path must end with .tif. Got {path}"
    assert (georef_like is None) or (crs is None and transform is None), "Only one between georef_like and crs/transform can be specified"
    if (crs is not None) or (transform is not None):
        assert (crs is not None) and (transform is not None), "Both crs and transform must be specified"
    if georef_like is not None:
        assert georef_like.endswith('.tif'), f"georef_like path must end with .tif. Got {georef_like}"
    if nodata:
        assert img.shape[0] == 1, "Input image must be grayscale if nodata is specified"
        assert img.dtype != np.uint8, "Input image must be float or boolean if nodata is specified"
    if img.dtype == bool:
        img = img.astype(np.float32)
    if descriptions is not None:
        assert isinstance(descriptions, list), "descriptions must be a list. Got type {type(descriptions)}"
    
    # create metadata for the image
    meta = {
        'height': img.shape[1],
        'width': img.shape[2],
        'count': img.shape[0],
        'dtype': img.dtype,
        'nodata': nodata,
        'compress': compress
    }

    if georef_like is not None:

        # read georeference information from the image at path georef_like
        with rio.open(georef_like, 'r') as src:
            assert src.crs != None, "Image on path georef_like is not georeferenced."
            crs = src.crs
            transform = src.transform
        
    if (crs is not None) and (transform is not None):
            
            # expand metadata with georeference information
            meta = {**meta,
                'crs': crs,
                'transform': transform,
                'driver': 'GTiff'
                }
    
    # write image to disk
    if os.path.dirname(path) != '' and not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    with rio.open(path, 'w', **meta) as dst:
        dst.write(img) # rasterio expects images to be formatted as CHW
    
    # add descriptions
    if descriptions is not None:
        with rio.open(path, 'r+') as dst:
            for i, desc in enumerate(descriptions):
                dst.set_band_description(i+1, desc)



def concatenate_rasters(file_list, output_fn):
    """
    Concatenate a list of rasters into a single raster.
    """

    # open all files as rasterio datasets
    src_ds = [rio.open(file) for file in file_list]

    # read metadata
    metas = [src.meta for src in src_ds]
    counts = [src.count for src in src_ds]
    descriptions = [src.descriptions for src in src_ds]

    # compute total number of layers
    out_meta = metas[0]
    out_meta.update(count=sum(counts))
    out_desc = [name for d in descriptions for name in d]

    # read each layer and write it to stack
    id = 1
    
    # to avoid overwriting files, create a temporary file
    temp_fn = f'.temp/{str(uuid.uuid4().hex)}'

    # concatenate datasets and write to disk
    with rio.open(temp_fn, 'w', **out_meta) as dst:
        for src in src_ds:
            for i in range(1, src.count + 1):
                dst.write_band(id, src.read(i))
                id += 1
            src.close()
        dst.descriptions = out_desc

    # rename temp file as output file (NB: this eventually overwrites output_fn,
    # if it already exists)
    os.replace(temp_fn, output_fn)