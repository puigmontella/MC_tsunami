from pyproj import Transformer
import rasterio
import geopandas as gpd
import numpy as np
from rasterio.features import rasterize

BATHY_TIF = "../bati_32m_merged_EPSG32628.tif"
SAND_GPKG = "../EMODnet_Seabed_substrate_Multiscale/layerVector_folk7/Folk_7cl_2.gpkg"
ROCK_GPKG = "../EMODnet_Seabed_substrate_Multiscale/layerVector_folk7/Folk_7cl_5.gpkg"

raster_path_ref=BATHY_TIF
with rasterio.open(raster_path_ref) as src:
    shape = (src.height, src.width)
    transform = src.transform
    rcrs = src.crs
    rb = src.bounds
    prof = src.profile.copy()


    
raster_crs="EPSG:32628"

transformer = Transformer.from_crs(raster_crs, "EPSG:4326", always_xy=True)
xs = [rb.left, rb.right, rb.left, rb.right]
ys = [rb.bottom, rb.bottom, rb.top, rb.top]
lon, lat = transformer.transform(xs, ys)
bbox4326 = (min(lon), min(lat), max(lon), max(lat))

sand_gdf = gpd.read_file(SAND_GPKG, bbox=bbox4326).to_crs(raster_crs)
rock_gdf = gpd.read_file(ROCK_GPKG, bbox=bbox4326).to_crs(raster_crs)


# --- rasterize to masks (1 inside polygon, 0 outside) ---
sand_mask = rasterize(
    ((geom, 1) for geom in sand_gdf.geometry if geom is not None and not geom.is_empty),
    out_shape=shape,
    transform=transform,
    fill=0,
    dtype="uint8",
    all_touched=True
)

rock_mask = rasterize(
    ((geom, 1) for geom in rock_gdf.geometry if geom is not None and not geom.is_empty),
    out_shape=shape,
    transform=transform,
    fill=0,
    dtype="uint8",
    all_touched=True
)

# --- write to GeoTIFFs (aligned to bathy) ---
out_prof = prof.copy()
out_prof.update(dtype="uint8", count=1, nodata=0, compress="deflate")

with rasterio.open("emodnet_sand_mask_epsg32628.tif", "w", **out_prof) as dst:
    dst.write(sand_mask, 1)

with rasterio.open("emodnet_rock_mask_epsg32628.tif", "w", **out_prof) as dst:
    dst.write(rock_mask, 1)