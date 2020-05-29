#####RadoLan to GeoTIFF
import wradlib as wrl
import numpy as np
import warnings
warnings.filterwarnings('ignore')

wdir = 'C:/Users/Adria/Desktop/KI_Regen' + '/radolan/grid/'
filename = ('C:/Users/Adria/Desktop/KI_Regen/raa01-rw_10000-2004211050-dwd---bin.gz')
data_raw, meta = wrl.io.read_radolan_composite(filename)

# This is the RADOLAN projection
proj_osr = wrl.georef.create_osr("dwd-radolan")

# Get projected RADOLAN coordinates for corner definition
xy_raw = wrl.georef.get_radolan_grid(900, 900)

data, xy = wrl.georef.set_raster_origin(data_raw, xy_raw, 'upper')

# create 3 bands
data = np.stack((data, data+100, data+1000))
ds = wrl.georef.create_raster_dataset(data, xy, projection=proj_osr)
wrl.io.write_raster_dataset(wdir + "geotiff.tif", ds, 'GTiff')

# Read from GeoTIFF
ds1 = wrl.io.open_raster(wdir + "geotiff.tif")
data1, xy1, proj1 = wrl.georef.extract_raster_dataset(ds1, nodata=-9999.)
np.testing.assert_array_equal(data1, data)
np.testing.assert_array_equal(xy1, xy)