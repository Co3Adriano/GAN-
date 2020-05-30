# gdal lib 
# from asc into tif file and clipping with shapefile into tif

# imports 
import gdal
from osgeo import gdal, osr
import os
import shutil
import numpy as np
import shapefile

#root directory

root_dir = 'E:/BachelorArbeit/KI_Regen/RadoLan_ASC_SET/Starkregen/RW_20150102-2350.asc'


# get Projection from tif-Project-File #works
ds=gdal.Open(r'C:/Users/Adria/Desktop/ProjectRessource.tif')
prj=ds.GetProjection()
print(prj)

srs=osr.SpatialReference(wkt=prj)
if srs.IsProjected:
    print (srs.GetAttrValue('projcs'))
    print (srs.GetAttrValue('geogcs'))



# Set file 
output_file = "out.tif"

# Create gtif #works
driver = gdal.GetDriverByName("GTiff")
ds_in = gdal.Open('E:/BachelorArbeit/KI_Regen/RadoLan_ASC_SET/Starkregen/RW_20190127-0550.asc')
dst_ds = driver.CreateCopy(output_file,ds_in )


# set the reference info # works
dst_ds.SetProjection( srs.ExportToWkt() )


#clip raster with shape file into asc file #works
sf = "E:/BachelorArbeit/KI_Regen/ShapeFile/EsriShapeAachen.shp"




#gdalwarp into cropped.tif # works
filepath = 'E:/BachelorArbeit/KI_Regen/RadoLan_ASC_SET/temp/croppped.tif'
result = gdal.Warp(destNameOrDestDS=filepath ,srcDSOrSrcDSTab = dst_ds,  dstSRS=srs, cutlineDSName=sf)






# iteration through files in foolder
