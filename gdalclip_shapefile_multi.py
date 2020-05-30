#script for clipping with shapefile.shp multiple ascii files into tif into asci again

# using gdallib
# imports 
import gdal
from osgeo import gdal, osr
import os
import shutil
import numpy as np
import shapefile


# get Projection from tif-Project-File #works
ds=gdal.Open(r'C:/Users/Adria/Desktop/ProjectRessource.tif')
prj=ds.GetProjection()
print(prj)

srs=osr.SpatialReference(wkt=prj)
if srs.IsProjected:
    print (srs.GetAttrValue('projcs'))
    print (srs.GetAttrValue('geogcs'))
           


# root dir of origin asc file          
root_dir = 'E:/BachelorArbeit//KI_Regen//RadoLan_ASC_SET//Starkregentest'  # root_directory hier befinden sich die zu sortierenden Datein
          
           
for subdir, dirs, files in os.walk(root_dir):
    for filename in files:
        filepath = subdir + os.sep + filename

        if filepath.endswith(".asc") :
            print (filepath)
            with open(os.path.join(subdir, filename)) as f:
             print(filepath)
             # Set file 
             output_file =  "temp.tif"

             # Create gtif #works
             driver = gdal.GetDriverByName("GTiff")
             ds_in = gdal.Open(filepath)
             dst_ds = driver.CreateCopy(output_file,ds_in )


             # set the reference info # works
             dst_ds.SetProjection( srs.ExportToWkt() )


             #clip raster with shape file into asc file #works
             sf = "E:/BachelorArbeit/KI_Regen/ShapeFile/EsriShapeAachen.shp"

             #gdalwarp into cropped.tif # works
             savefilepath = 'E:/BachelorArbeit/KI_Regen/RadoLan_ASC_SET/cropped/' + filename 
             result = gdal.Warp(destNameOrDestDS=savefilepath ,srcDSOrSrcDSTab = dst_ds,  dstSRS=srs, cutlineDSName=sf)

