import cv2
from osgeo import gdal
import numpy as np

def mergeTifImages(ImageName1,ImageName2,year,version,region):
    img_tif1 = gdal.Open("dataset/clustering/Cultivos_clases/GEE_"+str(year)+"/tiposv"+str(version)+"/"+region+"/"+ImageName1+".tif");
    img_tif2 = gdal.Open("dataset/clustering/Cultivos_clases/GEE_"+str(year)+"/tiposv"+str(version)+"/"+region+"/"+ImageName2+".tif");

    #convierto primera imagen a array
    img_array1 = img_tif1.ReadAsArray(); # (16, 7822, 8192)
    #convierto segunda imagen a array
    img_array2 = img_tif2.ReadAsArray(); # (16, 7822, 1677)

    print("shape1: ", img_array1.shape)
    print("shape2: ", img_array2.shape)

    #concateno las dos imagenes
    img_array = np.concatenate((img_array1, img_array2), axis=2)
    print("shape: ", img_array.shape) #(16, 7822, 9869)

    if img_array.dtype == np.float32:
        arr_type = gdal.GDT_Float32
    else:
        arr_type = gdal.GDT_Int32
    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create("dataset/clustering/Cultivos_clases/GEE_"+str(year)+"/tiposv"+str(version)+"/"+region+"/"+ImageName1.split("-")[0]+".tif", img_array.shape[2], img_array.shape[1], 16, arr_type)
    out_ds.WriteArray(img_array)

mergeTifImages("ZONAMEDIAcollectionComposite-0000000000-0000000000","ZONAMEDIAcollectionComposite-0000000000-0000008192",2019,3,"ZONAMEDIA")