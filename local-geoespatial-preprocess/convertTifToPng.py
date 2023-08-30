import cv2
from osgeo import gdal


import numpy as np
import matplotlib.pyplot as plt

def convertTifToPng(nameImage):
    img_tif = gdal.Open('dataset/clustering/Cultivos_clases/'+nameImage+'.tif')

    print("***********INFORMATION WINTER-2019***********")
    print("Projection: ", img_tif.GetProjection())  # get projection
    print("Columns:", img_tif.RasterXSize)  # number of columns
    print("Rows:", img_tif.RasterYSize)  # number of rows
    print("Band count:", img_tif.RasterCount)  # number of bands
    print("GeoTransform", img_tif.GetGeoTransform())

    img_array = img_tif.GetRasterBand(1).ReadAsArray()

    img_cluster = np.zeros(img_array.shape)
    print(img_cluster.shape)

    img_cluster[img_array > 0] = 255

    #la guardamos en formato png
    cv2.imwrite("dataset/clustering/Cultivos_clases/"+nameImage+".png", img_cluster)

convertTifToPng('NavarreCultivos')
convertTifToPng('NavarreForestal')
convertTifToPng('NavarreImproductivo')

