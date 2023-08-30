from sklearn.cluster import BisectingKMeans
from sklearn.cluster import Birch
import cv2
import os
import numpy as np
from sklearn.cluster import KMeans
from osgeo import gdal
import matplotlib.pyplot as plt


imagesFolder = "dataset/tif//";
ListImages=os.listdir(os.path.join(imagesFolder, "512"))
print('número de imagenes: '+ str(len(ListImages)))

def segementImage(image, format, algorithm):
    path = imagesFolder+"512/"+image
    img_tif = gdal.Open(path)
    bands = 13
    # numpy estructure where I´m going to load the image with all the bands (YSize, XSize, bands)
    # in the imagen download there are 16 bands, but the 14 and 15 are empty  and the 16 is teh mask cloud, in setinel-images there are 13 bands
    # https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2#bands
    img_array = np.zeros((img_tif.RasterYSize, img_tif.RasterXSize, 13))

    # Inserting the values into the numpy array :)
    for band in range(13):
        img_array[:, :, band] = img_tif.GetRasterBand(band + 1).ReadAsArray()

    # Next we need to reshape our array, we need to keep the columns as 13 this time (img.shape[2])
    new_shape = (img_array.shape[0] * img_array.shape[1], img_array.shape[2])

    # based on this shape we can take the X input to the kmeans algorithm
    X = img_array[:, :, :13].reshape(new_shape)

    # whe apply kmenas to input array
    # we choose four clusters
    if(algorithm == "KMeans"):
        alg = KMeans(n_clusters=4, n_init=10)
    elif(algorithm == "BisectingKMeans"):
        alg = BisectingKMeans(n_clusters=4, n_init=10)
    elif(algorithm == "Birch"):
        alg = Birch(n_clusters=4)

    # we fit our data flatten
    alg.fit(X)

    # we obtain the results of the kmeans
    X_segmented = alg.labels_
    X_segmented = X_segmented.reshape(img_array[:, :, 0].shape)
    if(format == "jpg"):
        # visualizamos la salida
        nameImage = image.split('.')[0]+'cluster-kmeans.jpg'
        plt.imsave("dataset/clustering/"+algorithm+"/jpg/" + nameImage, X_segmented)
    elif(format == "tif"):
        if X_segmented.dtype == np.float32:
            arr_type = gdal.GDT_Float32
        else:
            arr_type = gdal.GDT_Int32
        driver = gdal.GetDriverByName("GTiff")
        out_ds = driver.Create( "dataset/clustering/"+algorithm+"/tif/" + image.split('.')[0] + 'cluster-kmeans.tif', 512, 512, 1,  arr_type)
        out_ds.WriteArray(X_segmented)

#Reading the images and inserting them into a list
for i in range(len(ListImages)):
    print(i)
    segementImage(ListImages[i],"jpg","BisectingKMeans")
