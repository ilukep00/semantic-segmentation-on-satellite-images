import cv2
import os
import numpy as np
from sklearn.cluster import KMeans
from osgeo import gdal
import matplotlib.pyplot as plt


imagesFolder = "dataset/tif//";
ListImages=os.listdir(os.path.join(imagesFolder, "512"))
imagenesEstacion = []
estacion = "winter"
n_clusters = 5
for imageName in ListImages:
    if(estacion in imageName):
        imagenesEstacion.append(imageName)
print('número de imagenes: '+ str(len(imagenesEstacion)))


def preprare_image(image):
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
    return img_array[:, :, :13].reshape(new_shape)

print("new shape: "+str((262144*len(imagenesEstacion),13)))

X = np.zeros((262144*len(imagenesEstacion),13))
#Reading the images and inserting them into a list
for i in range(len(imagenesEstacion)):
    X[i*262144:(i+1)*262144,:13] = preprare_image(imagenesEstacion[i])

print('training data:'+str(X))

#aplico kmeans a X
#whe apply kmenas to input array
#we choose four clusters
kmeans = KMeans(n_clusters=n_clusters, n_init = 10)

#we fit our data flatten
kmeans.fit(X)

X_segmented = kmeans.labels_

print(str(X_segmented.shape))

for i in range(len(imagenesEstacion)):
    X_out = X_segmented[i * 262144: (i+1)*262144].reshape((512,512))
    plt.imsave("dataset/clustering/KMeans/"+str(n_clusters)+"_clusters/jpg/"+imagenesEstacion[i].split('.')[0]+".jpg", X_out)




