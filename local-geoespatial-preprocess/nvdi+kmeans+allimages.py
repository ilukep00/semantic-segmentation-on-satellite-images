import os
import numpy as np
from osgeo import gdal
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
#obtenemos direccion de las imagenes.

imagesFolder = "dataset/tif//"
listImages = os.listdir(os.path.join(imagesFolder, "512"))

def nvdi_indexPath(imagePath):
    if (len(imagePath.split('.')) == 2):
        # reading image
        img_tif = gdal.Open(imagesFolder + "512/" + imagePath)
        # getting red band(4):
        red_band = img_tif.GetRasterBand(4).ReadAsArray()
        # getting nir band(5):
        nir_band = img_tif.GetRasterBand(5).ReadAsArray()

        ndvi_image = (nir_band - red_band) / (nir_band + red_band)
        return ndvi_image

arrayofpixels = np.zeros((512*512*len(listImages),1))
i = 0
for imagePath in listImages:
    if len(imagePath.split('.')) == 2:
        arrayofpixels[i*512*512:(i+1)*(512*512)] =nvdi_indexPath(imagePath).reshape(-1,1)
        i += 1

kmeans = KMeans(n_clusters=5, n_init=10)
kmeans.fit(arrayofpixels)

nvdi_image_segmented = kmeans.labels_

def save_image(imageName, labelsPixel,format):
    img = labelsPixel.reshape(512,512)
    if format == 'png':
        nameImage = imageName.split('.')[0] + 'cluster-nvdi.png'
        plt.imsave("dataset/clustering/nvdi+kmeans/seasons/" + nameImage, img)

i = 0
for imagePath in listImages:
    if len(imagePath.split('.')) == 2:
        save_image(imagePath,nvdi_image_segmented[i * 512 * 512:(i + 1) * (512 * 512)],'png')
        i +=1

