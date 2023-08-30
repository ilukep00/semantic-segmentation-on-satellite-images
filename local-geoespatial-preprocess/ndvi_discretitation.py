import os
import numpy as np
from osgeo import gdal
import matplotlib.pyplot as plt

#obtenemos direccion de las imagenes.

imagesFolder = "dataset/tif//"
listImages = os.listdir(os.path.join(imagesFolder, "512"))
print("Número de imagenes: " + str(len(listImages)))


def nvdi_discretize_image(imagePath,format):
    if(len(imagePath.split('.')) == 2):
        #reading image
        img_tif = gdal.Open(imagesFolder+"512/"+imagePath)
        #getting red band(4):
        red_band = img_tif.GetRasterBand(4).ReadAsArray()
        #getting nir band(5):
        nir_band = img_tif.GetRasterBand(5).ReadAsArray()

        ndvi_image = (nir_band-red_band)/(nir_band+red_band)

        nvdi_image_segmented = np.zeros(ndvi_image.shape)

        nvdi_image_segmented[ndvi_image <= 0.1] = 0; #No vegetation
        nvdi_image_segmented[(ndvi_image > 0.1) & (ndvi_image <= 0.2)] = 1; #open .... (https://eos.com/es/blog/ndvi-preguntas-frecuentes/)
        nvdi_image_segmented[(ndvi_image > 0.2) & (ndvi_image <= 0.4)] = 2; #sparse vegetation
        nvdi_image_segmented[(ndvi_image > 0.4) & (ndvi_image <= 0.6)] = 3; #Moderate vegation
        nvdi_image_segmented[(ndvi_image > 0.6)] = 4; #Dense Vegetation
        if (format == "jpg"):
            nameImage = imagePath.split('.')[0] + 'cluster-nvdi.jpg'
            plt.imsave("dataset/clustering/nvdi/jpg/" + nameImage, nvdi_image_segmented)
        elif(format == "png"):
            nameImage = imagePath.split('.')[0] + 'cluster-nvdi.png'
            plt.imsave("dataset/clustering/nvdi/png/" + nameImage, nvdi_image_segmented)
        elif (format == "tif"):
            if nvdi_image_segmented.dtype == np.float32:
                arr_type = gdal.GDT_Float32
            else:
                arr_type = gdal.GDT_Int32
            driver = gdal.GetDriverByName("GTiff")
            out_ds = driver.Create( "dataset/clustering/nvdi/tif/" + imagePath.split('.')[0] + 'cluster-nvdi.tif', 512, 512, 1,  arr_type)
            out_ds.WriteArray(nvdi_image_segmented)


for imagePath in listImages:
    nvdi_discretize_image(imagePath,'png')




