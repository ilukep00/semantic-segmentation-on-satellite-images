import matplotlib.pyplot as plt
from osgeo import gdal
import numpy as np
import random
import cv2
path = "dataset/clustering/Cultivos_clases/GEE/"
def splitImages(folder, imgName, ImgCultivosName, imageImproductivoName, ImageForestalName, num):
    img_tif = gdal.Open(path+"/"+folder+"/"+imgName+".tif")
    cultivos_tif = gdal.Open(path + "/" + folder + "/" + ImgCultivosName + ".tif")
    improductivo_tif = gdal.Open(path + "/" + folder + "/" + imageImproductivoName + ".tif")
    forestal_tif = gdal.Open(path + "/" + folder + "/" + ImageForestalName + ".tif")

    image_array = img_tif.ReadAsArray()
    cultivos_array = cultivos_tif.ReadAsArray()
    improductivo_array = improductivo_tif.ReadAsArray()
    forestal_array =forestal_tif.ReadAsArray()

    max_X = image_array.shape[1] - 250
    max_Y = image_array.shape[2] - 250

    for i in range(num):
        #random points
        randx = random.randint(250, max_X)
        randy = random.randint(250, max_Y)

        image_to_write = np.zeros((500, 500, 3))

        #imagen Real
        image_to_write[:, :, 0] = image_array[3, randx - 250:randx + 250, randy - 250:randy + 250]
        image_to_write[:, :, 1] = image_array[2, randx - 250:randx + 250, randy - 250:randy + 250]
        image_to_write[:, :, 2] = image_array[1, randx - 250:randx + 250, randy - 250:randy + 250]

        plt.imsave(path+"/"+folder+"/dataset/images/"+folder+str(i)+".jpg",image_to_write/image_to_write.max())

        #Cultivos images
        img_cluster_to_write = np.zeros((500, 500))
        img_cluster = cultivos_array[randx - 250:randx + 250, randy - 250:randy + 250]
        img_cluster_to_write[img_cluster > 0] = 255

        cv2.imwrite(path + "/" + folder + "/dataset/semantic/Cultivos/cultivos_" + folder + str(i) + ".png",
                   img_cluster_to_write)

        # improductive images
        img_cluster_to_write = np.zeros((500, 500))
        img_cluster = improductivo_array[randx - 250:randx + 250, randy - 250:randy + 250]
        img_cluster_to_write[img_cluster > 0] = 255

        cv2.imwrite(path + "/" + folder + "/dataset/semantic/Improductivo/improductivo_" + folder + str(i) + ".png",
                    img_cluster_to_write)

        # forestal images
        img_cluster_to_write = np.zeros((500, 500))
        img_cluster = forestal_array[randx - 250:randx + 250, randy - 250:randy + 250]
        img_cluster_to_write[img_cluster > 0] = 255

        cv2.imwrite(path + "/" + folder + "/dataset/semantic/Forestal/forestal_" + folder + str(i) + ".png",
                    img_cluster_to_write)

def splitImages_testing(folder, imgName, num):
    img_tif = gdal.Open(path + "/" + folder + "/" + imgName + ".tif")

    image_array = img_tif.ReadAsArray()

    max_X = image_array.shape[1] - 250
    max_Y = image_array.shape[2] - 250

    for i in range(num):
        randx = random.randint(250, max_X)
        randy = random.randint(250, max_Y)

        image_to_write = np.zeros((500, 500, 3))

        # imagen Real
        image_to_write[:, :, 0] = image_array[3, randx - 250:randx + 250, randy - 250:randy + 250]
        image_to_write[:, :, 1] = image_array[2, randx - 250:randx + 250, randy - 250:randy + 250]
        image_to_write[:, :, 2] = image_array[1, randx - 250:randx + 250, randy - 250:randy + 250]

        plt.imsave("dataset/clustering/Cultivos_clases/Validation/Images/" + folder + str(i) + ".jpg",
                   image_to_write / image_to_write.max())

#splitImages("BAZTAN","collectionCompositeBaztan","BaztanCultivos2","BaztanImproductivo2","BaztanForestal2",150)
#splitImages("CUENCADEPAMPLONA","collectionCompositeCuencaPamplona","CuencaPamplonaCultivos2","CuencaPamplonaImproductivo2","CuencaPamplonaForestal2",150)
#splitImages("PIRINEOS","collectionCompositePirineos","PirineosCultivos2","PirineosImproductivo2","PirineosForestal2",150)
#splitImages("RIBERA","collectionCompositerRibera","RiberaCultivos2","RiberaImproductivo2","RiberaForestal2",150)
#splitImages("ULTZAMA","collectionCompositeUltzama","UltzamaCultivos2","UltzamaImproductivo2","UltzamaForestal2",150)
#splitImages("ZONAMEDIA","collectionCompositeZonaMedia","ZonaMediaCultivos2","ZonaMediaImproductivo2","ZonaMediaForestal2",150)
splitImages_testing("LARIOJA","collectionCompositeLaRioja1",150)
