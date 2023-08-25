import matplotlib.pyplot as plt
from osgeo import gdal
import numpy as np
import random
import cv2
import os

path2019 = "dataset/clustering/Cultivos_clases/GEE_2019/"
path2021 = "dataset/clustering/Cultivos_clases/GEE_2021/"
def splitImages(folder, imgName, ImgCultivosHerbaceosName, ImgCultivosLenososName, ImgForestalArboladoName, ImgForestalNoArboladoName, ImgImproductivoName, num, stage,path):
    img_tif = gdal.Open(path+"/"+folder+"/"+imgName+".tif")
    cultivosHerbaceos_tif = gdal.Open(path + "/" + folder + "/" + ImgCultivosHerbaceosName + ".tif")
    cultivosLenosos_tif = gdal.Open(path + "/" + folder + "/" + ImgCultivosLenososName + ".tif")
    forestalArbolado_tif = gdal.Open(path + "/" + folder + "/" + ImgForestalArboladoName + ".tif")
    forestalNoArbolado_tif = gdal.Open(path + "/" + folder + "/" + ImgForestalNoArboladoName + ".tif")
    improductivo_tif = gdal.Open(path + "/" + folder + "/" + ImgImproductivoName + ".tif")



    image_array = img_tif.ReadAsArray()
    cultivosHerbaceos_array = cultivosHerbaceos_tif.ReadAsArray()
    cultivosLenosos_array = cultivosLenosos_tif.ReadAsArray()
    forestalArbolado_array = forestalArbolado_tif.ReadAsArray()
    forestalNoArbolado_array = forestalNoArbolado_tif.ReadAsArray()
    improductivo_array = improductivo_tif.ReadAsArray()

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

        plt.imsave(path+"/dataset/"+stage+"/Images/"+folder+str(i)+".jpg",image_to_write/image_to_write.max())

        #Cultivos Herbaceos images
        img_cluster_to_write = np.zeros((500, 500))
        img_cluster = cultivosHerbaceos_array[randx - 250:randx + 250, randy - 250:randy + 250]
        img_cluster_to_write[img_cluster > 0] = 255

        cv2.imwrite(path + "/dataset/"+stage+"/Semantic/CultivosHerbaceos/cultivosHerbaceos_" + folder + str(i) + ".png",
                   img_cluster_to_write)

        #Cultivos Leñosos images
        img_cluster_to_write = np.zeros((500, 500))
        img_cluster = cultivosLenosos_array[randx - 250:randx + 250, randy - 250:randy + 250]
        img_cluster_to_write[img_cluster > 0] = 255

        cv2.imwrite(path + "/dataset/"+stage+"/Semantic/CultivosLenosos/cultivosLenosos_" + folder + str(i) + ".png",
                    img_cluster_to_write)

        # Forestal Arbolado images
        img_cluster_to_write = np.zeros((500, 500))
        img_cluster = forestalArbolado_array[randx - 250:randx + 250, randy - 250:randy + 250]
        img_cluster_to_write[img_cluster > 0] = 255

        cv2.imwrite(path + "/dataset/"+stage+"/Semantic/ForestalArbolado/forestalArbolado_" + folder + str(i) + ".png",
                    img_cluster_to_write)

        # Forestal no Arbolado images
        img_cluster_to_write = np.zeros((500, 500))
        img_cluster = forestalNoArbolado_array[randx - 250:randx + 250, randy - 250:randy + 250]
        img_cluster_to_write[img_cluster > 0] = 255

        cv2.imwrite(path + "/dataset/"+stage+"/Semantic/ForestalNoArbolado/forestalNoArbolado_" + folder + str(i) + ".png",
                    img_cluster_to_write)

        # improductive images
        img_cluster_to_write = np.zeros((500, 500))
        img_cluster = improductivo_array[randx - 250:randx + 250, randy - 250:randy + 250]
        img_cluster_to_write[img_cluster > 0] = 255

        cv2.imwrite(path + "/dataset/"+stage+"/Semantic/Improductivo/improductivo_" + folder + str(i) + ".png",
                    img_cluster_to_write)

def splitImages_testing(folder, imgName, num,path):
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

        plt.imsave(path + "/dataset/Testing/" + folder + str(i) + ".jpg",
                   image_to_write / image_to_write.max())

#splitImages("ZONAMEDIA","ZONAMEDIAcollectionComposite","ZONAMEDIAcultivosHerbaceos","ZONAMEDIACultivosLenosos","ZONAMEDIAforestalArbolado","ZONAMEDIAforestalNoArbolado","ZONAMEDIAimproductivo",100,"Testing",path2021)
def splitImages2021(stage,num,path):
    ListFolders2021 = os.listdir(os.path.join(path, ""))
    ListFolders2021.remove("dataset")
    for folder in ListFolders2021:
        splitImages(folder, folder+"collectionComposite", folder+"cultivosHerbaceos",
                    folder+"CultivosLenosos", folder+"forestalArbolado", folder+"forestalNoArbolado",
                    folder+"improductivo", num, stage, path)

splitImages2021("Testing",100,path2021)
