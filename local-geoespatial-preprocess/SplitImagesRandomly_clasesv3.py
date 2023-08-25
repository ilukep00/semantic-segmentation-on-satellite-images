import matplotlib.pyplot as plt
from osgeo import gdal
import numpy as np
import random
import cv2
import os

path2019 = "dataset/clustering/Cultivos_clases/GEE_2019/tiposv3/"
path2021 = "dataset/clustering/Cultivos_clases/GEE_2021/tiposv3/"
def splitImages(folder, imgName, ImgAguaName, ImgCultivosName, ImgForestalArboladoName, ImgForestalNoArboladoName, ImgImproductivoName, num, stage,path, formatImg = 'jpg'):
    img_tif = gdal.Open(path+"/"+folder+"/"+imgName+".tif")
    agua_tif = gdal.Open(path + "/" + folder + "/" + ImgAguaName + ".tif")
    cultivos_tif = gdal.Open(path + "/" + folder + "/" + ImgCultivosName + ".tif")
    forestalArbolado_tif = gdal.Open(path + "/" + folder + "/" + ImgForestalArboladoName + ".tif")
    forestalNoArbolado_tif = gdal.Open(path + "/" + folder + "/" + ImgForestalNoArboladoName + ".tif")
    improductivo_tif = gdal.Open(path + "/" + folder + "/" + ImgImproductivoName + ".tif")



    image_array = img_tif.ReadAsArray()
    print("image shape :",image_array.shape)
    agua_array = agua_tif.ReadAsArray()
    print("agua shape :",agua_array.shape)
    cultivos_array = cultivos_tif.ReadAsArray()
    print("cultivos_array shape :", cultivos_array.shape)
    forestalArbolado_array = forestalArbolado_tif.ReadAsArray()
    forestalNoArbolado_array = forestalNoArbolado_tif.ReadAsArray()
    improductivo_array = improductivo_tif.ReadAsArray()

    max_X = image_array.shape[1] - 250
    max_Y = image_array.shape[2] - 250
    i = 0
    while i < num:
        #random points
        randx = random.randint(250, max_X)
        randy = random.randint(250, max_Y)

        image_to_write = np.zeros((500, 500, 3))

        # Agua images
        img_cluster_to_write = np.zeros((500, 500))
        img_cluster = agua_array[randx - 250:randx + 250, randy - 250:randy + 250]
        if(stage == "Training" and np.sum([img_cluster>0]) == 0):
            continue
        img_cluster_to_write[img_cluster > 0] = 255

        cv2.imwrite(path + "/dataset/" + formatImg + "/" + stage + "/Semantic/Agua/agua_" + folder + str(i) + ".png", img_cluster_to_write)

        #imagen Real
        if(formatImg == "jpg"):
            image_to_write[:, :, 0] = image_array[3, randx - 250:randx + 250, randy - 250:randy + 250]
            image_to_write[:, :, 1] = image_array[2, randx - 250:randx + 250, randy - 250:randy + 250]
            image_to_write[:, :, 2] = image_array[1, randx - 250:randx + 250, randy - 250:randy + 250]

            plt.imsave(path+"/dataset/"+formatImg+"/"+stage+"/Images/"+folder+str(i)+".jpg",image_to_write/image_to_write.max())
        elif(formatImg == "tif"):
            if image_to_write.dtype == np.float32:
                arr_type = gdal.GDT_Float32
            else:
                arr_type = gdal.GDT_Int32
            driver = gdal.GetDriverByName("GTiff")
            out_ds = driver.Create(path+"/dataset/"+formatImg+"/"+stage+"/Images/"+folder+str(i)+".tif", 500, 500, 16,arr_type)
            out_ds.WriteArray(image_array[:,randx - 250:randx + 250, randy - 250:randy + 250])

        #Cultivos images
        img_cluster_to_write = np.zeros((500, 500))
        img_cluster = cultivos_array[randx - 250:randx + 250, randy - 250:randy + 250]
        img_cluster_to_write[img_cluster > 0] = 255

        cv2.imwrite(path + "/dataset/" + formatImg + "/" + stage+"/Semantic/Cultivos/cultivos_" + folder + str(i) + ".png",
                    img_cluster_to_write)

        # Forestal Arbolado images
        img_cluster_to_write = np.zeros((500, 500))
        img_cluster = forestalArbolado_array[randx - 250:randx + 250, randy - 250:randy + 250]
        img_cluster_to_write[img_cluster > 0] = 255

        cv2.imwrite(path + "/dataset/" + formatImg + "/" + stage+"/Semantic/ForestalArbolado/forestalArbolado_" + folder + str(i) + ".png",
                    img_cluster_to_write)

        # Forestal no Arbolado images
        img_cluster_to_write = np.zeros((500, 500))
        img_cluster = forestalNoArbolado_array[randx - 250:randx + 250, randy - 250:randy + 250]
        img_cluster_to_write[img_cluster > 0] = 255

        cv2.imwrite(path + "/dataset/"+ formatImg + "/" +stage+"/Semantic/ForestalNoArbolado/forestalNoArbolado_" + folder + str(i) + ".png",
                    img_cluster_to_write)

        # improductive images
        img_cluster_to_write = np.zeros((500, 500))
        img_cluster = improductivo_array[randx - 250:randx + 250, randy - 250:randy + 250]
        img_cluster_to_write[img_cluster > 0] = 255

        cv2.imwrite(path + "/dataset/"+ formatImg + "/"+stage+"/Semantic/Improductivo/improductivo_" + folder + str(i) + ".png",
                    img_cluster_to_write)
        i += 1
        print(i)


def hasWater(aguaSplited_array):
    return np.sum([aguaSplited_array>0]) > 0

#splitImages("ZONAMEDIA","ZONAMEDIAcollectionComposite","ZONAMEDIAcultivosHerbaceos","ZONAMEDIACultivosLenosos","ZONAMEDIAforestalArbolado","ZONAMEDIAforestalNoArbolado","ZONAMEDIAimproductivo",100,"Testing",path2021)
def splitImages2021(num):
    ListFolders2021 = os.listdir(os.path.join(path2021, ""))
    ListFolders2021.remove("dataset")
    for folder in ListFolders2021:
        splitImages(folder, folder+"collectionComposite", folder+"agua",
                    folder+"Cultivos", folder+"forestalArbolado", folder+"forestalNoArbolado",
                    folder+"improductivo", num, "Testing", path2021,'tif')

def splitPirineosImages2021(num):
    folder = "PIRINEOS"
    splitImages(folder, folder + "collectionComposite", folder + "agua",
                folder + "Cultivos", folder + "forestalArbolado", folder + "forestalNoArbolado",
                folder + "improductivo", num, "Testing", path2021)
def splitAllImages(stage,num,path,formatImg):
    ListFolders = []
    if stage == "Training":
        ListFolders = ["BAZTAN","CUENCADEPAMPLONA","PIRINEOS","RIBERA"]
    elif stage == "Validation":
        ListFolders = ["ULTZAMA","ZONAMEDIA"]
    for folder in ListFolders:
        splitImages(folder, folder+"collectionComposite", folder+"agua",
                    folder+"Cultivos", folder+"forestalArbolado", folder+"forestalNoArbolado",
                    folder+"improductivo", num, stage, path, formatImg)

#splitAllImages("Validation",300,path2019,'tif')

splitImages2021(100)
