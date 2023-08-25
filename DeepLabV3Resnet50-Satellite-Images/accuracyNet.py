import torchvision.models.segmentation
import torch
import torchvision.transforms as tf
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from osgeo import gdal

#VERSION 2 DE CLASES
modelPath_v2 = "modelos_v2//" # El path del  modelo entrenado (Luego habra que hacer un bucle por cada modelo entrenado)
imagesFolder_v2 = "dataset_v2//"

ListImagesValidation_v2 = os.listdir(os.path.join(imagesFolder_v2, "Validation/Images"))
ListImagesTesting_v2 = os.listdir(os.path.join(imagesFolder_v2, "Testing/NAVARRA_2021/Images"))
ListModelsTrained_v2 = os.listdir(os.path.join(modelPath_v2))

#VERSION 3 DE CLASES
modelPath_v3_deeplab = "modelos_v3_deeplab//"
imagesFolder_v3 = "dataset_v3//"

ListImagesValidation_v3 = os.listdir(os.path.join(imagesFolder_v3,"Validation/Images"))
ListModelsTrained_v3_deeplab = os.listdir(os.path.join(modelPath_v3_deeplab))

#MODELO ENTRENADO RED TIPO FCN
modelPath_v3_fcn = "modelos_v3_fcn//"
ListModelsTrained_v3_fcn = os.listdir(os.path.join(modelPath_v3_fcn))

#MODELO ENTRENADO CON DEEPLAB COLOR INFRARROJO
modelPath_v3_deeplab_infrarrojo = "modelos_v3_deeplab_COLOR_INFRARROJO//"
ListModelsTrained_v3_deeplab_infrarrojo = os.listdir(os.path.join(modelPath_v3_deeplab_infrarrojo))

#MODELO ENTRENADO CON DEEPLAB AGRICULTURA
modelPath_v3_deeplab_agricultura = "modelos_v3_deeplab_agricultura//"
ListModelsTrained_v3_deeplab_agricultura = os.listdir(os.path.join(modelPath_v3_deeplab_agricultura))

#MODELO ENTRENADO CON DEEPLAB VEGETACION
modelPath_v3_deeplab_vegetacion = "modelos_v3_deeplab_vegetacion//"
ListModelsTrained_v3_deeplab_vegetacion= os.listdir(os.path.join(modelPath_v3_deeplab_vegetacion))

#MODELO ENTRENADO CON DEEPLAB ZONASURBANAS
modelPath_v3_deeplab_zonasurbanas = "modelos_v3_deeplab_zonasurbanas//"
ListModelsTrained_v3_deeplab_zonasurbanas= os.listdir(os.path.join(modelPath_v3_deeplab_zonasurbanas))

#MODELO ENTRENADO CON DEEPLAB VEGETACION SALUDABLE
modelPath_v3_deeplab_vegetacionsaludable = "modelos_v3_deeplab_vegetacionsaludable//"
ListModelsTrained_v3_deeplab_vegetacionsaludable= os.listdir(os.path.join(modelPath_v3_deeplab_vegetacionsaludable))


#IMAGENES TIF
imagesFolder_v3_tif = "dataset_v3_tif//"
ListImagesValidation_v3_tif = os.listdir(os.path.join(imagesFolder_v3_tif,"Validation/Images"))

height = width = 500 # Tamaño de la imagen a segmentar

#Definimos el conjunto de transformaciones que se van a utlizar sobre las imagenes sando el modulo de tranformacion de TorchVision definido en train
transformImg = tf.Compose([tf.ToPILImage(mode='RGB'),tf.ToTensor()])

# La primera parte identifica si el computador tiene GPU o CPU. Si tiene Cuda GPU el entrenamiento se realiza en el GPU:
# Para cualquier dataset practico, entrenar usando la CPU es extremadamente lento.
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def preparingNet_deeplab(modelPath,numClusters):
    # A continuación, cargamos la segmentación semántica de deep lab net:
    Net = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=False)

    # antes con deeplab era 256 en vez de 512 con fcn
    # Queremos reemplazar la ultima capa convolucional con 21 neuronas de salida por una que tenga 5 neuronas de salida:
    Net.classifier[4] = torch.nn.Conv2d(256, numClusters, kernel_size=(1, 1), stride=(1, 1))

    Net = Net.to(device)  # Set net to GPU or CPU

    # Cargamos el modelo entrenado y cargado anteriormente en el archivo modelPath
    Net.load_state_dict(torch.load(modelPath))

    # Convertimos la red del modo entrenamiento al modo evaluacion.
    # Esto indica que no se calcularán estadísticas de normalización de lotes.
    Net.eval()
    return Net
def preparingNet_fcn(modelPath,numClusters):
    # A continuación, cargamos la segmentación semántica de deep lab net:
    #Net = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=False)
    Net = torchvision.models.segmentation.fcn_resnet50(pretrained=False)

    # antes con deeplab era 256 en vez de 512 con fcn
    # Queremos reemplazar la ultima capa convolucional con 21 neuronas de salida por una que tenga 5 neuronas de salida:
    Net.classifier[4] = torch.nn.Conv2d(512, numClusters, kernel_size=(1, 1), stride=(1, 1))

    Net = Net.to(device)  # Set net to GPU or CPU


    # Cargamos el modelo entrenado y cargado anteriormente en el archivo modelPath
    Net.load_state_dict(torch.load(modelPath))

    # Convertimos la red del modo entrenamiento al modo evaluacion.
    # Esto indica que no se calcularán estadísticas de normalización de lotes.
    Net.eval()
    return Net
def PredNetOfImg(imagenName,Net,path,bandsToUse = []):
    if imagenName.split(".")[1] == 'tif':
        img_tif = gdal.Open(path+"/Images/"+imagenName)
        img_array_1Band = img_tif.GetRasterBand(bandsToUse[0]).ReadAsArray()
        img_array_2Band = img_tif.GetRasterBand(bandsToUse[1]).ReadAsArray()
        img_array_3Band = img_tif.GetRasterBand(bandsToUse[2]).ReadAsArray()

        img_array = np.zeros((img_array_1Band.shape[0],img_array_1Band.shape[1],3))

        img_array[:, :, 0] = img_array_1Band
        img_array[:, :, 1] = img_array_2Band
        img_array[:, :, 2] = img_array_3Band

    else:
        img_array = cv2.imread(path+"/Images/"+imagenName)
    height_orgin, width_orgin, d = img_array.shape  # Obtenemos el tamaño original de la imagen
    img_array = transformImg(img_array)  # Transformamos la imagen al formato definido anteriormente
    # Convertimos los datos en variables  de gradiente que pueden ser usados por la red
    img_array = torch.autograd.Variable(img_array, requires_grad=False).to(device).unsqueeze(0)
    # Para ejecutar la red  sin recoger gradientes.
    # Los gradientes solo son relevantes para el entrenamiento y recopilarlos requiere muchos recursos
    with torch.no_grad():
        Prd = Net(img_array)['out']

    # Acuerdate que la salida sera mapeada con 5 canales por imagen con cada canal representando la probabilidad de cada clase
    # Para encontrar la clase perteneciente de cada pixel, cogemos el canal con el mayor valor de los tres con la funcion argmax
    seg = torch.argmax(Prd[0], 0).cpu().detach().numpy()
    return seg
def realLabelImg_v2(imagenName,path):
    cultivosHerbaceos = cv2.imread(path+"/Semantic/CultivosHerbaceos/cultivosHerbaceos_" + imagenName.split('.')[0] + '.png', 0)
    cultivosLenosos = cv2.imread(path+"/Semantic/CultivosLenosos/cultivosLenosos_" + imagenName.split('.')[0] + '.png', 0)
    forestalArbolado = cv2.imread(path+"/Semantic/ForestalArbolado/forestalArbolado_" + imagenName.split('.')[0] + '.png', 0)
    forestalNoArbolado = cv2.imread(path+"/Semantic/ForestalNoArbolado/forestalNoArbolado_" + imagenName.split('.')[0] + '.png', 0)


    AnnMap = np.zeros((500,500), np.float32)
    AnnMap[forestalNoArbolado == 255] = 1
    AnnMap[forestalArbolado == 255] = 2
    AnnMap[cultivosLenosos == 255] = 3
    AnnMap[cultivosHerbaceos == 255] = 4
    return AnnMap
def realLabelImg_v3(imagenName,path):
    agua = cv2.imread(path + "/Semantic/Agua/agua_" + imagenName.split('.')[0] + '.png', 0)
    cultivos =  cv2.imread(path + "/Semantic/Cultivos/cultivos_" + imagenName.split('.')[0] + '.png', 0)
    forestalArbolado = cv2.imread(path + "/Semantic/ForestalArbolado/forestalArbolado_" + imagenName.split('.')[0] + '.png', 0)
    forestalNoArbolado = cv2.imread(path + "/Semantic/ForestalNoArbolado/forestalNoArbolado_" + imagenName.split('.')[0] + '.png', 0)

    AnnMap = np.zeros((500, 500), np.float32)
    AnnMap[forestalNoArbolado == 255] = 1
    AnnMap[forestalArbolado == 255] = 2
    AnnMap[cultivos== 255] = 3
    AnnMap[agua == 255] = 4
    return AnnMap

def accuracyModel(Net,ListImages,path,bandsToUse = []):
    RateEqualPixelsSum = 0
    output_real = np.zeros((500,500),np.float32)
    for imagenName in ListImages:
        output_pred = PredNetOfImg(imagenName,Net,path,bandsToUse)
        if(path.split("/")[0] == "dataset_v2"):
            output_real = realLabelImg_v2(imagenName,path)
        elif(path.split("/")[0] == "dataset_v3" or path.split("/")[0] == "dataset_v3_tif"):
            output_real = realLabelImg_v3(imagenName, path)
        equalPixels = np.sum(output_pred == output_real)
        RateEqualPixelsSum += (equalPixels*100)/(500*500)
    return round(RateEqualPixelsSum/len(ListImages),3)

def printEqualRatePerImage(equalPixelsPerImage):
    for imagenRate in equalPixelsPerImage:
        print("MODELO: "+imagenRate[0]+" Accuracy: "+str(imagenRate[1])+ " %")

def accuracyPerModel(path,ListModels,ListImages,numClusters,modelFolder,bandsToUse = []):
    max = 0
    bestModel = ""
    i = 0
    Net = None
    for modelName in ListModels:
        i = i + 1
        modelPath = modelFolder+modelName
        print("modelo:" + modelPath)
        if "deeplab" in modelFolder:
            Net = preparingNet_deeplab(modelPath,numClusters)
        else:
            Net = preparingNet_fcn(modelPath,numClusters)
        accuracy = accuracyModel(Net,ListImages,path,bandsToUse)
        if accuracy > max:
            max = accuracy
            bestModel = modelName
    return bestModel,max

def bestModel_v2():
    bestModel, max = accuracyPerModel("dataset_v2/Validation/",ListModelsTrained_v2,ListImagesValidation_v2,5)
    print("El mejor modelo es: " + str(bestModel))
    print("Con un accuracy de :" + str(max) + " %")

def bestModel_v3():
    bestModel, max = accuracyPerModel("dataset_v3/Validation/",ListModelsTrained_v3_deeplab, ListImagesValidation_v3, 5, modelPath_v3_deeplab)
    print("El mejor modelo es: " + str(bestModel))
    print("Con un accuracy de :" + str(max) + " %")

def bestModel_v3_fcn():
    bestModel, max = accuracyPerModel("dataset_v3/Validation/",ListModelsTrained_v3_fcn,ListImagesValidation_v3, 5, modelPath_v3_fcn)
    print("El mejor modelo es: " + str(bestModel))
    print("Con un accuracy de :" + str(max) + " %")

def bestModel_v3_deeplab_infrarrojo(bandsToUse):
    bestModel, max = accuracyPerModel("dataset_v3_tif/Validation/", ListModelsTrained_v3_deeplab_infrarrojo,ListImagesValidation_v3_tif, 5, modelPath_v3_deeplab_infrarrojo, bandsToUse)
    print("El mejor modelo es: " + str(bestModel))
    print("Con un accuracy de :" + str(max) + " %")

def bestModel_v3_deeplab_agricultura(bandsToUse):
    bestModel, max = accuracyPerModel("dataset_v3_tif/Validation/", ListModelsTrained_v3_deeplab_agricultura, ListImagesValidation_v3_tif, 5,modelPath_v3_deeplab_agricultura, bandsToUse)
    print("El mejor modelo es: " + str(bestModel))
    print("Con un accuracy de :" + str(max) + " %")

def bestModel_v3_deeplab_vegetacion(bandsToUse):
    bestModel, max = accuracyPerModel("dataset_v3_tif/Validation/", ListModelsTrained_v3_deeplab_vegetacion, ListImagesValidation_v3_tif, 5,modelPath_v3_deeplab_vegetacion, bandsToUse)
    print("El mejor modelo es: " + str(bestModel))
    print("Con un accuracy de :" + str(max) + " %")

def bestModel_v3_deeplab_zonasurbanas():
    bestModel, max = accuracyPerModel("dataset_v3_tif/Validation/", ListModelsTrained_v3_deeplab_zonasurbanas, ListImagesValidation_v3_tif, 5,modelPath_v3_deeplab_zonasurbanas, [13,12,4])
    print("El mejor modelo es: " + str(bestModel))
    print("Con un accuracy de :" + str(max) + " %")

def bestModel_v3_deeplab_vegetacionsaludable():
    bestModel, max = accuracyPerModel("dataset_v3_tif/Validation/", ListModelsTrained_v3_deeplab_vegetacionsaludable, ListImagesValidation_v3_tif, 5,modelPath_v3_deeplab_vegetacionsaludable, [8,12,2])
    print("El mejor modelo es: " + str(bestModel))
    print("Con un accuracy de :" + str(max) + " %")
    
bestModel_v3_deeplab_zonasurbanas()


