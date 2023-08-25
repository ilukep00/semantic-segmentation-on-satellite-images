import torchvision.models.segmentation
import torch
import torchvision.transforms as tf
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from osgeo import gdal


# IMAGENES DE TESTING
imagesFolder_v3 = "dataset_v3_tif//"
ListImagesTesting_v3 = os.listdir(os.path.join(imagesFolder_v3,"Testing/NAVARRA_2021/Images"))

# MODELOS VERSION 3
model_v3_deeplab = "modelos_v3_deeplab_vegetacionsaludable/14000.torch"


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
    equalPixelsPerImage = []
    RateEqualPixelsSum = 0
    output_real = np.zeros((500,500),np.float32)
    for imagenName in ListImages:
        print(imagenName)
        output_pred = PredNetOfImg(imagenName,Net,path,bandsToUse)
        if(path.split("/")[0] == "dataset_v2"):
            output_real = realLabelImg_v2(imagenName,path)
        elif(path.split("/")[0] == "dataset_v3" or path.split("/")[0] == "dataset_v3_tif"):
            output_real = realLabelImg_v3(imagenName, path)
        equalPixels = np.sum(output_pred == output_real)
        print(equalPixels)
        equalPixelsPerImage.append((imagenName,(equalPixels*100)/(500*500)))
        RateEqualPixelsSum += (equalPixels*100)/(500*500)
    return round(RateEqualPixelsSum/len(ListImages),3),equalPixelsPerImage

Net = preparingNet_deeplab(model_v3_deeplab,5)
accuracy, accuracyPerImage = accuracyModel(Net,ListImagesTesting_v3,"dataset_v3_tif/Testing/NAVARRA_2021",[8,12,2])
print("accuracy del modelo 14000.torch es: ",accuracy)