import torchvision.models.segmentation
import torch
import torchvision.transforms as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from osgeo import gdal
import os
import cv2

#VERSION 2
modelPath_v2 = "modelos_v2/27600.torch"

imagesTestingFolderLR_v2 = "dataset_v2/Testing/LARIOJA//"
ListImagesTestingLR_v2 = os.listdir(os.path.join(imagesTestingFolderLR_v2, "Images"))

imagesTestingFolderNV_v2 = "dataset_v2/Testing/NAVARRA_2021//"
ListImagesTestingNV_v2 = os.listdir(os.path.join(imagesTestingFolderNV_v2, "Images"))

#VERSION 3
modelPath_v3 = "modelos_v3_deeplab/28600.torch"

imagesTestingFolderLR_v3 = "dataset_v3/Testing/LARIOJA//"
ListImagesTestingLR_v3 = os.listdir(os.path.join(imagesTestingFolderLR_v3, "Images"))

imagesTestingFolderNV_v3 = "dataset_v3/Testing/NAVARRA_2021//"
ListImagesTestingNV_v3 = os.listdir(os.path.join(imagesTestingFolderNV_v3, "Images"))

#VERSION 3 FCN
modelPath_v3_fcn = "modelos_v3_fcn/27000.torch"

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
def predNetOfImg(imageName,Net,pathFolder):
    img_array = cv2.imread(pathFolder+"/Images/"+imageName)
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
def imgPredInColor_v2(img,seg):
    output = np.zeros(img.shape)
    values = np.unique(seg.ravel())
    values3D = [np.array([0, 0, 0])]
    for value in values:
        if (value == 1):  # forestal no arbolado :) 485727
            output[seg == value, 0] = 35
            output[seg == value, 1] = 142
            output[seg == value, 2] = 107
            values3D.append(np.array([35, 142, 107]))
        elif (value == 2):  # forestal arbolado :)
            output[seg == value, 1] = 100
            values3D.append(np.array([0, 100, 0]))
        elif (value == 3):  # cultivos leñosos
            output[seg == value, 0] = 30
            output[seg == value, 1] = 105
            output[seg == value, 2] = 210
            values3D.append(np.array([30, 105, 210]))
        elif (value == 4):  # cultivos Herbaceos
            output[seg == value, 1] = 255
            output[seg == value, 2] = 255
            values3D.append(np.array([0, 255, 255]))
    return output,values3D



def realImgLabeledColor_v2(img,imgname,pathFolder):
    realLabelImg = np.zeros(img.shape)

    cultivosHerbaceosPath = pathFolder+"/Semantic/CultivosHerbaceos/cultivosHerbaceos_" + str(imgname) + ".png"
    cultivosHerbaceos = cv2.imread(cultivosHerbaceosPath,0)
    realLabelImg[cultivosHerbaceos == 255, 1] = 255
    realLabelImg[cultivosHerbaceos == 255, 2] = 255

    cultivosLenososPath = pathFolder+"/Semantic/CultivosLenosos/cultivosLenosos_" + str(imgname) + ".png"
    cultivosLenosos = cv2.imread(cultivosLenososPath,0)
    realLabelImg[cultivosLenosos == 255, 0] = 30
    realLabelImg[cultivosLenosos == 255, 1] = 105
    realLabelImg[cultivosLenosos == 255, 2] = 210

    forestalArboladoPath = pathFolder+"/Semantic/ForestalArbolado/forestalArbolado_" + str(imgname) + ".png"
    forestalArbolado = cv2.imread(forestalArboladoPath,0)
    realLabelImg[forestalArbolado == 255, 1] = 100

    forestalNoArboladoPath = pathFolder+"/Semantic/ForestalNoArbolado/forestalNoArbolado_" + str(imgname) + ".png"
    forestalNoArbolado = cv2.imread(forestalNoArboladoPath,0)
    realLabelImg[forestalNoArbolado == 255, 0] = 35
    realLabelImg[forestalNoArbolado == 255, 1] = 142
    realLabelImg[forestalNoArbolado == 255, 2] = 107


    return realLabelImg

def saveTwoColorImagesWithColorLegend_v2(imgLeft, imgRight,valuescolor,imageName,pathFolder,folder):
    fig, axs = plt.subplot_mosaic([['left', 'right']], layout='constrained')
    axs['left'].imshow(imgLeft[:, :, ::-1] / 255)
    im = axs['right'].imshow(imgRight[:, :, ::-1] / 255)
    patches = []
    for value in valuescolor:
        color = np.ones(4)
        if (np.all(value == [0, 0, 0])):
            color[:3] = value[::-1] / 255
            patches.append(mpatches.Patch(color=color, label="No Vegetacion"))
        elif (np.all(value == [35, 142, 107])):
            color[:3] = value[::-1] / 255
            patches.append(mpatches.Patch(color=color, label="forestal No Arbolado"))
        elif (np.all(value == [0, 100, 0])):
            color[:3] = value[::-1] / 255
            patches.append(mpatches.Patch(color=color, label="forestal arbolado"))
        elif (np.all(value == [30, 105, 210])):
            color[:3] = value[::-1] / 255
            patches.append(mpatches.Patch(color=color, label="cultivos Leñosos"))
        elif (np.all(value == [0, 255, 255])):
            color[:3] = value[::-1] / 255
            patches.append(mpatches.Patch(color=color, label="cultivos Herbaceos"))
    fig.legend(handles=patches, loc='outside upper right')
    plt.savefig(pathFolder+"/outputs/"+folder+"/output_" + imageName)

def imgPredInColor_v3(img,seg):
    output = np.zeros(img.shape)
    values = np.unique(seg.ravel())
    values3D = [np.array([0, 0, 0])]
    for value in values:
        if (value == 1):  # forestal no arbolado :) 485727
            output[seg == value, 0] = 35
            output[seg == value, 1] = 142
            output[seg == value, 2] = 107
            values3D.append(np.array([35, 142, 107]))
        elif (value == 2):  # forestal arbolado :)
            output[seg == value, 1] = 100
            values3D.append(np.array([0, 100, 0]))
        elif (value == 3):  # cultivos
            output[seg == value, 1] = 255
            output[seg == value, 2] = 255
            values3D.append(np.array([0, 255, 255]))
        elif (value == 4):  # agua
            output[seg == value, 0] = 255
            values3D.append(np.array([255, 0, 0]))
    return output,values3D

def realImgLabeledColor_v3(img,imgname,pathFolder):
    realLabelImg = np.zeros(img.shape)

    forestalArboladoPath = pathFolder + "/Semantic/ForestalArbolado/forestalArbolado_" + str(imgname) + ".png"
    forestalArbolado = cv2.imread(forestalArboladoPath, 0)
    realLabelImg[forestalArbolado == 255, 1] = 100

    forestalNoArboladoPath = pathFolder + "/Semantic/ForestalNoArbolado/forestalNoArbolado_" + str(imgname) + ".png"
    forestalNoArbolado = cv2.imread(forestalNoArboladoPath, 0)
    realLabelImg[forestalNoArbolado == 255, 0] = 35
    realLabelImg[forestalNoArbolado == 255, 1] = 142
    realLabelImg[forestalNoArbolado == 255, 2] = 107

    cultivos = pathFolder + "/Semantic/Cultivos/cultivos_" + str(imgname) + ".png"
    cultivos = cv2.imread(cultivos, 0)
    realLabelImg[cultivos == 255, 1] = 255
    realLabelImg[cultivos == 255, 2] = 255

    aguaPath = pathFolder + "/Semantic/Agua/agua_" + str(imgname) + ".png"
    agua = cv2.imread(aguaPath, 0)
    realLabelImg[agua == 255, 0] = 255

    return realLabelImg

def saveTwoColorImagesWithColorLegend_v3(imgLeft, imgRight,valuescolor,imageName,pathFolder,folder):
    fig, axs = plt.subplot_mosaic([['left', 'right']], layout='constrained')
    axs['left'].imshow(imgLeft[:, :, ::-1] / 255)
    im = axs['right'].imshow(imgRight[:, :, ::-1] / 255)
    patches = []
    for value in valuescolor:
        color = np.ones(4)
        if (np.all(value == [0, 0, 0])):
            color[:3] = value[::-1] / 255
            patches.append(mpatches.Patch(color=color, label="No Vegetacion"))
        elif (np.all(value == [35, 142, 107])):
            color[:3] = value[::-1] / 255
            patches.append(mpatches.Patch(color=color, label="forestal No Arbolado"))
        elif (np.all(value == [0, 100, 0])):
            color[:3] = value[::-1] / 255
            patches.append(mpatches.Patch(color=color, label="forestal arbolado"))
        elif (np.all(value == [0, 255, 255])):
            color[:3] = value[::-1] / 255
            patches.append(mpatches.Patch(color=color, label="cultivos"))
        elif (np.all(value == [255, 0, 0])):
            color[:3] = value[::-1] / 255
            patches.append(mpatches.Patch(color=color, label="agua"))
    fig.legend(handles=patches, loc='outside upper right')
    plt.savefig(pathFolder+"/outputs_fcn/"+folder+"/output_" + imageName)
def saveRealImageVsLabelImgInColor(seg,imageName,pathFolder):
    imagePath = pathFolder+"/Images/" + imageName
    img = cv2.imread(imagePath);
    if "dataset_v2" in imagePath:
        output,values3D = imgPredInColor_v2(img,seg)
        saveTwoColorImagesWithColorLegend_v2(img,output,values3D,imageName,pathFolder,"outputsTesting_color")
    elif "dataset_v3" in imagePath:
        output, values3D = imgPredInColor_v3(img, seg)
        saveTwoColorImagesWithColorLegend_v3(img, output, values3D, imageName, pathFolder, "outputsTesting_color")

def saveRealImageLabeledVsLabelImgInColor(seg,imageName,pathFolder):
    imagePath = pathFolder + "/Images/" + imageName
    img = cv2.imread(imagePath);
    if "dataset_v2" in imagePath:
        realOutput = realImgLabeledColor_v2(img,imageName.split(".")[0], pathFolder)
        predOutput, values3D = imgPredInColor_v2(img, seg)
        saveTwoColorImagesWithColorLegend_v2(realOutput, predOutput, values3D,imageName,pathFolder,"outputsTestingvsRealOutput_color")
    elif "dataset_v3" in imagePath:
        realOutput = realImgLabeledColor_v3(img, imageName.split(".")[0], pathFolder)
        predOutput, values3D = imgPredInColor_v3(img, seg)
        saveTwoColorImagesWithColorLegend_v3(realOutput, predOutput, values3D, imageName, pathFolder, "outputsTestingvsRealOutput_color")

def outputTesting_color(ListImages,pathFolder,modelPath):
    version = modelPath.split("/")[0].split("_")[1]  # version de la imagen
    if "deeplab" in modelPath:
        if version == 'v1':
            #Preparando la red neuronal deeplab para version 1 de clases
            Net = preparingNet_deeplab(modelPath, 3)
        else:
            # Preparando la red neuronal
            Net = preparingNet_deeplab(modelPath, 5)
    elif "fcn" in modelPath:
        if version == 'v1':
            # Preparando la red neuronal deeplab para version 1 de clases
            Net = preparingNet_fcn(modelPath, 3)
        else:
            # Preparando la red neuronal
            Net = preparingNet_fcn(modelPath, 5)
    for imageName in ListImages:
        saveRealImageVsLabelImgInColor(predNetOfImg(imageName,Net,pathFolder),imageName,pathFolder)

def outputTestingVsRealOutput_color(ListImages,pathFolder,modelPath):
    version = modelPath.split("/")[0].split("_")[1]  # version de la imagen
    if "deeplab" in modelPath:
        if version == 'v1':
            # Preparando la red neuronal deeplab para version 1 de clases
            Net = preparingNet_deeplab(modelPath, 3)
        else:
            # Preparando la red neuronal
            Net = preparingNet_deeplab(modelPath, 5)
    elif "fcn" in modelPath:
        if version == 'v1':
            # Preparando la red neuronal deeplab para version 1 de clases
            Net = preparingNet_fcn(modelPath, 3)
        else:
            # Preparando la red neuronal
            Net = preparingNet_fcn(modelPath, 5)
    for imageName in ListImages:
        saveRealImageLabeledVsLabelImgInColor(predNetOfImg(imageName, Net, pathFolder), imageName, pathFolder)




outputTesting_color(ListImagesTestingNV_v3,imagesTestingFolderNV_v3,modelPath_v3_fcn)
outputTestingVsRealOutput_color(ListImagesTestingNV_v3,imagesTestingFolderNV_v3,modelPath_v3_fcn)
