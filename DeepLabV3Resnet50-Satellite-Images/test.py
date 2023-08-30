import torchvision.models.segmentation
import torch
import torchvision.transforms as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from osgeo import gdal
import os
import cv2
from pyparsing import col

os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
modelPath_v2 = "modelos_v2/27600.torch" # El path del  modelo entrenado
modelPath_v3_deeplab = "modelos_v3_deeplab/28600.torch" # El path del  modelo entrenado
modelPath_v3_fcn = "modelos_v3_fcn/27000.torch" # El path del  modelo entrenado
#********** CAMBIAR PATH *********
#imagePath = "dataset/Validation/Images/jpg/LARIOJA63.jpg" #El path de la imagen de test a segmentar
#imagePath = "dataset_v2/Testing/LARIOJAYPV4.jpg"
imagePath_v2 = "dataset_v2/Validation/Images/ULTZAMA9.jpg"
imagePath_v3 = "dataset_v3/Testing/NAVARRA_2021/Images/BAZTANYULTZAMA1.jpg"
height = width = 512 # Tamaño de la imagen a segmentar

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


#cargamos la imagen de test
#img_tif = gdal.Open(imagePath)
def predNetOfImg(imagePath,Net):
    img_array = cv2.imread(imagePath)
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


#mostramos la imagen de etiquetas real vs la predicha

def showImageRealVsLabelImg_v2(seg,imagePath):
    values = np.unique(seg.ravel())
    img = cv2.imread(imagePath,0)
    fig, axs = plt.subplot_mosaic([['left', 'right']], layout='constrained')
    axs['left'].imshow(img,cmap='gray',vmin = 0, vmax = 255,  label="imagen Real")
    im = axs['right'].imshow(seg,cmap='gray',vmin = 0, vmax = 5, label="test3")
    colors = [ im.cmap(im.norm(value)) for value in values]
    patches = []
    print(colors)
    for i in range(len(values)):
        if colors[i] == (0.0, 0.0, 0.0, 1.0): # 0 no vegetacion
            patches.append( mpatches.Patch(color=colors[i], label="No Vegetacion" ))
        elif colors[i] == (0.2, 0.2, 0.2, 1.0): # 1 forestal no arbolado
            patches.append( mpatches.Patch(color=colors[i], label="forestal No Arbolado "))
        elif colors[i] ==  (0.4, 0.4, 0.4, 1.0): # 2 forestal arbolado
            patches.append(mpatches.Patch(color=colors[i], label="forestal arbolado"))
        elif colors[i] ==  (0.6, 0.6, 0.6, 1.0): # 3 cultivos Leñosos
            patches.append(mpatches.Patch(color=colors[i], label="cultivos Leñosos"))
        elif colors[i] ==  (0.8, 0.8, 0.8, 1.0): # 4 cultivos Herbaceos
            patches.append(mpatches.Patch(color=colors[i], label="cultivos Herbaceos"))
    fig.legend(handles=patches, loc='outside upper right' )
    plt.show()
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

def plottingTwo3DImagesWithColorLegend_v2(imgLeft, imgRight,values3D):
    fig, axs = plt.subplot_mosaic([['left', 'right']], layout='constrained')
    axs['left'].imshow(imgLeft[:, :, ::-1] / 255)
    im = axs['right'].imshow(imgRight[:, :, ::-1] / 255)
    patches = []
    for value in values3D:
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
    plt.show()
def realImgLabeled_v2(imgname,imagePath):
    img = cv2.imread(imagePath);
    realLabelImg = np.zeros(img.shape)

    cultivosHerbaceosPath = "dataset_v2/Validation/Semantic/CultivosHerbaceos/cultivosHerbaceos_" + str(imgname) + ".png"
    cultivosHerbaceos = cv2.imread(cultivosHerbaceosPath,0)
    realLabelImg[cultivosHerbaceos == 255, 1] = 255
    realLabelImg[cultivosHerbaceos == 255, 2] = 255

    cultivosLenososPath = "dataset_v2/Validation/Semantic/CultivosLenosos/cultivosLenosos_" + str(imgname) + ".png"
    cultivosLenosos = cv2.imread(cultivosLenososPath,0)
    realLabelImg[cultivosLenosos == 255, 0] = 30
    realLabelImg[cultivosLenosos == 255, 1] = 105
    realLabelImg[cultivosLenosos == 255, 2] = 210

    forestalArboladoPath = "dataset_v2/Validation/Semantic/ForestalArbolado/forestalArbolado_" + str(imgname) + ".png"
    forestalArbolado = cv2.imread(forestalArboladoPath,0)
    realLabelImg[forestalArbolado == 255, 1] = 100

    forestalNoArboladoPath = "dataset_v2/Validation/Semantic/ForestalNoArbolado/forestalNoArbolado_" + str(imgname) + ".png"
    forestalNoArbolado = cv2.imread(forestalNoArboladoPath,0)
    realLabelImg[forestalNoArbolado == 255, 0] = 35
    realLabelImg[forestalNoArbolado == 255, 1] = 142
    realLabelImg[forestalNoArbolado == 255, 2] = 107

    #plt.imshow(realLabelImg[:,:,::-1]/255)
    #plt.show()

    return realLabelImg

#***************************************************************
#*------------------------- VERSION 3 -------------------------*
#***************************************************************
def showImageRealVsLabelImg_v3(seg,imagePath):
    values = np.unique(seg.ravel())
    img = cv2.imread(imagePath, 0)
    fig, axs = plt.subplot_mosaic([['left', 'right']], layout='constrained')
    axs['left'].imshow(img, cmap='gray', vmin=0, vmax=255, label="imagen Real")
    im = axs['right'].imshow(seg, cmap='gray', vmin=0, vmax=5, label="test3")
    colors = [im.cmap(im.norm(value)) for value in values]
    patches = []
    for i in range(len(values)):
        if colors[i] == (0.0, 0.0, 0.0, 1.0):  # 0 no vegetacion
            patches.append(mpatches.Patch(color=colors[i], label="No Vegetacion"))
        elif colors[i] == (0.2, 0.2, 0.2, 1.0):  # 1 forestal no arbolado
            patches.append(mpatches.Patch(color=colors[i], label="forestal No Arbolado "))
        elif colors[i] == (0.4, 0.4, 0.4, 1.0):  # 2 forestal arbolado
            patches.append(mpatches.Patch(color=colors[i], label="forestal arbolado"))
        elif colors[i] == (0.6, 0.6, 0.6, 1.0):  # 3 cultivos Leñosos
            patches.append(mpatches.Patch(color=colors[i], label="cultivos"))
        elif colors[i] == (0.8, 0.8, 0.8, 1.0):  # 4 cultivos Herbaceos
            patches.append(mpatches.Patch(color=colors[i], label="agua"))
    fig.legend(handles=patches, loc='outside upper right')
    plt.show()

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

def plottingTwo3DImagesWithColorLegend_v3(imgLeft, imgRight,values3D):
    fig, axs = plt.subplot_mosaic([['left', 'right']], layout='constrained')
    axs['left'].imshow(imgLeft[:, :, ::-1] / 255)
    im = axs['right'].imshow(imgRight[:, :, ::-1] / 255)
    patches = []
    for value in values3D:
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
    plt.show()
def realImgLabeled_v3(imgname,imagePath):
    img = cv2.imread(imagePath);
    realLabelImg = np.zeros(img.shape)
    pathSemantic = imagePath.split("Images")[0]

    forestalArboladoPath = pathSemantic+"/Semantic/ForestalArbolado/forestalArbolado_" + str(imgname) + ".png"
    forestalArbolado = cv2.imread(forestalArboladoPath,0)
    realLabelImg[forestalArbolado == 255, 1] = 100

    forestalNoArboladoPath = pathSemantic+"/Semantic/ForestalNoArbolado/forestalNoArbolado_" + str(imgname) + ".png"
    forestalNoArbolado = cv2.imread(forestalNoArboladoPath,0)
    realLabelImg[forestalNoArbolado == 255, 0] = 35
    realLabelImg[forestalNoArbolado == 255, 1] = 142
    realLabelImg[forestalNoArbolado == 255, 2] = 107

    cultivos = pathSemantic+"/Semantic/Cultivos/cultivos_" + str(imgname) + ".png"
    cultivos = cv2.imread(cultivos,0)
    realLabelImg[cultivos == 255, 1] = 255
    realLabelImg[cultivos == 255, 2] = 255

    aguaPath = pathSemantic+"/Semantic/Agua/agua_" + str(imgname) + ".png"
    agua = cv2.imread(aguaPath,0)
    realLabelImg[agua == 255, 0] = 255

    #plt.imshow(realLabelImg[:,:,::-1]/255)
    #plt.show()
    return realLabelImg

def showImageRealVsLabelImg(seg,imagePath):
    if "dataset_v3" in imagePath:
        showImageRealVsLabelImg_v3(seg,imagePath)
    elif "dataset_v2" in imagePath:
        showImageRealVsLabelImg_v2(seg, imagePath)
def showRealImageVsLabelImgInColor(seg,imagePath):
    img = cv2.imread(imagePath);
    if "dataset_v3" in imagePath:
        output,values3D = imgPredInColor_v3(img,seg)
        plottingTwo3DImagesWithColorLegend_v3(img,output,values3D)
    elif "dataset_v2" in imagePath:
        output, values3D = imgPredInColor_v2(img, seg)
        plottingTwo3DImagesWithColorLegend_v2(img, output, values3D)
def showRealImageLabeledVsLabelImgInColor(seg,imgname,imagePath):
    img = cv2.imread(imagePath);
    if "dataset_v3" in imagePath:
        realOutput = realImgLabeled_v3(imgname, imagePath)
        predOutput, values3D = imgPredInColor_v3(img, seg)
        plottingTwo3DImagesWithColorLegend_v3(realOutput, predOutput, values3D)
    elif "dataset_v2" in imagePath:
        realOutput = realImgLabeled_v2(imgname, imagePath)
        predOutput, values3D = imgPredInColor_v2(img, seg)
        plottingTwo3DImagesWithColorLegend_v2(realOutput, predOutput, values3D)

def testImage_v3(modelPath,imagePath):
    modelVersion = modelPath.split("/")[0].split("_")[1] #version del Modelo
    imageVersion = imagePath.split("/")[0].split("_")[1] #version de la imagen
    imageName = imagePath_v3.split("/")[len(imagePath_v3.split("/"))-1].split('.')[0]
    if(modelVersion != imageVersion): #compatibilidad de versiones
        print("Diferentes versiones de tipos")
        return;
    if "deeplab" in modelPath:
        if modelVersion == 'v1':
            #Preparando la red neuronal deeplab para version 1 de clases
            Net = preparingNet_deeplab(modelPath, 3)
        else:
            # Preparando la red neuronal
            Net = preparingNet_deeplab(modelPath, 5)
    elif "fcn" in modelPath:
        if modelVersion == 'v1':
            # Preparando la red neuronal deeplab para version 1 de clases
            Net = preparingNet_fcn(modelPath, 3)
        else:
            # Preparando la red neuronal
            Net = preparingNet_fcn(modelPath, 5)
    seg = predNetOfImg(imagePath,Net)

    #mostrando la imagen real vs predicha en blanco y negro
    showImageRealVsLabelImg(seg,imagePath)
    showRealImageVsLabelImgInColor(seg,imagePath)
    showRealImageLabeledVsLabelImgInColor(seg,imageName,imagePath)

testImage_v3(modelPath_v3_deeplab,imagePath_v3)