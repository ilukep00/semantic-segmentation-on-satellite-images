import os
import matplotlib.pyplot as plt
import numpy as np
from osgeo import gdal
import cv2
import torchvision.models.segmentation
import torch
import torchvision.transforms as tf

os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

Learning_Rate = 1e-5 # Tamaño del ppaso del descenso del gradiente en el entrenamiento
width=height=500 # image width and height
batchSize=3 # nº de imagenes en cada iteracion, lo he bajado de 3 a 2 para que no pete en memoria (CUDA)

# Creamos una lista de todas las imagenes del dataset de entrenamiento
#DATASET_V1
ImagesFolder_v1 = "dataset_v1//"
ListImages_v1 = os.listdir(os.path.join(ImagesFolder_v1, "Images"))

#DATASET_V2
ImagesFolder_v2="dataset_v2//"
ListImages_v2=os.listdir(os.path.join(ImagesFolder_v2, "Training/Images"))

#DATASET_V3
ImagesFolder_v3="dataset_v3//"
ListImages_v3=os.listdir(os.path.join(ImagesFolder_v3, "Training/Images"))

#DATASET_V3_tif
ImagesFolder_v3_tif = "dataset_v3_tif//"
ListImages_v3_tif = os.listdir(os.path.join(ImagesFolder_v3_tif, "Training/Images"))

# Definimos un conjunto de transformaciones que se van a utilizar sobre las imagenes usando el modulo de tranformacion de TorchVision

transformImg=tf.Compose([tf.ToPILImage(mode='RGB'),tf.ToTensor()])
#este no se usa el de abajo
transformImg_tif = tf.Compose([tf.ToPILImage(mode='RGBA'),tf.ToTensor()])
transformAnn=tf.Compose([tf.ToPILImage(),tf.ToTensor()])

#Creamos una funcion que nos permitirá cargar una imagen aleatoria y  el correspondiente mapa de anotaciones para entrenamiento :

def ReadRandomImage_v1():

    # Primero, cogemos un indice aleatorio de la lista de imagenes y cargamos la imagen correspondiente
    index = np.random.randint(0,len(ListImages_v1))

    nameImage = ListImages_v1[index]
    pathImage = os.path.join(ImagesFolder_v1, "images",nameImage)

    img_array = cv2.imread(pathImage)
    #Después queremos cargar las máscaras de las diferentes clases de las imagenes
    SemanticFolder = "dataset_v1//"
    cultivos = cv2.imread(os.path.join(SemanticFolder,   "Semantic/Cultivos", "cultivos_"+nameImage.split('.')[0]+'.png'),0)
    forestal = cv2.imread(os.path.join(SemanticFolder,   "Semantic/Forestal", "forestal_"+nameImage.split('.')[0]+'.png'),0)
    improductivo = cv2.imread(os.path.join(SemanticFolder,   "Semantic/Improductivo", "improductivo_"+nameImage.split('.')[0]+'.png'),0)


    # Para entrenar la red, necesitamos crear un mapa de segmentacion donde los valores de pixeles que pertenecen a la clase NoVegetation sean 0,
    # a la clase Open sean 1, a la clase  SparseVegetation sean 2, a la clase ModerateVegetation sean 3 y a la clase DenseVegetation sean 4.
    AnnMap = np.zeros(img_array.shape[0:2],np.float32)
    AnnMap[cultivos == 255] = 1
    AnnMap[forestal == 255] = 2
    #plt.imshow(img_array)
    #plt.show()
    img_array = transformImg(img_array)
    #plt.imshow(AnnMap,cmap='gray', vmin = 0, vmax = 4)
    #plt.show()
    AnnMap = transformAnn(AnnMap)

    return img_array,AnnMap

def ReadRandomImage_v2():

    # Primero, cogemos un indice aleatorio de la lista de imagenes y cargamos la imagen correspondiente
    index = np.random.randint(0,len(ListImages_v2))

    nameImage = ListImages_v2[index]
    pathImage = os.path.join(ImagesFolder_v2, "Training/Images",nameImage)

    img_array = cv2.imread(pathImage)
    #Después queremos cargar las máscaras de las diferentes clases de las imagenes
    SemanticFolder = "dataset_v2//"
    cultivosHerbaceos = cv2.imread(os.path.join(SemanticFolder,   "Training/Semantic/CultivosHerbaceos", "cultivosHerbaceos_"+nameImage.split('.')[0]+'.png'),0)
    cultivosLenosos = cv2.imread(os.path.join(SemanticFolder,   "Training/Semantic/CultivosLenosos", "cultivosLenosos_"+nameImage.split('.')[0]+'.png'),0)
    forestalArbolado = cv2.imread(os.path.join(SemanticFolder,   "Training/Semantic/ForestalArbolado", "forestalArbolado_"+nameImage.split('.')[0]+'.png'),0)
    forestalNoArbolado = cv2.imread(os.path.join(SemanticFolder,   "Training/Semantic/ForestalNoArbolado", "forestalNoArbolado_"+nameImage.split('.')[0]+'.png'),0)
    improductivo = cv2.imread(os.path.join(SemanticFolder,   "Training/Semantic/Improductivo", "improductivo_"+nameImage.split('.')[0]+'.png'),0)


    # Para entrenar la red, necesitamos crear un mapa de segmentacion donde los valores de pixeles que pertenecen a la clase NoVegetation sean 0,
    # a la clase Open sean 1, a la clase  SparseVegetation sean 2, a la clase ModerateVegetation sean 3 y a la clase DenseVegetation sean 4.

    AnnMap = np.zeros(img_array.shape[0:2],np.float32)
    AnnMap[forestalNoArbolado == 255] = 1
    AnnMap[forestalArbolado == 255] = 2
    AnnMap[cultivosLenosos == 255] = 3
    AnnMap[cultivosHerbaceos == 255] = 4
    #plt.imshow(img_array)
    #plt.show()
    img_array = transformImg(img_array)
    #plt.imshow(AnnMap,cmap='gray', vmin = 0, vmax = 4)
    #plt.show()
    AnnMap = transformAnn(AnnMap)

    return img_array,AnnMap

def ReadRandomImage_v3():

    # Primero, cogemos un indice aleatorio de la lista de imagenes y cargamos la imagen correspondiente
    index = np.random.randint(0,len(ListImages_v3))

    nameImage = ListImages_v3[index]
    pathImage = os.path.join(ImagesFolder_v3, "Training/Images",nameImage)

    img_array = cv2.imread(pathImage)
    #Después queremos cargar las máscaras de las diferentes clases de las imagenes
    SemanticFolder = "dataset_v3//"
    agua = cv2.imread(os.path.join(SemanticFolder,   "Training/Semantic/Agua", "agua_"+nameImage.split('.')[0]+'.png'),0)
    cultivos = cv2.imread(os.path.join(SemanticFolder,   "Training/Semantic/Cultivos", "cultivos_"+nameImage.split('.')[0]+'.png'),0)
    forestalArbolado = cv2.imread(os.path.join(SemanticFolder,   "Training/Semantic/ForestalArbolado", "forestalArbolado_"+nameImage.split('.')[0]+'.png'),0)
    forestalNoArbolado = cv2.imread(os.path.join(SemanticFolder,   "Training/Semantic/ForestalNoArbolado", "forestalNoArbolado_"+nameImage.split('.')[0]+'.png'),0)
    improductivo = cv2.imread(os.path.join(SemanticFolder,   "Training/Semantic/Improductivo", "improductivo_"+nameImage.split('.')[0]+'.png'),0)


    # Para entrenar la red, necesitamos crear un mapa de segmentacion donde los valores de pixeles que pertenecen a la clase NoVegetation sean 0,
    # a la clase Open sean 1, a la clase  SparseVegetation sean 2, a la clase ModerateVegetation sean 3 y a la clase DenseVegetation sean 4.
    AnnMap = np.zeros(img_array.shape[0:2],np.float32)
    AnnMap[forestalNoArbolado == 255] = 1
    AnnMap[forestalArbolado == 255] = 2
    AnnMap[cultivos == 255] = 3
    AnnMap[agua == 255] = 4
    #plt.imshow(img_array)
    #plt.show()
    img_array = transformImg(img_array)
    #plt.imshow(AnnMap,cmap='gray', vmin = 0, vmax = 4)
    #plt.show()
    AnnMap = transformAnn(AnnMap)

    return img_array,AnnMap

def ReadRandomImage_v3_tif(bandsToUse):
    # Primero, cogemos un indice aleatorio de la lista de imagenes y cargamos la imagen correspondiente
    index = np.random.randint(0, len(ListImages_v3_tif))

    nameImage = ListImages_v3_tif[index]
    pathImage = os.path.join(ImagesFolder_v3_tif, "Training/Images", nameImage)

    img_tif = gdal.Open(pathImage)
    img_array_1Band = img_tif.GetRasterBand(bandsToUse[0]).ReadAsArray()
    img_array_2Band = img_tif.GetRasterBand(bandsToUse[1]).ReadAsArray()
    img_array_3Band = img_tif.GetRasterBand(bandsToUse[2]).ReadAsArray()

    img_array = np.zeros((img_array_1Band.shape[0],img_array_1Band.shape[1],3))
    img_array[:, :, 0] = img_array_1Band
    img_array[:, :, 1] = img_array_2Band
    img_array[:, :, 2] = img_array_3Band
    # Después queremos cargar las máscaras de las diferentes clases de las imagenes
    SemanticFolder = "dataset_v3_tif//"
    agua = cv2.imread(os.path.join(SemanticFolder, "Training/Semantic/Agua", "agua_" + nameImage.split('.')[0] + '.png'), 0)
    cultivos = cv2.imread(os.path.join(SemanticFolder, "Training/Semantic/Cultivos", "cultivos_" + nameImage.split('.')[0] + '.png'), 0)
    forestalArbolado = cv2.imread(os.path.join(SemanticFolder, "Training/Semantic/ForestalArbolado","forestalArbolado_" + nameImage.split('.')[0] + '.png'), 0)
    forestalNoArbolado = cv2.imread(os.path.join(SemanticFolder, "Training/Semantic/ForestalNoArbolado","forestalNoArbolado_" + nameImage.split('.')[0] + '.png'), 0)
    improductivo = cv2.imread(os.path.join(SemanticFolder, "Training/Semantic/Improductivo","improductivo_" + nameImage.split('.')[0] + '.png'), 0)

    # Para entrenar la red, necesitamos crear un mapa de segmentacion donde los valores de pixeles que pertenecen a la clase NoVegetation sean 0,
    # a la clase Open sean 1, a la clase  SparseVegetation sean 2, a la clase ModerateVegetation sean 3 y a la clase DenseVegetation sean 4.
    AnnMap = np.zeros(img_array.shape[0:2], np.float32)
    AnnMap[forestalNoArbolado == 255] = 1
    AnnMap[forestalArbolado == 255] = 2
    AnnMap[cultivos == 255] = 3
    AnnMap[agua == 255] = 4

    img_array = transformImg(img_array)

    AnnMap = transformAnn(AnnMap)

    return img_array, AnnMap
# Para el entrenamiento, necesitamos usar un lote de imagenes.
# Esto significa que varias imagenes son apiladas una encima de otra en una Matriz 4D
# Creamos el lote usando la siguiente función:

def LoadBatch(channels,version,bandsToUse): #Load bach of images

    # Esta primera parte crea una matriz vacia que guardara imagenes de dimensiones [batchSize, channels, height, width],
    # donde channels es el número de capas de la imagen; que es 3 para RGB y 1 para el mapa de anotaciones.
    images = torch.zeros([batchSize, channels, height, width])
    ann = torch.zeros([batchSize, height, width])

    # La siguiente parte carga un conjunto de imagenes y antaciones a la matriz vacia , usanndo la funcion definida anteriormente ReadRandomImage().
    for i in range(batchSize):
        if(version == 3):
            images[i], ann[i] = ReadRandomImage_v3_tif(bandsToUse)
        elif (version == 2):
            images[i], ann[i] = ReadRandomImage_v2()
        else:
            images[i], ann[i] = ReadRandomImage_v1()

    return images, ann

# IMrPLEMENTACION DE LA RED NEURONAL
# # La primera parte identifica si el computador tiene GPU o CPU. Si tiene Cuda GPU el entrenamiento se realiza en el GPU:
# # Para cualquier dataset pactico, entrenar usando la CPU es extremadamente lento.
def training_neuralNetwork(channels,version,pathToSaveModels,bandsToUse):
    #Net = None
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # A continuación, cargamos la segmentación semántica de deep lab net:
    Net =torchvision.models.segmentation.deeplabv3_resnet50(pretrained=False)

    #Queremos reemplazar la ultima capa convolucional con 21 neuronas de salida por una que tenga k neuronas de salida (Las k clases que tenemos):
    if version == 2 or version == 3:
        Net.classifier[4] = torch.nn.Conv2d(256, 5, kernel_size=(1, 1), stride=(1,1))
    elif version == 1:
        Net.classifier[4] = torch.nn.Conv2d(256, 3, kernel_size=(1, 1), stride=(1,1))

    #Cargamos nuestra red neuronal en nuestro dispositivo GPU o CPU:
    Net=Net.to(device)
    print(Net)
    if(channels == 4):
        Net.backbone['conv1'] = torch.nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    # Finalmente cargamos el optimizador
    # El optimizador controlara el error del gradiente durante las fases de backpropagation en el entrenamiento.
    # El optimizador Adam es uno de los más rapidos disponibles

    optimizer=torch.optim.Adam(params=Net.parameters(),lr=Learning_Rate) # Create adam optimizer

    #Finally, we start the training loop:
    error = []
    index = []
    for itr in range(30000):
        print(itr)

        # We load the batch of images and annotations:
        images, ann = LoadBatch(channels,version,bandsToUse)  # LoadBatch fue definida anteriormente y carga el lote de imagenes y mascaras.

        images = torch.autograd.Variable(images, requires_grad=False).to(device)  # Load image
        ann = torch.autograd.Variable(ann, requires_grad=False).to(device)  # Load annotation



        #- torch.autograd.Variable convierte los datos en variables  de gradiente que pueden ser usados por la red
        #- Seteamos Requires_grad=False  ya que no queremos aplicar el gradiente a la imagen, solo a las capas de la red
        #- to(device) copia el tensor al mismo dispositivo (GPU o CPU) que la red.

        # And finally we input the image to the net and get the prediction:
        Pred = Net(images)['out']  # make prediction

        # Una vez que hacemos la prediccion , podemos compararla con la real y calcular la perdida
        # Primero, definimos la funcion de perdida.
        criterion = torch.nn.CrossEntropyLoss()

        # usamos esa funcion para calcular la perdida entre la prediccion y la real.
        Loss = criterion(Pred, ann.long())
        # Una vez que calculamos la perdida, podemos aplicar el brackpropagation y cambiar los pesos de la red
        Loss.backward()  # Backpropogate loss
        optimizer.step()  # Apply gradient descent change to weight

        # Necesitamos guardar el modelo de entrenamiento, de lo contrario se perdera una vez el programa de detenga
        # Ahorrar requiere mucho tiempo por lo que queremos hacerlo una vez cada 1000 pasos
        if (itr % 100 == 0) and (itr != 0):
            print('Savinng Model' + str(itr) + '.torch')
            error.append(Loss.item())
            index.append(itr)
            torch.save(Net.state_dict(), pathToSaveModels+'/'+ str(itr) + '.torch')

    #Despues de correr este script almenos 3000 pasos , la red debera dar resultados decentes.

    plt.plot(index,error)
    plt.show()

training_neuralNetwork(3,3,'deeplabv3','modelos_v3_deeplab_vegetacionsaludable',[8,12,2])