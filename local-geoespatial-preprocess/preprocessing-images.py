import matplotlib.pyplot as plt
from osgeo import gdal
import numpy as np
import cv2

#path images
path_winter = "dataset/clustering/Cultivos_clases/NavarreinWinter.tif"
path_spring = "dataset/clustering/Cultivos_clases/NavarreinSpring.tif"
path_summer = "dataset/clustering/Cultivos_clases/NavarreinSummer.tif"
path_autumn = "dataset/clustering/Cultivos_clases/NavarreinAutumn.tif"

#opening images
#winter_tif = gdal.Open(path_winter)
spring_tif = gdal.Open(path_spring)
summer_tif = gdal.Open(path_summer)
autumn_tif = gdal.Open(path_autumn)


#information
#print("***********INFORMATION WINTER-2019***********")
#print("Projection: ", winter_tif.GetProjection())  # get projection
#print("Columns:", winter_tif.RasterXSize)  # number of columns
#print("Rows:", winter_tif.RasterYSize)  # number of rows
#print("Band count:", winter_tif.RasterCount)  # number of bands
#print("GeoTransform", winter_tif.GetGeoTransform())


#print("***********INFORMATION SPRING-2019***********")
#print("Projection: ", spring_tif.GetProjection())  # get projection
#print("Columns:", spring_tif.RasterXSize)  # number of columns
#print("Rows:", spring_tif.RasterYSize)  # number of rows
#print("Band count:", spring_tif.RasterCount)  # number of bands
#print("GeoTransform", spring_tif.GetGeoTransform())

#print("***********INFORMATION SUMMER-2019***********")
#print("Projection: ", summer_tif.GetProjection())  # get projection
#print("Columns:", summer_tif.RasterXSize)  # number of columns
#print("Rows:", summer_tif.RasterYSize)  # number of rows
#print("Band count:", summer_tif.RasterCount)  # number of bands
#print("GeoTransform", summer_tif.GetGeoTransform())

#print("***********INFORMATION AUTUMN-2019***********")
#print("Projection: ", autumn_tif.GetProjection())  # get projection
#print("Columns:", autumn_tif.RasterXSize)  # number of columns
#print("Rows:", autumn_tif.RasterYSize)  # number of rows
#print("Band count:", autumn_tif.RasterCount)  # number of bands
#print("GeoTransform", autumn_tif.GetGeoTransform())

#convert to array
#winter_array = winter_tif.ReadAsArray()
#print("shape : ",winter_array.shape)
spring_array = spring_tif.ReadAsArray()
summer_array = summer_tif.ReadAsArray()
autumn_array = autumn_tif.ReadAsArray()

def divide_image(image_array,filename,format,size):
    i_index = 0
    for i in range(0,image_array.shape[1],size):
        j_index = 0
        for j in range(0,image_array.shape[2],size):
            i_top = i
            i_bottom = i + size
            j_left = j
            j_rigth = j + size
            if i + size > image_array.shape[1]:
                i_bottom = image_array.shape[1]
            if j + size > image_array.shape[2]:
                j_rigth = image_array.shape[2]
            if(format == "jpg"):
                image_to_write = np.zeros((i_bottom-i_top, j_rigth-j_left, 3))
                image_to_write[:, :, 0] = image_array[3, i_top:i_bottom, j_left:j_rigth]
                image_to_write[:, :, 1] = image_array[2, i_top:i_bottom, j_left:j_rigth]
                image_to_write[:, :, 2] = image_array[1, i_top:i_bottom, j_left:j_rigth]
                if (image_to_write.max() > 1):
                    image_to_write = image_to_write / image_to_write.max()
                plt.imsave("dataset/clustering/Cultivos_clases/NavarreImages/"+filename+"/jpg/"+str(size)+'/'+filename+'X'+str(size)+'-'+str(i_index)+'-'+str(j_index)+'.jpg',image_to_write)
            elif(format == 'tif'):
                if image_array.dtype == np.float32:
                    arr_type = gdal.GDT_Float32
                else:
                    arr_type = gdal.GDT_Int32
                driver = gdal.GetDriverByName("GTiff")
                out_ds = driver.Create("dataset/clustering/Cultivos_clases/NavarreImages/"+filename+"/tif/"+str(size)+'/'+filename+'X'+str(size)+'-'+str(i_index)+'-'+str(j_index)+'.tif',j_rigth-j_left, i_bottom-i_top, 16, arr_type)
                out_ds.WriteArray(image_array[:,i_top:i_bottom, j_left:j_rigth])
            j_index+=1
        i_index+=1

#************ winter divisions *****************

#divide_image(winter_array,'winternavarre','jpg',256)
#divide_image(winter_array,'winternavarre','tif',256)

#divide_image(winter_array,'winternavarre','jpg',512)
#divide_image(winter_array,'winternavarre','tif',512)

#************ spring divisions *****************

#divide_image(spring_array,'springnavarre','jpg',256)
#divide_image(spring_array,'springnavarre','tif',256)

divide_image(spring_array,'Spring','jpg',512)
divide_image(spring_array,'Spring','tif',512)

#************ summer divisions *****************

#divide_image(summer_array,'summernavarre','jpg',256)
#divide_image(summer_array,'summernavarre','tif',256)

divide_image(summer_array,'Summer','jpg',512)
divide_image(summer_array,'Summer','tif',512)

#************ autumn divisions *****************

#divide_image(autumn_array,'autumnnavarre','jpg',256)
#divide_image(autumn_array,'autumnnavarre','tif',256)

divide_image(autumn_array,'Autumn','jpg',512)
divide_image(autumn_array,'Autumn','tif',512)











