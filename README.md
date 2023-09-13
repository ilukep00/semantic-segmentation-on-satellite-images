# Segmentación Semántica sobre imágenes por satélite

La teledetección es el área de la ciencia que permite la monitorización de la tierra sin tener
contacto con los objetos que observa. Desde los años 60 diferentes gobiernos y empresas
han lanzado satélites con el objetivo de monitorizar de la Tierra. Sin embargo, la identi-
cación de objetos y procesamiento de los datos es todavía una tarea manual y de precisión.
En este trabajo se desea automatizar la detección de objetos en los datos provenientes de
satélites con sensores multi-espectrales mediante técnicas de segmentación semántica. La
segmentación semántica es una técnica importante de la Visión Articial, que consiste en
otorgar a cada pixel de una imagen una categoría, consiguiendo generar subregiones que
contengan un signicado semántico en común.

En este trabajo se ha tratado de implementar esta técnica mediante la utilización de la librería Pytorch del lenguaje de programación Python, donde se espera clasicar los diferentes tipos de regiones en las imágenes por satélite: tierras, bosques, ríos, nubes...
