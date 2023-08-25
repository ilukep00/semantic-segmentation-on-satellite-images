//---------------------------- IMPORTS --------------------------------------------
var miRegionNavarra = ee.FeatureCollection("projects/ee-lukepastor-test/assets/navarre"),
    navarraCultivos2021 = ee.FeatureCollection("projects/ee-lukepastor-test/assets/MapaCultivosNavarra"),
    dataset = ee.ImageCollection("COPERNICUS/S2"),
    cuencaPamplona = 
    /* color: #0b4a8b */
    /* displayProperties: [
      {
        "type": "rectangle"
      }
    ] */
    ee.Geometry.Polygon(
        [[[-1.6542574308587743, 42.86511419279247],
          [-1.6542574308587743, 42.69174252480807],
          [-1.2649294279290868, 42.69174252480807],
          [-1.2649294279290868, 42.86511419279247]]], null, false),
    zonaMedia = 
    /* color: #ffc82d */
    /* displayProperties: [
      {
        "type": "rectangle"
      }
    ] */
    ee.Geometry.Polygon(
        [[[-1.9028231046868993, 42.44503839626763],
          [-1.9028231046868993, 42.297925147754654],
          [-1.6727968595697118, 42.297925147754654],
          [-1.6727968595697118, 42.44503839626763]]], null, false),
    ribera = 
    /* color: #00ffff */
    /* displayProperties: [
      {
        "type": "rectangle"
      }
    ] */
    ee.Geometry.Polygon(
        [[[-1.6202165063890184, 42.187634431527485],
          [-1.6202165063890184, 41.96337951439402],
          [-1.4056397851976121, 41.96337951439402],
          [-1.4056397851976121, 42.187634431527485]]], null, false),
    pirineos = 
    /* color: #bf04c2 */
    /* displayProperties: [
      {
        "type": "rectangle"
      }
    ] */
    ee.Geometry.Polygon(
        [[[-1.129848890227374, 42.93186147488595],
          [-1.129848890227374, 42.768759225975494],
          [-0.871670179289874, 42.768759225975494],
          [-0.871670179289874, 42.93186147488595]]], null, false),
    urbasayandia = 
    /* color: #ff0000 */
    /* displayProperties: [
      {
        "type": "rectangle"
      }
    ] */
    ee.Geometry.Polygon(
        [[[-2.207882337492999, 42.926833896801156],
          [-2.207882337492999, 42.8100788975056],
          [-1.804134778899249, 42.8100788975056],
          [-1.804134778899249, 42.926833896801156]]], null, false),
    baztanyultzama = 
    /* color: #00ff00 */
    /* displayProperties: [
      {
        "type": "rectangle"
      }
    ] */
    ee.Geometry.Polygon(
        [[[-1.8969071569492302, 43.189497311934225],
          [-1.8969071569492302, 42.9034717622558],
          [-1.6661942663242302, 42.9034717622558],
          [-1.6661942663242302, 43.189497311934225]]], null, false);



function imagenNavarra(){
  var collectionCompositeList = dataset
    .filterDate('2021-04-01','2021-10-31') //TOI
    .filterBounds(miRegionNavarra) //ROI
    .filterMetadata('CLOUDY_PIXEL_PERCENTAGE','less_than',1)
    .sort('CLOUDY_PIXEL_PERCENTAGE',false)
    .toList(50)
    
  print(collectionCompositeList)
  
  var imagen = ee.ImageCollection([ee.Image(collectionCompositeList.get(12)),
                                   ee.Image(collectionCompositeList.get(15)),
                                   ee.Image(collectionCompositeList.get(21)),
                                   ee.Image(collectionCompositeList.get(37)),
                                   ee.Image(collectionCompositeList.get(49))]).mosaic().clip(miRegionNavarra);
                                   
  Map.addLayer(imagen,{bands:['B4','B3','B2'], min:0, max:6000},'imagen navarra ');
  return imagen.toFloat();
}

//FUNCION PARA FILTRAR LOS CULTIVOS LEÑOSOS.

function claseCultivosLeñosos(){
  var cultivosLeñosos = navarraCultivos2021.filter(ee.Filter.eq('GRUPO','Cultivos le�osos secano')).merge( navarraCultivos2021.filter(ee.Filter.eq('GRUPO','Cultivos le�osos regad�o')));
  
  Map.addLayer(cultivosLeñosos, {color: 'FF0000'}, "Navarra Cultivos Leñosos");
  
  var cultivosLeñososImg = cultivosLeñosos
  .filter(ee.Filter.notNull(['CGRUPO']))
  .reduceToImage({
     properties: ['CGRUPO'],
      reducer: ee.Reducer.first()
  });
  
  return cultivosLeñososImg.toFloat();
}

//FUNCION PARA FILTRAR LA CLASE CULTIVOS HERBACEOS.

function claseCultivosHerbaceos(){
  var cultivosHerbaceos = navarraCultivos2021.filter(ee.Filter.eq('GRUPO','Cultivos herb�ceos regad�o')).merge( navarraCultivos2021.filter(ee.Filter.eq('GRUPO','Cultivos herb�ceos secano')));
  
  Map.addLayer(cultivosHerbaceos, {color: '00FF00'}, "Navarra Cultivos Herbaceos");
  
  var cultivosHerbaceosImg = cultivosHerbaceos
    .filter(ee.Filter.notNull(['CGRUPO']))
    .reduceToImage({
      properties: ['CGRUPO'],
      reducer: ee.Reducer.first()
  });
  
  return cultivosHerbaceosImg.toFloat();
}

//FUNCION PARA FILTRAR LA CLASE CULTIVOS.

function claseCultivos(){
  var cultivos = navarraCultivos2021.filter(ee.Filter.eq('GRUPO','Cultivos le�osos secano')).merge( navarraCultivos2021.filter(ee.Filter.eq('GRUPO','Cultivos le�osos regad�o')));
  cultivos = cultivos.merge(navarraCultivos2021.filter(ee.Filter.eq('GRUPO','Cultivos herb�ceos regad�o')));
  cultivos = cultivos.merge(navarraCultivos2021.filter(ee.Filter.eq('GRUPO','Cultivos herb�ceos secano')));
  
  Map.addLayer(cultivos, {color:'FF0000'}, "Navarra Cultivos");
  
  var cultivosImg = cultivos
    .filter(ee.Filter.notNull(['CGRUPO']))
    .reduceToImage({
      properties: ['CGRUPO'],
      reducer: ee.Reducer.first()
    });
  
  return cultivosImg.toFloat();
}

//FUNCION PARA FILTRAL LA CLASE FORESTAL NO ARBOLADO.

function claseForestalNoArbolado(){
  var forestalNoArbolado = navarraCultivos2021.filter(ee.Filter.eq('GRUPO','Forestal no arbolado'));

  Map.addLayer(forestalNoArbolado, {color: '0000FF'}, "Forestal no arbolado");
  
  var forestalNoArboladoImg = forestalNoArbolado
    .filter(ee.Filter.notNull(['CGRUPO']))
    .reduceToImage({
      properties: ['CGRUPO'],
      reducer: ee.Reducer.first()
  });
  
  return forestalNoArboladoImg.toFloat();
}

//FUNCION PARA FILTRAR LA CLASE FORESTAL ARBOLADO

function claseForestalArbolado(){
  var forestalArbolado = navarraCultivos2021.filter(ee.Filter.eq('GRUPO','Frondosas')).merge(navarraCultivos2021.filter(ee.Filter.eq('GRUPO','Con�feras')));
  forestalArbolado = forestalArbolado.merge(navarraCultivos2021.filter(ee.Filter.eq('GRUPO','Con�feras/Frondosas')));
  
  Map.addLayer(forestalArbolado, {color: 'FBF404'},"Forestal arbolado")
  
  var forestalArboladoImg = forestalArbolado
    .filter(ee.Filter.notNull(['CGRUPO']))
    .reduceToImage({
      properties: ['CGRUPO'],
      reducer: ee.Reducer.first()
  });
  
  return forestalArboladoImg.toFloat();
}

//FUNCION PARA FILTRAR LA CLASE IMPRODUCTIVO

function claseImproductivo(){
  var improductivo = navarraCultivos2021.filter(ee.Filter.eq('GRUPO','Improductivo'));

  Map.addLayer(improductivo, {color:'654E92'}, "Improductivo");
  
  var improductivoImg = improductivo
    .filter(ee.Filter.notNull(['CGRUPO']))
    .reduceToImage({
      properties: ['CGRUPO'],
      reducer: ee.Reducer.first()
  });
  
  return improductivoImg.toFloat();
}

//FUNCION PARA FILTRAR LA CLASE AGUA

function claseAgua(){
  var agua = navarraCultivos2021.filter(ee.Filter.eq('COBERTURAP','CURSOS DE AGUA')).merge(navarraCultivos2021.filter(ee.Filter.eq('COBERTURAP','EMBALSES')));
  agua = agua.merge(navarraCultivos2021.filter(ee.Filter.eq('COBERTURAP','LAGOS Y LAGUNAS')));
  agua = agua.merge(navarraCultivos2021.filter(ee.Filter.eq('COBERTURAP','BALSAS DE RIEGO')));
  agua = agua.merge(navarraCultivos2021.filter(ee.Filter.eq('COBERTURAP','LAMINA DE AGUA ARTIFICIAL')));
  
  Map.addLayer(agua,{color:'0000FF'}, "Agua");
  
  var aguaImg = agua
    .filter(ee.Filter.notNull(['CGRUPO']))
    .reduceToImage({
      properties: ['CGRUPO'],
      reducer: ee.Reducer.first(),
  });
  
  aguaImg = aguaImg.rename(['water']);
  return aguaImg.toFloat();
}

//FUNCION INDICE NDWI
function ndwi_index(imagen){
  var imagenB3 = imagen.select('B3');
  var imagenB8 = imagen.select('B8');
  var ndwi = (imagenB3.subtract(imagenB8)).divide(imagenB3.add(imagenB8));
  

  // Create an NDWI image, define visualization parameters and display.
  var ndwiViz = {min: 0.3, max: 1, palette: ['00FFFF', '0000FF']};
  
  Map.addLayer(ndwi, ndwiViz, 'NDWI1', false)
  
  // Mask the non-watery parts of the image, where NDWI < 0.4.
  var ndwiMasked = ndwi.updateMask(ndwi.gte(0.2));
  Map.addLayer(ndwiMasked, ndwiViz, 'NDWI masked');
  ndwiMasked = ndwiMasked.rename(['water']);
  return ndwiMasked.toFloat();
}

//UNIR INDICE NDWI CON IMAGEN DE PARCELAS DEL AGUA

function mergeWater(aguaNavarre, ndwiNavarre){
  var imagenAgua = ee.ImageCollection([aguaNavarre,ndwiNavarre]).mosaic().clip(miRegionNavarra);
  Map.addLayer(imagenAgua, {bands:['water'], min:0, max:6000},'imagen agua mixed.');
  return imagenAgua.toFloat();
}

function saveImagesRegion(region){
  var geometry;
  if(region.toUpperCase() == "CUENCAPAMPLONA"){
    geometry = cuencaPamplona;
  }
  else if(region.toUpperCase() == "ZONAMEDIA"){
    geometry = zonaMedia;
  }
  else if(region.toUpperCase() == "RIBERA"){
    geometry = ribera;
  }
  else if(region.toUpperCase() == "PIRINEOS"){
    geometry = pirineos;
  }
  else if(region.toUpperCase() == "URBASAYANDIA"){
    geometry = urbasayandia;
  }
  else if(region.toUpperCase() == "BAZTANYULTZAMA"){
    geometry = baztanyultzama;
  }
  else {
    return
  }
  //---------------------------------------------------------
  Export.image.toDrive({
    image: navarra,
    description: region.toUpperCase()+'collectionComposite', // task name to be shown in the Tasks tab
    fileNamePrefix: region.toUpperCase()+'collectionComposite', // filename to be saved in the Google Drive
    scale: 10, // the spatial resolution of the image
    region: geometry,
  });
  //-----------------------------------------------------------------------
  //------------------------ CULTIVOS ------------------------------
  Export.image.toDrive({
    image: cultivosNavarra,
    description: region.toUpperCase()+'Cultivos', // task name to be shown in the Tasks tab
    fileNamePrefix: region.toUpperCase()+'Cultivos', // filename to be saved in the Google Drive
    scale: 10, // the spatial resolution of the image
    region: geometry,
  });
  //-----------------------------------------------------------------------
  //------------------------ FORESTAL NO ARBOLADO ------------------------------
  Export.image.toDrive({
    image: forestalNoArboladoNavarra,
    description: region.toUpperCase()+'forestalNoArbolado', // task name to be shown in the Tasks tab
    fileNamePrefix: region.toUpperCase()+'forestalNoArbolado', // filename to be saved in the Google Drive
    scale: 10, // the spatial resolution of the image
    region: geometry,
  });
  //-----------------------------------------------------------------------
  //------------------------ FORESTAL ARBOLADO ------------------------------
  Export.image.toDrive({
    image: forestalArboladoNavarra,
    description: region.toUpperCase()+'forestalArbolado', // task name to be shown in the Tasks tab
    fileNamePrefix: region.toUpperCase()+'forestalArbolado', // filename to be saved in the Google Drive
    scale: 10, // the spatial resolution of the image
    region: geometry,
  });
  //-----------------------------------------------------------------------
  //------------------------ IMPRODUTIVO ------------------------------
  Export.image.toDrive({
    image: improductivoNavarra,
    description: region.toUpperCase()+'improductivo', // task name to be shown in the Tasks tab
    fileNamePrefix: region.toUpperCase()+'improductivo', // filename to be saved in the Google Drive
    scale: 10, // the spatial resolution of the image
    region: geometry,
  });
  //-----------------------------------------------------------------------
  //------------------------ CULTIVOS HERBACEOS ------------------------------
  Export.image.toDrive({
    image: aguaMergeNavarra,
    description: region.toUpperCase()+'agua', // task name to be shown in the Tasks tab
    fileNamePrefix: region.toUpperCase()+'agua', // filename to be saved in the Google Drive
    scale: 10, // the spatial resolution of the image
    region: geometry,
  });
  //-----------------------------------------------------------------------
}

//obtengo imagen de navarra 
var navarra = imagenNavarra();

//obtengo imagen de la clase de cultivos 
var cultivosNavarra = claseCultivos();

//obtengo imagen clase cultivos herbaceos
//var cultivosHerbaceosNavarra = claseCultivosHerbaceos();

//obtengo imagen clase forestal no arbolado.
var forestalNoArboladoNavarra = claseForestalNoArbolado();

//obtengo imagen clase forestal arbolado
var forestalArboladoNavarra = claseForestalArbolado();

//obtengo imagen clase improductivo
var improductivoNavarra = claseImproductivo();

//obtengo clase agua.
var aguaNavarra = claseAgua();
var ndwiNavarra = ndwi_index(navarra);

var aguaMergeNavarra = mergeWater(aguaNavarra,ndwiNavarra);


saveImagesRegion("cuencaPamplona");
saveImagesRegion("ZOnaMedia");
saveImagesRegion("ribera");
saveImagesRegion("PirIneos");
saveImagesRegion("urbasayandia");
saveImagesRegion("baztanyultzama");