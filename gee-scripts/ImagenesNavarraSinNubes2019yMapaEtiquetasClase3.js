//---------------------------- IMPORTS --------------------------------------------
var baztanRegion = 
    /* color: #d63000 */
    /* shown: false */
    /* displayProperties: [
      {
        "type": "rectangle"
      }
    ] */
    ee.Geometry.Polygon(
        [[[-1.7897177119648089, 43.23312513085748],
          [-1.7897177119648089, 43.02264907855529],
          [-1.4738607783710589, 43.02264907855529],
          [-1.4738607783710589, 43.23312513085748]]], null, false),
    navarraCultivos = ee.FeatureCollection("projects/ee-lukepastor-test/assets/MapaCultivosNavarra2019"),
    miRegion = ee.FeatureCollection("projects/ee-lukepastor-test/assets/navarre"),
    dataset = ee.ImageCollection("COPERNICUS/S2"),
    cuencaDePamplonaRegion = 
    /* color: #d63000 */
    /* shown: false */
    /* displayProperties: [
      {
        "type": "rectangle"
      },
      {
        "type": "marker"
      }
    ] */
    ee.Geometry({
      "type": "GeometryCollection",
      "geometries": [
        {
          "type": "Polygon",
          "coordinates": [
            [
              [
                -2.224743865519463,
                42.921181863356516
              ],
              [
                -2.224743865519463,
                42.620769841468295
              ],
              [
                -1.181042693644463,
                42.620769841468295
              ],
              [
                -1.181042693644463,
                42.921181863356516
              ]
            ]
          ],
          "geodesic": false,
          "evenOdd": true
        },
        {
          "type": "Point",
          "coordinates": [
            -1.5551160246308982,
            43.17015790643073
          ]
        }
      ],
      "coordinates": []
    }),
    pirineosRegion = 
    /* color: #d63000 */
    /* displayProperties: [
      {
        "type": "rectangle"
      }
    ] */
    ee.Geometry.Polygon(
        [[[-1.318371795206963, 42.94942614979765],
          [-1.318371795206963, 42.77024270755683],
          [-0.8651857600507129, 42.77024270755683],
          [-0.8651857600507129, 42.94942614979765]]], null, false),
    RiberaRegion = 
    /* color: #98ff00 */
    /* shown: false */
    /* displayProperties: [
      {
        "type": "rectangle"
      }
    ] */
    ee.Geometry.Polygon(
        [[[-1.749338408138379, 42.12002325864109],
          [-1.749338408138379, 41.98389066861829],
          [-1.389536162044629, 41.98389066861829],
          [-1.389536162044629, 42.12002325864109]]], null, false),
    UltzamaRegion = 
    /* color: #ffc82d */
    /* shown: false */
    /* displayProperties: [
      {
        "type": "rectangle"
      }
    ] */
    ee.Geometry.Polygon(
        [[[-1.8972139582929004, 43.15208242638603],
          [-1.8972139582929004, 42.98454082535998],
          [-1.4824800715741504, 42.98454082535998],
          [-1.4824800715741504, 43.15208242638603]]], null, false),
    ZonaMediaRegion = 
    /* color: #d63000 */
    /* shown: false */
    /* displayProperties: [
      {
        "type": "rectangle"
      }
    ] */
    ee.Geometry.Polygon(
        [[[-2.2794750615399484, 42.71418508661029],
          [-2.2794750615399484, 42.456351477904754],
          [-1.3016918584149484, 42.456351477904754],
          [-1.3016918584149484, 42.71418508661029]]], null, false);


Map.centerObject(miRegion)

function imageNavarra(){
  var collectionCompositeList = dataset
    .filterDate('2019-06-05', '2019-09-20')
    .filterBounds(miRegion)  // Intersecting ROI;
    .filterMetadata('CLOUDY_PIXEL_PERCENTAGE','less_than',1)
    .sort('CLOUDY_PIXEL_PERCENTAGE',true)
    .toList(4,6)
    
  print(collectionCompositeList);
  
  var imagen = ee.ImageCollection([ee.Image(collectionCompositeList.get(0)),  ee.Image(collectionCompositeList.get(1)),ee.Image(collectionCompositeList.get(2)), ee.Image(collectionCompositeList.get(3)) ]).mosaic().clip(miRegion);
  
  print(imagen)
  
  Map.addLayer(imagen,{bands:['B4','B3','B2'], min:0, max:6000},'imagen navarra.');
  
  return imagen.toFloat()
}

//FUNCION PARA FILTRAR LA CLASE CULTIVOS LEÑOSOS

function claseCultivos(){
  var cultivos = navarraCultivos.filter(ee.Filter.eq('GRUPO','Cultivos le�osos secano')).merge( navarraCultivos.filter(ee.Filter.eq('GRUPO','Cultivos le�osos secano regad�o')));
  cultivos = cultivos.merge(navarraCultivos.filter(ee.Filter.eq('GRUPO','Cultivos herb�ceos secano regad�o')));
  cultivos = cultivos.merge(navarraCultivos.filter(ee.Filter.eq('GRUPO','Cultivos herb�ceos secano')));
  Map.addLayer(cultivos, {color: 'FF0000'}, "Navarra Cultivos");
  
  cultivos = cultivos
    .filter(ee.Filter.notNull(['CGRUPO']))
    .reduceToImage({
      properties: ['CGRUPO'],
      reducer: ee.Reducer.first()
  });

  return cultivos.toFloat();
}

//FUNCION PARA FILTRAR LA CLASE CULTIVOS HERBACEOS
/*
function claseCultivosHerbaceos(){
  var cultivosHerbaceos = navarraCultivos.filter(ee.Filter.eq('GRUPO','Cultivos herb�ceos secano regad�o')).merge( navarraCultivos.filter(ee.Filter.eq('GRUPO','Cultivos herb�ceos secano')));
  
  Map.addLayer(cultivosHerbaceos, {color: '00FF00'}, "Navarra Cultivos Herbaceos");
  
  cultivosHerbaceos = cultivosHerbaceos
    .filter(ee.Filter.notNull(['CGRUPO']))
    .reduceToImage({
      properties: ['CGRUPO'],
      reducer: ee.Reducer.first()
  });
  
  return cultivosHerbaceos.toFloat();
}*/

//FUNCION PARA FILTRAL LA CLASE FORESTAL NO ARBOLADO

function claseForestalNoArbolado(){
  var forestalNoArbolado = navarraCultivos.filter(ee.Filter.eq('GRUPO','Forestal no arbolado'));

  Map.addLayer(forestalNoArbolado, {color: '0000FF'}, "Forestal no arbolado");
  
  forestalNoArbolado = forestalNoArbolado
    .filter(ee.Filter.notNull(['CGRUPO']))
    .reduceToImage({
      properties: ['CGRUPO'],
      reducer: ee.Reducer.first()
  });
  
  return forestalNoArbolado.toFloat();
}

//FUNCION PARA FILTRAR LA CLASE FORESTAL ARBOLADO

function claseForestalArbolado(){
  var forestalArbolado = navarraCultivos.filter(ee.Filter.eq('GRUPO','Frondosas')).merge(navarraCultivos.filter(ee.Filter.eq('GRUPO','Con�feras')));
  forestalArbolado = forestalArbolado.merge(navarraCultivos.filter(ee.Filter.eq('GRUPO','Con�feras/Frondosas')));
  
  Map.addLayer(forestalArbolado, {color: 'FBF404'},"Forestal arbolado")
  
  forestalArbolado = forestalArbolado
    .filter(ee.Filter.notNull(['CGRUPO']))
    .reduceToImage({
      properties: ['CGRUPO'],
      reducer: ee.Reducer.first()
  });
  
  return forestalArbolado.toFloat();
}

//FUNCION PARA FILTRAR LA CLASE IMPRODUCTIVO

function claseImproductivo(){
  var improductivo = navarraCultivos.filter(ee.Filter.eq('GRUPO','Improductivo'));

  Map.addLayer(improductivo, {color:'654E92'}, "Improductivo");
  
  improductivo = improductivo
    .filter(ee.Filter.notNull(['CGRUPO']))
    .reduceToImage({
      properties: ['CGRUPO'],
      reducer: ee.Reducer.first()
  });
  
  return improductivo.toFloat();
}

//FUNCION PARA FILTRAR LA CLASE AGUA
function claseAgua(){
  var agua = navarraCultivos.filter(ee.Filter.eq('COBERTURAP','CURSOS DE AGUA')).merge(navarraCultivos.filter(ee.Filter.eq('COBERTURAP','EMBALSES')));
  agua = agua.merge(navarraCultivos.filter(ee.Filter.eq('COBERTURAP','LAGOS Y LAGUNAS')));
  agua = agua.merge(navarraCultivos.filter(ee.Filter.eq('COBERTURAP','BALSAS DE RIEGO')));
  agua = agua.merge(navarraCultivos.filter(ee.Filter.eq('COBERTURAP','LAMINA DE AGUA ARTIFICIAL')));
  Map.addLayer(agua, {color:'0000FF'}, "Agua");
  
  agua = agua
    .filter(ee.Filter.notNull(['CGRUPO']))
    .reduceToImage({
      properties: ['CGRUPO'],
      reducer: ee.Reducer.first(),
  });

  Map.addLayer(agua, {bands:['first'], min:0, max:6000},'imagen agua.');
  
  agua = agua.rename(['water']);
  return agua.toFloat();
}

//INDICE NWDI
function ndwi_index(imagen){
    var imagenB3 = imagen.select('B3');
    var imagenB8 = imagen.select('B8');
    var ndwi = (imagenB3.subtract(imagenB8)).divide(imagenB3.add(imagenB8));
    

    // Create an NDWI image, define visualization parameters and display.
    var ndwiViz = {min: 0.3, max: 1, palette: ['00FFFF', '0000FF']};
    
    Map.addLayer(ndwi, ndwiViz, 'NDWI1', false)
    
    // Mask the non-watery parts of the image, where NDWI < 0.2.
    var ndwiMasked = ndwi.updateMask(ndwi.gte(0.2));
    Map.addLayer(ndwiMasked, ndwiViz, 'NDWI masked');
    ndwiMasked = ndwiMasked.rename(['water']);
    return ndwiMasked.toFloat();
}

//UNIR INDICE NDWI CON IMAGEN PARCELAS DE AGUA.
function mergeWater(aguaNavarre, ndwiNavarre){
  var imagenAgua = ee.ImageCollection([aguaNavarre,ndwiNavarre]).mosaic().clip(miRegion);
  Map.addLayer(imagenAgua, {bands:['water'], min:0, max:6000},'imagen agua mixed.');
  return imagenAgua.toFloat();
}

function saveImagesRegion(region){
  var geometry;
  if(region.toUpperCase() == 'BAZTAN'){
    geometry = baztanRegion;
  }
  else if(region.toUpperCase() == "CUENCADEPAMPLONA"){
    geometry = cuencaDePamplonaRegion.geometries().get(1);
  }
  else if(region.toUpperCase() == "PIRINEOS"){
    geometry = pirineosRegion;
  }
  else if(region.toUpperCase() == "RIBERA"){
    geometry = RiberaRegion;
  }
  else if(region.toUpperCase() == "ULTZAMA"){
    geometry = UltzamaRegion;
  }
  else if(region.toUpperCase() == "ZONAMEDIA"){
    geometry = ZonaMediaRegion;
  }
  
  //---------------------------------------------------------
  Export.image.toDrive({
    image: navarra,
    description: region+'collectionComposite', // task name to be shown in the Tasks tab
    fileNamePrefix: region+'collectionComposite', // filename to be saved in the Google Drive
    scale: 10, // the spatial resolution of the image
    region: geometry,
  });
  //-----------------------------------------------------------------------
  //------------------------ CULTIVOS ------------------------------
  Export.image.toDrive({
    image: cultivosNavarra,
    description: region+'Cultivos', // task name to be shown in the Tasks tab
    fileNamePrefix: region+'Cultivos', // filename to be saved in the Google Drive
    scale: 10, // the spatial resolution of the image
    region: geometry,
  });
  //-----------------------------------------------------------------------
  //------------------------ FORESTAL NO ARBOLADO ------------------------------
  Export.image.toDrive({
    image: forestalNoArboladoNavarra,
    description: region+'forestalNoArbolado', // task name to be shown in the Tasks tab
    fileNamePrefix: region+'forestalNoArbolado', // filename to be saved in the Google Drive
    scale: 10, // the spatial resolution of the image
    region: geometry,
  });
  //-----------------------------------------------------------------------
  //------------------------ FORESTAL ARBOLADO ------------------------------
  Export.image.toDrive({
    image: forestalArboladoNavarra,
    description: region+'forestalArbolado', // task name to be shown in the Tasks tab
    fileNamePrefix: region+'forestalArbolado', // filename to be saved in the Google Drive
    scale: 10, // the spatial resolution of the image
    region: geometry,
  });
  //-----------------------------------------------------------------------
  //------------------------ IMPRODUTIVO ------------------------------
  Export.image.toDrive({
    image: improductivoNavarra,
    description: region+'improductivo', // task name to be shown in the Tasks tab
    fileNamePrefix: region+'improductivo', // filename to be saved in the Google Drive
    scale: 10, // the spatial resolution of the image
    region: geometry,
  });
  //-----------------------------------------------------------------------
    //------------------------ AGUA ------------------------------
  Export.image.toDrive({
    image: aguaMergeNavarra,
    description: region+'agua', // task name to be shown in the Tasks tab
    fileNamePrefix: region+'agua', // filename to be saved in the Google Drive
    scale: 10, // the spatial resolution of the image
    region: geometry,
  });
  //-----------------------------------------------------------------------
}

//obtengo imagen de navarra
var navarra = imageNavarra();

//obtengo clase cultivos.
var cultivosNavarra = claseCultivos();
//print(cultivosLeñososNavarra);

//obtengo clase cultivos herbaceos.
//var cultivosHerbaceosNavarra = claseCultivosHerbaceos();

//obtengo clase forestal no arbolado.
var forestalNoArboladoNavarra = claseForestalNoArbolado();

//obtengo clase forestal arbolado.
var forestalArboladoNavarra = claseForestalArbolado();

//obtengo clave improductivo.
var improductivoNavarra = claseImproductivo();

var aguaNavarra = claseAgua();
var ndwiNavarra = ndwi_index(navarra);

var aguaMergeNavarra = mergeWater(aguaNavarra,ndwiNavarra);
saveImagesRegion("Baztan");
saveImagesRegion("CUENCADEPAMPLONA");
saveImagesRegion("PIRINEOS");
saveImagesRegion("RIBERA");
saveImagesRegion("ULTZAMA");
saveImagesRegion("ZONAMEDIA");