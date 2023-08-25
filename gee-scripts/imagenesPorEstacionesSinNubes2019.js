var miRegion = ee.FeatureCollection("projects/ee-lukepastor-test/assets/navarre")
var dataset = ee.ImageCollection('COPERNICUS/S2')
Map.centerObject(miRegion)
//We define a function to apply a cloud mask to Sentinel-2 images using the QA60 band 
//which contains information about cloud masks. Then, similarly to the Landsat collections, 
//we will filter our Sentinel-2 collection by our defined parameters (startDate, endDate, 
//country, etc)
// Function to mask clouds S2
function maskS2srClouds(image) {
  var qa = image.select('QA60');

  // Bits 10 and 11 are clouds and cirrus, respectively.
  var cloudBitMask = 1 << 10;
  var cirrusBitMask = 1 << 11;

  // Both flags should be set to zero, indicating clear conditions.
  var mask = qa.bitwiseAnd(cloudBitMask).eq(0)
      .and(qa.bitwiseAnd(cirrusBitMask).eq(0));

  return image.updateMask(mask).divide(10000);
}
// Filter Sentinel-2 collection
//--------------------------- WINTER ----------------------------------
var collectionCompositeWinter = dataset
  .filterDate('2018-12-21', '2019-03-20')
  .filterBounds(miRegion)  // Intersecting ROI;
  .filterMetadata('CLOUDY_PIXEL_PERCENTAGE','less_than',20)
  .map(maskS2srClouds)
  .median()
  .clip(miRegion);

// Add composite to map
Map.addLayer(collectionCompositeWinter,{bands:['B4','B3','B2'],min:0.02,max:0.3,
                          gamma:1.5},'Navarre Winter');
//-----------------------------------------------------------------------
//--------------------------- SPRING ------------------------------------
var collectionCompositeSpring = dataset //cojo dataset
  .filterDate('2019-03-21', '2019-06-22') //toi
  .filterBounds(miRegion) //rtoi
  .filterMetadata('CLOUDY_PIXEL_PERCENTAGE','less_than',20) //filtro por porcentaje de nubes
  .map(maskS2srClouds) //aplico mascara de nubes
  .median()
  .clip(miRegion);

Map.addLayer(collectionCompositeSpring,{bands:['B4','B3','B2'],min:0.02,max:0.3,
                          gamma:1.5},'Navarra Spring');
//-----------------------------------------------------------------------
//--------------------------- SUMMER ------------------------------------
var collectionCompositeSummer = dataset //cojo dataset
  .filterDate('2019-06-22', '2019-09-21') //toi
  .filterBounds(miRegion) //rtoi
  .filterMetadata('CLOUDY_PIXEL_PERCENTAGE','less_than',20) //filtro por porcentaje de nubes
  .map(maskS2srClouds) //aplico mascara de nubes
  .median()
  .clip(miRegion);

Map.addLayer(collectionCompositeSummer,{bands:['B4','B3','B2'],min:0.02,max:0.3,
                          gamma:1.5},'Navarra Summer');
//-----------------------------------------------------------------------
//--------------------------- AUTUMN ------------------------------------
var collectionCompositeAutumn = dataset //cojo dataset
  .filterDate('2019-09-22', '2019-12-21') //toi
  .filterBounds(miRegion) //rtoi
  .filterMetadata('CLOUDY_PIXEL_PERCENTAGE','less_than',20) //filtro por porcentaje de nubes
  .map(maskS2srClouds) //aplico mascara de nubes
  .median()
  .clip(miRegion);

Map.addLayer(collectionCompositeAutumn,{bands:['B4','B3','B2'],min:0.02,max:0.3,
                          gamma:1.5},'Navarra Autumn');
                          

//EXPORTING IMAGES
//------------------------ WINTER ------------------------------
Export.image.toDrive({
  image: collectionCompositeWinter,
  description: 'Navarre in winter', // task name to be shown in the Tasks tab
  fileNamePrefix: 'NavarreWinterSentinel2', // filename to be saved in the Google Drive
  scale: 20, // the spatial resolution of the image
  region: miRegion,
});
//-----------------------------------------------------------------------
//------------------------ SPRING ------------------------------
Export.image.toDrive({
  image: collectionCompositeSpring,
  description: 'Navarre in spring', // task name to be shown in the Tasks tab
  fileNamePrefix: 'NavarreSpringSentinel2', // filename to be saved in the Google Drive
  scale: 20, // the spatial resolution of the image
  region: miRegion,
});
//-----------------------------------------------------------------------
//------------------------ SUMMER ------------------------------
Export.image.toDrive({
  image: collectionCompositeSummer,
  description: 'Navarre in summer', // task name to be shown in the Tasks tab
  fileNamePrefix: 'NavarreSummerSentinel2', // filename to be saved in the Google Drive
  scale: 20, // the spatial resolution of the image
  region: miRegion,
});
//-----------------------------------------------------------------------
//------------------------ AUTUMN ------------------------------
Export.image.toDrive({
  image: collectionCompositeAutumn,
  description: 'Navarre in autumn', // task name to be shown in the Tasks tab
  fileNamePrefix: 'NavarreAutumnSentinel2', // filename to be saved in the Google Drive
  scale: 20, // the spatial resolution of the image
  region: miRegion,
});
//-----------------------------------------------------------------------