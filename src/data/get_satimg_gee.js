# Copy and Paste following code in GEE code editor

//feature collection for administrative boundires
var countries = ee.FeatureCollection("FAO/GAUL/2015/level1");

// filter sabah region form adminsitrative boundaries
var sabah = countries.filter(
  ee.Filter.and(
    ee.Filter.eq("ADM0_NAME" , "Malaysia"),
    ee.Filter.eq("ADM1_NAME", "Sabah")
  )
);

// visualize
//Map.centerObject(sabah, 8);
//Map.addLayer(sabah, {"color" : "FF0000"}, "sabah boundary");

// Sentinel2 Images
var sentinel2 = ee.ImageCollection("COPERNICUS/S2")
.filterBounds(sabah)
.filterDate("2015-10-24","2016-10-24")
.filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 10));

// Select RGB Bands
var rgbComposites = sentinel2.select(["B4","B3","B2"]);

// Download to drive
var batch = require("users/fitoprincipe/geetools:batch")
batch.Download.ImageCollection.toDrive(
  rgbComposites, 
  "sabah_sentinel2_imgs",
  { 
    scale : 10,
    region : sabah.getInfo()["coordinates"],
    type : "float"
  }
)

// Access first image to see any metadata in image
//var firstImage = rgbComposites.first()
//print(firstImage)

// Visualize
var visParams = {
  min : 0,
  max : 3000,
  bands : ["B4","B3","B2"]
};
Map.centerObject(sabah, 8);
Map.addLayer(rgbComposites, visParams, "Sentinel2 RGB");

print("Total number of sentinel images : ", sentinel2.size())

