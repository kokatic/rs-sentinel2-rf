import os
import rasterio
from rasterio import features
import geopandas as gpd
from shapely.geometry import mapping
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import joblib  # For saving the model

# Define paths
dataset_path = './dataset/avril'
reference_image_path = os.path.join(dataset_path, 'T32SKC_20240430T102021_B02_10m.jp2')
shapefile_path = './features/features.shp'
output_raster_path = './output/rasterized_labels.tif'
model_path = './output/random_forest_model.pkl'

# Get band files
band_files = sorted([f for f in os.listdir(dataset_path) if f.endswith('.jp2')])

# Read shapefile
gdf = gpd.read_file(shapefile_path)

# Define unique values in the type_nom column
unique_values = gdf['type_num'].unique()

# Load reference image to get shape and transform
with rasterio.open(reference_image_path) as src:
    out_shape = src.shape
    transform = src.transform

# Create a dictionary to map unique values to pixel values
type_to_pixel = {value: idx + 1 for idx, value in enumerate(unique_values)}

# Rasterize geometries based on type_nom values
rasterized_labels = features.rasterize(
    ((mapping(geom), type_to_pixel[type_nom]) for geom, type_nom in zip(gdf.geometry, gdf['type_num'])),
    out_shape=out_shape,
    transform=transform,
    fill=0,
    all_touched=True,
    dtype=rasterio.uint8
)

# Write raster to file
with rasterio.open(output_raster_path, 'w', driver='GTiff', width=out_shape[1], height=out_shape[0],
                   count=1, dtype=rasterio.uint8, crs=src.crs, transform=transform) as dst:
    dst.write(rasterized_labels, 1)

print("Rasterization completed. Rasterized labels saved to:", output_raster_path)

# Classification using Random Forest
# Stack all band data for training
bands_data = []

for band_file in band_files:
    with rasterio.open(os.path.join(dataset_path, band_file)) as src:
        bands_data.append(src.read(1).ravel())

bands_data = np.vstack(bands_data).T

# Read rasterized labels
labels = rasterized_labels.ravel()

# Remove unlabeled data for training
mask = labels > 0
X = bands_data[mask]
y = labels[mask]

# Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X, y)

# Save the model
joblib.dump(rf_model, model_path)

print("Random Forest model trained and saved to:", model_path)

# Predict the entire image
predicted_labels = rf_model.predict(bands_data).reshape(out_shape)

# Write the classified raster to file
output_classified_raster_path = './output/classified_image.tif'
with rasterio.open(output_classified_raster_path, 'w', driver='GTiff', width=out_shape[1], height=out_shape[0],
                   count=1, dtype=rasterio.uint8, crs=src.crs, transform=transform) as dst:
    dst.write(predicted_labels, 1)

print("Classification completed. Classified image saved to:", output_classified_raster_path)
