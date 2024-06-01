import os
import numpy as np
import geopandas as gpd
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import rasterio
from rasterio.mask import mask

# Define paths
validation_dataset_path = './validation_dataset'
validation_shapefile_path = './features/validation_features.shp'
model_path = './output/random_forest_model.pkl'

# Get validation band files
validation_band_files = sorted([f for f in os.listdir(validation_dataset_path) if f.endswith('.jp2')])

# Read validation shapefile
val_gdf = gpd.read_file(validation_shapefile_path)

# Function to extract pixel values for given geometries from a raster
def extract_pixels(raster_path, geometries):
    with rasterio.open(raster_path) as src:
        out_image, out_transform = mask(src, geometries, crop=True)
        return out_image.reshape(-1)

# Extract pixel values for validation set
val_pixels = np.vstack([extract_pixels(os.path.join(validation_dataset_path, band_file), val_gdf.geometry) 
                        for band_file in validation_band_files]).T

# Extract labels
val_labels = val_gdf['type_num'].values

# Load the trained Random Forest model
rf_model = joblib.load(model_path)

# Predict on validation set
val_predictions = rf_model.predict(val_pixels)

# Calculate accuracy
accuracy = accuracy_score(val_labels, val_predictions)
print(f"Validation Accuracy: {accuracy}")

# Generate classification report
class_report = classification_report(val_labels, val_predictions)
print("Classification Report:\n", class_report)

# Generate confusion matrix
conf_matrix = confusion_matrix(val_labels, val_predictions)
print("Confusion Matrix:\n", conf_matrix)

print("Validation completed.")
