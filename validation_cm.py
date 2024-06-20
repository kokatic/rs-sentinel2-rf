import os
import rasterio
import geopandas as gpd
import numpy as np
from rasterio.features import geometry_mask
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Define paths
classified_raster_path = './output/classified_image.tif'
geojson_path = './features/validation.geojson'  # Corrected path

# Read classified raster
with rasterio.open(classified_raster_path) as src:
    classified_image = src.read(1)
    transform = src.transform

# Read GeoJSON validation data
gdf = gpd.read_file(geojson_path)

# Rasterize the validation polygons to obtain a mask
shape = (classified_image.shape[0], classified_image.shape[1])
validation_mask = geometry_mask(gdf.geometry, transform=transform, out_shape=shape)

# Extract true labels from the validation mask and the classified raster
true_labels = classified_image[validation_mask]
predicted_labels = classified_image[validation_mask]

# Calculate confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels, labels=np.unique(true_labels))

# Define real labels
real_labels = ['corp', 'bati', 'water', 'desert']

# Filter confusion matrix to include only the real labels
conf_matrix_filtered = conf_matrix[:len(real_labels), :len(real_labels)]


# Plot confusion matrix with real labels for both true and predicted labels
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix_filtered, annot=True, fmt='d', cmap='Blues', 
            xticklabels=real_labels, yticklabels=real_labels)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()
