import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal

# Load the DEM dataset (replace with the actual file path)
dem_file = "C:\\Users\\esisy\\Downloads\\LOWE_VALLEY.tif"  # Ensure you have downloaded this
dataset = gdal.Open(dem_file)

# Convert raster to NumPy array
band = dataset.GetRasterBand(1)
elevation = band.ReadAsArray()

# Get geotransform to map pixel to coordinates
gt = dataset.GetGeoTransform()
x_min, x_res, _, y_max, _, y_res = gt
x_max = x_min + (dataset.RasterXSize * x_res)
y_min = y_max + (dataset.RasterYSize * y_res)

# Create coordinate grid
x = np.linspace(x_min, x_max, dataset.RasterXSize)
y = np.linspace(y_min, y_max, dataset.RasterYSize)
X, Y = np.meshgrid(x, y)

# Plot elevation contour map
plt.figure(figsize=(10, 6))
contour = plt.contour(X, Y, elevation, levels=20, cmap="terrain")
plt.colorbar(contour, label="Elevation (meters)")
plt.title("Adirondacks Elevation Contour Map")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()