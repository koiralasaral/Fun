# '''Absolutely! To visualize actual high-speed rail tracks in Python, we can use Folium to map railway routes and GeoPandas to process shapefile data if available. Here’s how you can approach it:
# Steps:
# - Obtain rail track data (Shapefile, GeoJSON, or public datasets).
# - Use Folium to plot the railway track on an interactive map.
# - If data allows, analyze curves and track geometry for better insights.
# - Overlay train stations to show key stops.


# Install dependencies:
# Ensure you have these libraries installed:
# pip install folium geopandas pandas shapely matplotlib


# Example Script:'''
import folium
import geopandas as gpd

# Load rail track data (Shapefile or GeoJSON)
railway_data = gpd.read_file("C:\\Users\\LENOVO\\Downloads\\hotosm_chn_railways_lines_geojson\\hotosm_chn_railways_lines_geojson.geojson")  # Replace with actual file path
# Filter railway data to include only lines connecting 10 most populated cities in China
# List of top 10 most populated cities in China (as of recent data)
selected_cities = ["Shanghai", "Beijing", "Chongqing", "Tianjin", "Guangzhou", "Shenzhen", "Chengdu", "Nanjing", "Wuhan", "Xi'an"]

# Inspect the columns in the dataset to find the appropriate column for filtering
print("Columns in the dataset:", railway_data.columns)

# Replace 'city' with the correct column name if available, or skip filtering if not applicable
if 'city' in railway_data.columns:
    filtered_data = railway_data[railway_data['city'].isin(selected_cities)]
else:
    print("Column 'city' not found. Skipping city-based filtering.")
    filtered_data = railway_data

# Extract coordinates for the filtered railway lines
railway_coords = []
for geom in filtered_data.geometry:
    if geom.type == 'LineString':
        railway_coords.extend(geom.coords[:])
    elif geom.type == 'MultiLineString':
        for line in geom:
            railway_coords.extend(line.coords[:])
# Extract coordinates for mapping
railway_coords = railway_data.geometry.iloc[0].coords[:]

# Create Folium map centered at a station
m = folium.Map(location=[railway_coords[0][1], railway_coords[0][0]], zoom_start=6)
# Add markers for selected cities
for city in selected_cities:
    city_data = railway_data[railway_data['city'] == city]
    if not city_data.empty:
        city_coords = city_data.geometry.iloc[0].centroid.coords[0]
        folium.Marker(
            location=[city_coords[1], city_coords[0]],
            popup=city,
            icon=folium.Icon(color="blue", icon="info-sign")
        ).add_to(m)
# Plot railway track
folium.PolyLine([(lat, lon) for lon, lat in railway_coords], color="red", weight=3).add_to(m)

# Save and show map
m.save("railway_map.html")
print("Map saved! Open 'railway_map.html' to view.")
