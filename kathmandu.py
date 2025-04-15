import folium
from sympy import symbols, series
from geopy.distance import geodesic

# Coordinates for the marker and Kathmandu
latitude, longitude = 27.990759677003116, 87.02728875673719
kathmandu_coords = (27.7172, 85.3240)

# Create Folium map centered on the marker
mymap = folium.Map(location=[latitude, longitude], zoom_start=8)

# Add marker with a popup to copy coordinates
marker_popup = f"Coordinates: {latitude}, {longitude}"
folium.Marker(
    [latitude, longitude],
    popup=marker_popup,
    tooltip="Click for coordinates"
).add_to(mymap)

# Draw a line between the two points
folium.PolyLine([kathmandu_coords, [latitude, longitude]], color="blue").add_to(mymap)

# Calculate and display the distance between points
distance = geodesic(kathmandu_coords, (latitude, longitude)).kilometers
folium.Marker(
    kathmandu_coords,
    popup=f"Kathmandu (Distance: {distance:.2f} km)"
).add_to(mymap)

# Taylor series at the point (example for f(x) = sin(x))
x = symbols('x')
taylor_expansion = series(x**2, x, 0, 4)  # Example expansion
print(f"Taylor Series around x=0: {taylor_expansion}")

# Save map to HTML
mymap.save("folium_map.html")
import folium
from sympy import symbols, series, sin, cos
from geopy.distance import geodesic

# Constants
latitude, longitude = 27.990759677003116, 87.02728875673719
kathmandu_coords = (27.7172, 85.3240)
speed_kmh = 1200
speed_ms = speed_kmh / 3.6  # Convert to m/s
g = 9.8  # Acceleration due to gravity (m/s^2)

# Create Folium map
mymap = folium.Map(location=[latitude, longitude], zoom_start=8)

# Add marker with popup for coordinates
marker_popup = f"Coordinates: {latitude}, {longitude}"
folium.Marker(
    [latitude, longitude],
    popup=marker_popup,
    tooltip="Click for coordinates"
).add_to(mymap)

# Draw a line to Kathmandu
folium.PolyLine([kathmandu_coords, [latitude, longitude]], color="blue").add_to(mymap)

# Calculate the distance between points
distance = geodesic(kathmandu_coords, (latitude, longitude)).kilometers
folium.Marker(
    kathmandu_coords,
    popup=f"Kathmandu (Distance: {distance:.2f} km)"
).add_to(mymap)

# Define symbols and equations of motion
t = symbols('t')  # Time
angle = 45  # Launch angle in degrees
angle_rad = angle * (3.141592653589793 / 180)  # Convert to radians
x_t = speed_ms * cos(angle_rad) * t  # Horizontal motion
y_t = speed_ms * sin(angle_rad) * t - (0.5 * g * t**2)  # Vertical motion

# Taylor series expansion at t = 2 seconds
taylor_x = series(x_t, t, 2, 4)  # Expansion up to 4th degree
taylor_y = series(y_t, t, 2, 4)  # Expansion up to 4th degree

# Add Taylor series to the map
popup_content = f"""
Taylor Series Expansion at t=2s:<br>
Horizontal motion (x): {taylor_x}<br>
Vertical motion (y): {taylor_y}
"""
folium.Marker(
    [latitude, longitude],
    popup=popup_content
).add_to(mymap)

# Save the map
mymap.save("folium_map_with_taylor_series.html")
