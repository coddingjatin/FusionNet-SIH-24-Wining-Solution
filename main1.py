import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
from matplotlib import cm
from matplotlib.colors import Normalize
from PIL import Image

# Step 1: Load the SAR image and DEM data
# Replace these paths with your actual file locations
sar_image_path = 'ROIs1970_fall_s1_8_p585.png'  # Grayscale SAR image

# Generate synthetic DEM data if DEM is unavailable
def generate_synthetic_dem(shape):
    x = np.linspace(0, 10, shape[1])  # X-axis
    y = np.linspace(0, 10, shape[0])  # Y-axis
    x, y = np.meshgrid(x, y)
    return np.sin(x) + np.cos(y)  # Example wavy terrain

try:
    # Load SAR image
    sar_image = np.array(Image.open(sar_image_path).convert("L"))  # Convert to grayscale
    print(f"SAR image loaded with shape: {sar_image.shape}")

    # Synthetic DEM generation
    dem_data = generate_synthetic_dem(sar_image.shape)
    print(f"Generated synthetic DEM with shape: {dem_data.shape}")

except FileNotFoundError as e:
    print(f"Error loading files: {e}")
    exit()

# Step 2: Normalize the SAR image for color mapping
sar_normalized = (sar_image - sar_image.min()) / (sar_image.max() - sar_image.min())

# Step 3: Apply a colormap to the SAR image
cmap = cm.viridis  # You can choose other colormaps like cm.plasma or cm.inferno
norm = Normalize(vmin=0, vmax=1)
colored_sar = cmap(norm(sar_normalized))  # RGB values from the colormap

# Step 4: Create a PyVista Structured Grid for 3D visualization
x, y = np.meshgrid(np.arange(dem_data.shape[1]), np.arange(dem_data.shape[0]))
grid = pv.StructuredGrid(x, y, dem_data)

# Step 5: Convert the colorized SAR image to a texture
texture = pv.numpy_to_texture((colored_sar[:, :, :3] * 255).astype(np.uint8))

# Step 6: Plot the 3D model
plotter = pv.Plotter()
plotter.add_mesh(grid, texture=texture, show_edges=False)
plotter.add_axes()  # Add axes for orientation
plotter.show_grid()
plotter.view_isometric()  # Set isometric view for better visualization

# Show the interactive 3D plot
print("Rendering the 3D model...")
plotter.show()
