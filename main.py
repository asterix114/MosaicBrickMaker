from PIL import Image
import numpy as np

def find_closest_colors_vectorized(pixels, color_rgbs, color_names):
    """Find closest colors for all pixels at once using vectorized operations"""
    # Reshape pixels to 2D array (height*width, 3)
    pixels_flat = pixels.reshape(-1, 3)
    
    # Calculate squared Euclidean distances (no sqrt needed for comparison)
    # Using broadcasting: (h*w, 1, 3) - (1, n_colors, 3) -> (h*w, n_colors)
    distances = np.sum((pixels_flat[:, np.newaxis, :] - color_rgbs[np.newaxis, :, :]) ** 2, axis=2)
    
    # Find closest color indices for all pixels
    min_indices = np.argmin(distances, axis=1)
    
    # Get closest colors and names
    closest_colors = color_rgbs[min_indices]
    closest_names = [color_names[i] for i in min_indices]
    
    # Reshape back to image dimensions
    result_colors = closest_colors.reshape(pixels.shape)
    color_names_matrix = np.array(closest_names).reshape(pixels.shape[0], pixels.shape[1])
    
    return result_colors, color_names_matrix, min_indices

# Load the image
puffin_image = Image.open('C:/Users/ossip/Documents/projekter/DIYMosaicMaker[DMM]/Screenshot_20_crooped.png')

puffin_image_150px = puffin_image.copy()
puffin_image_150px.thumbnail((150, 150))

# Scale the image back up to its original size
puffin_image = puffin_image_150px.resize((1361, 1361), resample=Image.Resampling.NEAREST)

# Define the colors as RGB tuples
colors = {
    "RoundTile": {
        "Bright Red": (180, 0, 0),
        "Nougat": (187, 128, 90),
        "Dark Orange": (145, 80, 28),
        "Light Nougat": (225, 190, 161),
        "Reddish Brown": (95, 49, 9),
        "Medium Nougat": (170, 125, 85),
        "Bright Orange": (214, 121, 35),
        "Dark Brown": (55, 33, 0),
        "Flame Yellowish Orange": (252, 172, 0),
        "Sand Yellow": (137, 125, 98),
        "Brick Yellow": (204, 185, 141),
        "Warm Gold": (185, 149, 59),
        "Bright Yellow": (250, 200, 10),
        "Cool Yellow": (255, 236, 108),
        "Olive Green": (119, 119, 78),
        "Vibrant Yellow": (255, 255, 0),
        "Bright Yellowish Green": (165, 202, 24),
        "Spring Yellowish Green": (226, 249, 154),
        "Bright Green": (88, 171, 65),
        "Dark Green": (0, 133, 43),
        "Aqua": (211, 242, 234),
        "Bright Bluish Green": (0, 152, 148),
        "Medium Azur": (104, 195, 226),
        "Dark Azur": (70, 155, 195),
        "Black": (27, 42, 52),
        "Bright Blue": (30, 90, 168),
        "Sand Blue": (112, 129, 154),
        "Earth Blue": (25, 50, 90),
        "Medium Lilac": (68, 26, 145),
        "Medium Lavender": (160, 110, 185),
        "Lavender": (205, 164, 222),
        "Bright Reddish Violet": (144, 31, 118),
        "Bright Purple": (200, 80, 155),
        "Light Purple": (255, 158, 205),
        "New Dark Red": (114, 0, 18),
        "Vibrant Coral": (240, 109, 120),
        "White": (244, 244, 244),
        "Medium Stone Grey": (150, 150, 150),
        "Dark Stone Grey": (100, 100, 100)
    }
}

# Combine all colors into a single list of RGB tuples and get their names
color_list = []
color_name_list = []

for category_name, category_colors in colors.items():
    for color_name, rgb in category_colors.items():
        color_list.append(rgb)
        color_name_list.append(f"{category_name} - {color_name}")

# Convert to numpy arrays for faster computation
color_rgbs = np.array(color_list)
color_names = color_name_list

# Create a thumbnail copy
puffin_image_thumbnail = puffin_image.copy()

# Ensure the image is in RGB mode
if puffin_image_thumbnail.mode != 'RGB':
    puffin_image_thumbnail = puffin_image_thumbnail.convert('RGB')

# Convert image to numpy array
image_array = np.array(puffin_image_thumbnail)
height, width, channels = image_array.shape

print(f"Processing image of size {width} x {height} pixels...")
print(f"Using {len(color_list)} available colors")

# Process all pixels at once using vectorized operations
print("Processing pixels with vectorized operations...")
result_colors, color_names_matrix, min_indices = find_closest_colors_vectorized(
    image_array, color_rgbs, color_names
)

# Create the modified image from the result array
puffin_image_thumbnail = Image.fromarray(result_colors.astype(np.uint8))

# Calculate color usage statistics
unique_indices, counts = np.unique(min_indices, return_counts=True)
color_usage = {color_names[i]: count for i, count in zip(unique_indices, counts)}

# Show the modified image
print("Showing modified image...")
puffin_image_thumbnail.show()

# Create and show thumbnail version
print("Creating thumbnail...")
thumbnail_copy = puffin_image_thumbnail.copy()
thumbnail_copy.thumbnail((32, 32))
thumbnail_copy.show()

# Print color usage statistics
print("\nColor usage statistics (39 used colors percentage):")
sorted_colors = sorted(color_usage.items(), key=lambda x: x[1], reverse=True)
for color_name, count in sorted_colors[:39]:
    percentage = (count / (width * height)) * 100
    print(f"{color_name}: {count} pixels ({percentage:.2f}%)")
