from PIL import Image, ImageDraw, ImageOps
import numpy as np

def find_closest_colors_vectorized(pixels, color_rgbs, color_names):
    "Find closest colors for all pixels at once using vectorized operations"
    # Reshape pixels to 2D array (height*width, 3)
    pixels_flat = pixels.reshape(-1, 3)
    
    # Calculate squared Euclidean distances (no sqrt needed for comparison)
    # Using broadcasting: (h*w, 1, 3) - (1, n_colors, 3) -> (h*w, n_colors)
    distances = np.sum((pixels_flat[:, np.newaxis, :] - color_rgbs[np.newaxis, :, :]) ** 2, axis=2)
    
    # Find closest color indices for all pixels
    min_indices = np.argmin(distances, axis=1)
    
    # Get closest colors (skip names/matrix since unused)
    closest_colors = color_rgbs[min_indices]
    
    # Reshape back to image dimensions
    result_colors = closest_colors.reshape(pixels.shape)
    
    # Return None for unused color_names_matrix to match original signature
    return result_colors, None, min_indices

# Load the image
puffin_image = Image.open('C:/Users/ossip/Documents/projekter/DIYMosaicMaker[DMM]/Screenshot_20_crooped.png')

thumbnail_x = 128 # Width
thumbnail_y = 128 # Height

value = (puffin_image.width * 0.055) # experiment with this number if you have a lower pixel count like 128 x 128 or else it would be to pixelated
puffin_image

puffin_image_150px = puffin_image.copy()
puffin_image_150px.thumbnail((value, value))

valueW = (puffin_image.width // 2) # // 2 is the same as doing * 0.5 but without the .0 at the end
valueH = (puffin_image.height // 2)
puffin_image = puffin_image_150px.resize((valueW, valueH), resample=Image.Resampling.NEAREST)
puffin_image_150px

# Define the colors as RGB tuples and comment out the colors you don't want
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
        color_name_list.append(f"{color_name}")

# Convert to numpy arrays for faster computation
color_rgbs = np.asarray(color_list)
color_names = color_name_list

# Create a thumbnail copy
puffin_image_thumbnail = puffin_image.copy()

# Ensure the image is in RGB mode
if puffin_image_thumbnail.mode != 'RGB':
    puffin_image_thumbnail = puffin_image_thumbnail.convert('RGB')

# Convert image to numpy array
image_array = np.array(puffin_image_thumbnail)
height, width, channels = image_array.shape

print(f"Using {len(color_list)} available colors")

# Process all pixels at once using vectorized operations
result_colors, _, min_indices = find_closest_colors_vectorized(
    image_array, color_rgbs, color_names
)

# Create the modified image from the result array
puffin_image_thumbnail = Image.fromarray(result_colors.astype(np.uint8))

# Calculate color usage statistics for the main image
unique_indices, counts = np.unique(min_indices, return_counts=True)
color_usage_main = {color_names[i]: count for i, count in zip(unique_indices, counts)}

# Show the modified image
puffin_image_thumbnail.show()
puffin_image_thumbnail

# Create the thumbnail
thumbnail = puffin_image_thumbnail.copy()
thumbnail.thumbnail((thumbnail_x, thumbnail_y))
thumbnail.show()


# Convert the thumbnail to numpy array and process it
thumbnail_array = np.array(thumbnail)
print(f"Processing thumbnail of size {thumbnail_array.shape[1]} x {thumbnail_array.shape[0]} pixels")

# Process the thumbnail to map its colors
result_colors_32, _, min_indices_32 = find_closest_colors_vectorized(
    thumbnail_array, color_rgbs, color_names
)

# Calculate color usage statistics for the 32x32 thumbnail
unique_indices_32, counts_32 = np.unique(min_indices_32, return_counts=True)
color_usage_32x32 = {color_names[i]: count for i, count in zip(unique_indices_32, counts_32)}

# Prepare the output string
output_str = ""

# Add the category name once
output_str += f"{category_name}\n"  # Print the category name

# Add color usage statistics for the 32x32 thumbnail
sorted_colors_32 = sorted(color_usage_32x32.items(), key=lambda x: x[1], reverse=True)
total_pixels_32 = thumbnail_array.shape[0] * thumbnail_array.shape[1]

for color_name, count in sorted_colors_32[:39]:  # 39 as in the 39 colors that are in the list
    percentage = (count / total_pixels_32) * 100
    output_str += f"{color_name} {count} ({percentage:.2f}%)\n"

# Save the output string to a .txt file
with open('C:/Users/ossip/Documents/projekter/DIYMosaicMaker[DMM]/legopartslist.txt', 'w') as f:
    f.write(output_str)


print("Results saved to 'legopartslist.txt'")


# Build manual
# I need to know every single lego 1x1 flat plate and take the x and y pixel value and use it most effinectly 32 x 32 should be x 16 + 16 y 16 +16 and 2 by x and 2 by y
instructions = thumbnail.copy()
x16W = (instructions.width * 16)
x16H = (instructions.height * 16)
print(instructions.width, instructions.height)
instructions4 = instructions.resize((x16W, x16H), resample=Image.Resampling.NEAREST)
spcbetwn_ech_linW = 16
spcbetwn_ech_linH = 16
width, height = x16W, x16H
img1 = ImageDraw.Draw(instructions4) # Object to draw over main image
for i in range(0, width, spcbetwn_ech_linW):
    img1.line([i, 0, i, height], fill="blue", width=1) # vertical   
for j in range(0, height, spcbetwn_ech_linH):
    img1.line([0, j, width, j], fill="red", width=1) # Horizontal

instructions2 = instructions4.resize((valueW, valueH))

instructions3 = ImageOps.expand(instructions2, border=(150, 50, 150, 350), fill=(255, 255, 255))
instructions3.show()

# Hi :D
