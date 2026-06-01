import numpy as np
from scipy.spatial import KDTree
from matplotlib.colors import XKCD_COLORS

# Define the primary colors dictionary
primary_colors = {    
    "red": [(200, 0, 0), (255, 100, 100)],
    "green": [(0, 200, 0), (100, 255, 100)],
    "blue": [(0, 0, 200), (100, 100, 255)],
    "yellow": [(200, 200, 0), (255, 255, 205)],
    "light yellow": [(200, 236, 205), (255, 255, 255)],
    "black": [(0, 0, 0), (50, 50, 50)],
    "white": [(200, 200, 200), (255, 255, 255)],
    "grey": [(100, 100, 100), (200, 200, 200)],
    "purple": [(100, 0, 120), (225, 200, 225)],
    "orange": [(200, 100, 0), (255, 150, 50)],
    "light red": [(200, 100, 100), (255, 180, 180)],
    "dark red": [(100, 0, 0), (200, 100, 100)],
    "light green": [(100, 255, 100), (150, 255, 150)],
    "dark green": [(0, 100, 0), (100, 200, 100)],
    "light blue": [(100, 100, 255), (150, 150, 255)],
    "dark blue": [(0, 0, 100), (100, 100, 200)],
    "light grey": [(150, 150, 150), (200, 200, 200)],
    "dark grey": [(50, 50, 50), (100, 100, 100)]
}

def closest_xkcd_color(rgb):
    # Convert XKCD colors to RGB values
    xkcd_colors = {name: tuple(int(hex_code[i:i+2], 16) for i in (1, 3, 5)) for name, hex_code in XKCD_COLORS.items()}
    color_names = list(xkcd_colors.keys())
    rgb_values = np.array(list(xkcd_colors.values()))

    # Build a KDTree for fast nearest-neighbor search
    tree = KDTree(rgb_values)
    _, index = tree.query(rgb)
    return color_names[index]

def is_color_in_range(color, color_range):
    return all(
        lower <= c <= upper
        for c, (lower, upper) in zip(color, zip(color_range[0], color_range[1]))
    )

def get_primary_color_name(detected_color):    
    for color_name, color_range in primary_colors.items():
        if is_color_in_range(detected_color, color_range):
            return color_name
    return "Multi"

def grid_sample(image, grid_size=15):
    width, height = image.size
    pixels = image.load()    
    sampled_pixels = []
    sampled_coords = []
    for x in range(0, width, width // grid_size):
        for y in range(0, height, height // grid_size):            
            if image.mode == "RGB":
                r, g, b = pixels[x, y]
            elif image.mode == "RGBA":
                r, g, b, a = pixels[x, y]                            
            if (r, g, b) != (0, 0, 0):
                sampled_pixels.append((r, g, b))
                sampled_coords.append((x, y))
    return np.array(sampled_pixels), sampled_coords

def resize_image_from_center(image, scale_factor=1 / 3):        
    original_width, original_height = image.size
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)
    left = (original_width - new_width) // 2
    upper = (original_height - new_height) // 2    
    right = left + new_width
    lower = upper + new_height
    cropped_image = image.crop((left, upper, right, lower))
    image.close()     
    return cropped_image

def process_image_and_detect_colors(image):    
    image = resize_image_from_center(image)
    grid_pixels, grid_coords = grid_sample(image, grid_size=15)      
    if len(grid_pixels) > 0:
        color_counts = {}
        for pixel in grid_pixels:
            color_name = get_primary_color_name(tuple(pixel))
            if color_name in color_counts:
                color_counts[color_name] += 1
            else:
                color_counts[color_name] = 1
        primary_color_name = max(color_counts, key=color_counts.get)
        primary_color_value = tuple(pixel for pixel in grid_pixels if get_primary_color_name(tuple(pixel)) == primary_color_name)[0] 
                             
        primary_color_name = closest_xkcd_color((primary_color_value[0], primary_color_value[1], primary_color_value[2]) ).replace("xkcd:", "")
        
        return primary_color_name, primary_color_value
    return "Unknown", (0, 0, 0)