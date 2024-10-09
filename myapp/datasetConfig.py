from PIL import Image
import os

def convert_png_to_tif(input_folder, output_folder):
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Iterate through all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.png'):
            png_path = os.path.join(input_folder, filename)
            tif_path = os.path.join(output_folder, filename.replace('.png', '.tif'))
            
            # Open the .png file
            with Image.open(png_path) as img:
                # Convert and save as .tif
                img.save(tif_path, format='TIFF')
                print(f"Converted: {filename} -> {tif_path}")

# Example usage
input_folder = 'dataset/'
output_folder = 'tif/'
convert_png_to_tif(input_folder, output_folder)
