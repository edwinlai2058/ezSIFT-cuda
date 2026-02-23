
import os
import glob
from PIL import Image

# Define source directory (where the binaries and images are)
# Using absolute path structure based on where this script will be located (project root)
base_dir = os.path.dirname(os.path.abspath(__file__))
source_dir = os.path.join(base_dir, "platforms/desktop/build/bin")
output_dir = source_dir # Save directly in the bin folder as per user preference

# Verify source directory exists
if not os.path.exists(source_dir):
    print(f"Error: Source directory not found at {source_dir}")
    print("Did you run the build?")
    exit(1)

# Extensions to look for
extensions = ['*.pgm', '*.ppm']

files_to_convert = []
for ext in extensions:
    # Use glob to find files matching the extension in the source directory
    files_to_convert.extend(glob.glob(os.path.join(source_dir, ext)))

print(f"Found {len(files_to_convert)} image files in {source_dir}")

for file_path in files_to_convert:
    try:
        filename = os.path.basename(file_path)
        name_without_ext = os.path.splitext(filename)[0]
        output_path = os.path.join(output_dir, name_without_ext + ".png")
        
        # Check if PNG already exists to avoid redundant work
        if os.path.exists(output_path):
             print(f"Skipping {filename} (PNG already exists)")
             continue

        with Image.open(file_path) as img:
            img.save(output_path)
            print(f"Converted: {filename} -> {name_without_ext}.png")
            
    except Exception as e:
        print(f"Failed to convert {filename}: {str(e)}")

print(f"\nDone! PNG images are located in:\n{output_dir}")
