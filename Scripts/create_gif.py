import os
import imageio.v2 as imageio  # imageio.v2 avoids deprecation warning
from natsort import natsorted

# Set the input and output paths
image_dir = "./Images/"
output_gif = "./composite_animation.gif"
frame_duration = 0.5  # seconds per frame

# Collect and sort all PNG files
image_files = natsorted([
    os.path.join(image_dir, f)
    for f in os.listdir(image_dir)
    if f.endswith(".png")
])

# Read and compile images into a GIF
if image_files:
    print(f"Creating GIF with {len(image_files)} frames...")
    with imageio.get_writer(output_gif, mode='I', duration=frame_duration) as writer:
        for filename in image_files:
            image = imageio.imread(filename)
            writer.append_data(image)
            print(f"Added {filename}")
    print(f"GIF saved to: {output_gif}")
else:
    print("No PNG images found in ../Images/")
