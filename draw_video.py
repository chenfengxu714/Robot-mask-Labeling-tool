import cv2
import numpy as np
from PIL import Image
import os
import cv2
import numpy as np
from PIL import Image
import os
import re

def combine_rgb_and_mask(rgb_image_path, mask_image_path, output_image_path):
    """Combine an RGB image with a mask and save the result."""
    rgb_image = Image.open(rgb_image_path).convert("RGBA")
    mask_image = Image.open(mask_image_path).convert("L")

    # Convert mask to RGBA with an alpha channel
    mask_rgba = mask_image.convert("RGBA")
    mask_rgba = mask_rgba.point(lambda p: p * 0.5)  # Adjust the transparency level if needed

    combined_image = Image.alpha_composite(rgb_image, mask_rgba)
    combined_image.save(output_image_path)

def sort_files_numerically(file_list):
    """Sort files numerically based on the numerical part of the filename."""
    def extract_number(filename):
        match = re.search(r'(\d+)', filename)
        return int(match.group()) if match else -1
    
    return sorted(file_list, key=extract_number)

def create_video_from_images(rgb_dir, mask_dir, output_video_path, fps=30):
    """Create a video from RGB images and masks stored in different directories."""
    rgb_files = [f for f in os.listdir(rgb_dir) if f.endswith('.png') or f.endswith('.jpg')]
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.png') or f.endswith('.jpg')]

    rgb_files = sort_files_numerically(rgb_files)
    mask_files = sort_files_numerically(mask_files)

    if len(rgb_files) != len(mask_files):
        raise ValueError("The number of RGB images does not match the number of masks.")

    # OpenCV VideoWriter
    first_image = Image.open(os.path.join(rgb_dir, rgb_files[0]))
    width, height = first_image.size
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'XVID' for .avi
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for rgb_file, mask_file in zip(rgb_files, mask_files):
        rgb_image_path = os.path.join(rgb_dir, rgb_file)
        mask_image_path = os.path.join(mask_dir, mask_file)
        temp_output_image_path = 'temp_combined.png'
        
        combine_rgb_and_mask(rgb_image_path, mask_image_path, temp_output_image_path)
        
        # Read combined image
        combined_image = cv2.imread(temp_output_image_path)
        video_writer.write(combined_image)

    video_writer.release()
    os.remove(temp_output_image_path)


# Example usage
rgb_dir = '/home/lawrence/chenfeng_file/SAMV2/rebuttal_data_source/ur5_img_and_language/berkeley_autolab_ur5_traj_0'  # Update to your RGB images directory
mask_dir = '/home/lawrence/chenfeng_file/SAMV2/rebuttal_data_source/ur5_img_and_language_masks/berkeley_autolab_ur5_traj_0'  # Update to your masks directory
output_video_path = 'output_video.mp4'  # Desired output video file path
fps = 30  # Frames per second for the video

create_video_from_images(rgb_dir, mask_dir, output_video_path, fps)
