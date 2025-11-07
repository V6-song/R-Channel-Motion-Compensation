from PIL import Image
import matplotlib.pyplot as plt
import os
from option import opt

def extract_rgb_channels(image_path):
    image = Image.open(image_path)
    r, g, b = image.split()
    
    r_image = r.convert("L")  
    return image, r_image

def save_images(r_image, output_folder, base_name):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    r_image.save(f'{output_folder}/{base_name}.png')
    print(f"Saved {output_folder}\{base_name}.png")

def process_folder(input_folder, output_folder):
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file)
                base_name = os.path.splitext(file)[0]
                _, r_image = extract_rgb_channels(image_path)
                save_images(r_image, output_folder, base_name)

def main(folder_path):
    hazy_folder = os.path.join(folder_path, 'hazy')
    clear_folder = os.path.join(folder_path, 'clear')
    hazy_r_folder = os.path.join(folder_path, 'hazy_r')
    clear_r_folder = os.path.join(folder_path, 'clear_r')
    
    process_folder(hazy_folder, hazy_r_folder)
    process_folder(clear_folder, clear_r_folder)

folder_path = opt.base_dir
main(folder_path)