import os
import numpy as np
from PIL import Image, ImageEnhance
import random

# Random Rotation
def random_rotate(image):
    angle = np.random.uniform(-20, 20)
    image = image.rotate(angle)
    return image

# Adjust Brightness
def random_brightness(image, min_factor=0.3, max_factor=1.75):
    enhancer = ImageEnhance.Brightness(image)
    factor = random.uniform(min_factor, max_factor)
    return enhancer.enhance(factor)

# Adjust Contrast
def random_contrast(image, min_factor=0.3, max_factor=1.75):
    enhancer = ImageEnhance.Contrast(image)
    factor = random.uniform(min_factor, max_factor)
    return enhancer.enhance(factor)

# Color Jitter
def random_color_jitter(image, jitter_range=0.35):
    random.seed()
    r, g, b = image.split()
    min_range = -int(jitter_range * 255)
    max_range = int(jitter_range * 255) 
    r = r.point(lambda i: i + random.randint(min_range, max_range))
    g = g.point(lambda i: i + random.randint(min_range, max_range))
    b = b.point(lambda i: i + random.randint(min_range, max_range))
    return Image.merge('RGB', (r, g, b))

# Gaussian Noise
def add_gaussian_noise(image, mean=0, std=35):
    image_array = np.array(image)
    noise = np.random.normal(mean, std, image_array.shape)
    noisy_image = np.clip(image_array + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_image)

# Salt Pepper Noise
def add_salt_and_pepper_noise(image, salt_prob=0.1, pepper_prob=0.1):
    image_array = np.array(image)
    height, width, channels = image_array.shape
    # print(height,width)
    
    # Salt noise
    salt_mask = np.random.rand(height, width) < salt_prob
    image_array[salt_mask] = 255
    
    # Pepper noise
    
    pepper_mask = np.random.rand(height, width) < pepper_prob
    image_array[pepper_mask] = 0
    
    return Image.fromarray(image_array)

# Input folder and Output folder
input_folder = './data/flood_is_need_aug/'
output_folder = './data/flood_is_aug/'

# Make sure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Apply data enhancement to images and save them
count = 0
aug_count = 0
for filename in os.listdir(input_folder):
    # Read image
    img_path = os.path.join(input_folder, filename)
    image = Image.open(img_path)

    # Application Data Enhancement
    augmented_image = add_salt_and_pepper_noise(image)
    augments_dict = {'rotate':random_rotate(image),'brightness':random_brightness(image),
                     'contrast':random_contrast(image),'color_jitter':random_color_jitter(image),
                     'gaussian':add_gaussian_noise(image),'salt_pepper':add_salt_and_pepper_noise(image)}

    for augment_name, image_augmented in augments_dict.items():
        need_aug = random.random() 
        if need_aug < 1/len(augments_dict):
            image_augmented_path = os.path.join(output_folder, 
                                                f'{augment_name}-{filename}')
            # Save image
            image_augmented.save(image_augmented_path)
            aug_count += 1
    count += 1
    print(f"{count}——{aug_count}")
print('Augmentation complete.')
