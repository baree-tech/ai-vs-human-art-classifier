import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm

# Paths
root_dir = '/media/bareera/KINGSTON/Bareera/Dataset/AI_vs_RealDataset'
output_dir = '/media/bareera/KINGSTON/Bareera/Dataset/AI_vs_Real'
os.makedirs(output_dir, exist_ok=True)

# Only include one class at a time to prevent system overload
classes = ['human']  # You can  change to ['human'] ['AI'] or run in batches to avoid cpu overload

# Control how many images to take per subfolder
MAX_IMAGES_PER_SUBFOLDER = 500  # To avoid loading too many at once

# Train, val, test splits
splits = ['train', 'val', 'test']
split_ratio = {'train': 0.7, 'val': 0.2, 'test': 0.1}

# Create target folder structure
for split in splits:
    for cls in classes:
        os.makedirs(os.path.join(output_dir, split, cls), exist_ok=True)

# Helper function to collect images from nested folders (limited per subfolder)
def get_limited_images(folder_path, max_per_subfolder=MAX_IMAGES_PER_SUBFOLDER):
    image_paths = []
    for subdir, _, files in os.walk(folder_path):
        images = [os.path.join(subdir, file) for file in files
                  if file.lower().endswith(('.jpg', '.jpeg', '.png'))]
        random.shuffle(images)
        image_paths.extend(images[:max_per_subfolder])
    return image_paths

# Split and copy images
for cls in classes:
    print(f"Processing class: {cls}")
    full_class_path = os.path.join(root_dir, cls)

    all_images = get_limited_images(full_class_path)
    random.shuffle(all_images)
    n = len(all_images)

    train_split = int(n * split_ratio['train'])
    val_split = int(n * (split_ratio['train'] + split_ratio['val']))
 
    split_dict = {    # Do oneby one to prevent overload
        # 'train': all_images[:train_split],
        # Enable these one by one if system crashes:   
        # 'val': all_images[train_split:val_split],
        'test': all_images[val_split:]
    }

    for split in splits:
        if split not in split_dict:
            continue
        print(f"Copying {split} images for class {cls}...")
        for img_path in tqdm(split_dict[split], desc=f"{cls} - {split}", ncols=100):
            fname = Path(img_path).name
            dest = os.path.join(output_dir, split, cls, fname)
            shutil.copy(img_path, dest)

print("\n✅ Dataset successfully split and copied.")
