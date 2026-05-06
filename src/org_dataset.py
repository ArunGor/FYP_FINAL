import os
import shutil
import random

def setup_folders(base_path, output_path, split_ratio=0.8):
    # Base path: where you unzipped the 9,339 images
    # Output path: where you want the Train/Test folders
    
    classes = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    
    for cls in classes:
        # Create Train/Test directories for each class
        train_dir = os.path.join(output_path, 'train', cls)
        test_dir = os.path.join(output_path, 'test', cls)
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
        
        # Get all images for this class
        all_imgs = os.listdir(os.path.join(base_path, cls))
        random.shuffle(all_imgs)
        
        # Calculate split
        split_point = int(len(all_imgs) * split_ratio)
        train_imgs = all_imgs[:split_point]
        test_imgs = all_imgs[split_point:]
        
        # Move files
        for img in train_imgs:
            shutil.copy(os.path.join(base_path, cls, img), os.path.join(train_dir, img))
        for img in test_imgs:
            shutil.copy(os.path.join(base_path, cls, img), os.path.join(test_dir, img))
            
    print(f"Successfully split {len(classes)} classes into Train and Test folders.")


setup_folders('./malimg_paper_dataset_imgs', 'C:/Users/mrmon/fypmalware/data/bigprocessed/malimg_dataset')