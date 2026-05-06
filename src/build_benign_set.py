import os
import numpy as np
from PIL import Image

def create_benign_images(source_dir, output_dir, limit=500):
    os.makedirs(output_dir, exist_ok=True)
    count = 0
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith(('.exe', '.dll', '.sys')) and count < limit:
                try:
                    path = os.path.join(root, file)
                    with open(path, 'rb') as f:
                        print(f.name)
                        d = np.frombuffer(f.read(), dtype=np.uint8)
                    
                    # Convert to square image logic
                    side = int(len(d)**0.5)
                    if side < 10: continue # Skip tiny files
                    img_array = d[:side*side].reshape((side, side))
                    
                    img = Image.fromarray(img_array).resize((224, 224), Image.NEAREST)
                    img.save(f"{output_dir}/benign_{count}.png")
                    count += 1
                except:
                    continue
    print(f"Created {count} benign images in {output_dir}")

# Source: Windows System32 is a great place for standard benign binaries
create_benign_images('C:/Windows/System32', './malimg_dataset/train/Benign')
create_benign_images('C:/Windows/SysWOW64', './malimg_dataset/test/Benign', limit=100)