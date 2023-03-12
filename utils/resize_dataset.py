import os
from PIL import Image

input_dir = '/home/fwojciak/projects/Unsupervised-Classification/costarica_2017_32x32'
size = (32, 32)  # set the desired size of the images

for file_name in os.listdir(input_dir):
    input_path = os.path.join(input_dir, file_name)
    
    # skip non-image files
    if not os.path.isfile(input_path) or not file_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
        continue

    with Image.open(input_path) as img:
        img = img.resize(size)
        img.save(input_path)
