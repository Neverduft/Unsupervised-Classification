import os
from torch.utils.data import Dataset
from PIL import UnidentifiedImageError
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None, train=True):
        self.root_dir = root_dir
        self.transform = transform
        self.train = train  # training set or test set
        self.classes = []
        self.images = [f for f in os.listdir(root_dir) if f.endswith('.jpg') or f.endswith('.jpeg') or f.endswith('.png')]
        self.classes = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        
        self.data = []
        self.targets = []

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.root_dir, img_name)
        try:
            image = Image.open(img_path).convert('RGB')
        except UnidentifiedImageError:
            print(f"Skipping {img_path} due to UnidentifiedImageError.")
        if self.transform is not None:
            image = self.transform(image)        

        return {'image': image, 'target': 1}

    def get_image(self, index):
        img_name = self.images[index]
        img_path = os.path.join(self.root_dir, img_name)
        return img_path