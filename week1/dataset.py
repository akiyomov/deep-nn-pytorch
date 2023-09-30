import torch
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
import os
from PIL import Image

class Dataset(Dataset):
    def __init__(self,csv_file,data_dir,transform=None):
        self.transform = transform
        self.data_dir = data_dir
        self.data_name = pd.read_csv(csv_file)
        self.len = self.data_name.shape[0]
    
    def __len__(self):
        return self.len
    
    def __getitem__(self,idx):
        img_name = os.path.join(self.data_dir,self.data_name.iloc[idx,0])
        image = Image.open(img_name)
        y_label = torch.tensor(int(self.data_name.iloc[idx,1]))
        if self.transform:
            image = self.transform(image)
        return (image,y_label)
    
crop_tensor = transforms.Compose([transforms.CenterCrop(300),transforms.ToTensor()])