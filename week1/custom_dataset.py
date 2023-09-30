import torch
from torch.utils.data import Dataset
from torchvision import transforms


class add_mult(object):
    def __init__(self,addx=1,muly=2):
        self.addx = addx
        self.muly = muly
    def __call__(self,sample):
        x = sample[0]
        y = sample[1]
        x = x+self.addx
        y = y*self.muly
        sample = x,y
        return sample
    
class mult(object):
    def __init__(self,muly=2):
        self.muly = muly
    def __call__(self,sample):
        x = sample[0]
        y = sample[1]
        x = x*self.muly
        y = y*self.muly
        sample = x,y
        return sample

class CustomDataset(Dataset):
    def __init__(self,length=100,transform=transforms.Compose([add_mult(),mult()])):
        self.x = 2*torch.ones(length,2)
        self.y = torch.ones(length,1)
        self.len = length
        self.transform = transform
    def __getitem__(self,index):
        sample = self.x[index],self.y[index]
        if self.transform:
            sample = self.transform(sample)
        return sample
    def __len__(self):
        return self.len
