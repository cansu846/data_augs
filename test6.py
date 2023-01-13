import cv2
import numpy as np
from fileinput import filename
import torch
import torch.nn as nn 
import torch.optim as optim  
import torchvision.transforms as transforms 
import torchvision
import os
import pandas as pd
import shutil
from torch.utils.data import (Dataset,DataLoader) 
from skimage import io
import time
from torchvision.utils import save_image
import warnings
warnings.filterwarnings("ignore")

class veri(Dataset):
    def __init__(self, txt_file, root_dir, transform=None):
        self.annotations = pd.read_csv(txt_file )
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = io.imread(img_path)
        #y_label = torch.tensor(int(self.annotations.iloc[index, 1]))
       
        filename= img_path[len(self.root_dir)+1:-4]
        #print(self.transform)
        if self.transform:
            image = self.transform(image)

        return (image, filename)


veri_arttirma=transforms.Compose([
transforms.ToPILImage(), #bir tensor veya numpy arrayi pil ye dönüştürür
transforms.Resize((640,640)), 
#transforms.CenterCrop(), #görüntüyü merkezden kırpma
#transforms.RandomCrop((32,32)), #görütnyü random kırpar
transforms.ColorJitter(brightness=1.0,contrast=0.5),
#transforms.RandomRotation(degrees=90), #görüntüyü açıya göre döndürme
#transforms.RandomVerticalFlip(p=0.05), #veirlen görüntüyü rastgele olarak çevirir
#transforms.RandomGrayscale(p=0.2),
transforms.ToTensor(), #
transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
transforms.GaussianBlur(kernel_size=1),
#transforms.Grayscale(num_output_channels=1),
# transforms.RandomPerspective(distortion_scale=0.5, p=0.5, fill=3)
])


# f = open("train.txt", "r")
# a = f.read()
# arr = a.split("\n")

# print(arr)

# df = pd.DataFrame(arr)

# df.to_csv("train.csv")

# pd.DataFrame(arr).to_csv("train.csv")
    
dataset = veri(
                txt_file= "train.txt",
                root_dir= "C:/proje/data_augs/", #verinin konumu
                transform= veri_arttirma)
print(len(dataset))
print(dataset)  
print(dataset.annotations)

#görüntüleri dosyaya kaydetcek
foto_sayi=0
for i in range(5): #ne kadar fazla verirsek o kadar fazla veri
    for image ,fname in dataset:
        newfname = f'{fname}_'+str(foto_sayi)
        print(newfname)
        save_image(image,f"{newfname}.png")
        img =cv2.imread(f'C:/proje/data_augs/data/{newfname}.png')
        counts = np.count_nonzero(img)
        if(counts <= 10000):
            os.remove(f'C:/proje/data_augs/data/{newfname}.png')
            continue
        foto_sayi+=1
        shutil.copyfile(f"C:/proje/data_augs/txt/{fname}.txt",f"C:/proje/data_augs/{newfname}.txt")