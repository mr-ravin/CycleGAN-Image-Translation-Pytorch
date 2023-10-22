import torch
import glob
import cv2
import json
from torch.utils.data import Dataset


class CustomImageDataset(Dataset):
    def __init__(self, transform=None, data_type="A", mode="train"):
        self.path = "dataset/"+mode+"/"+data_type+"/"
        self.filenames_list = glob.glob(self.path+"*.jpg")
        self.transform = transform

    def __len__(self):
        return len(self.filenames_list)

    def __getitem__(self, idx):
        img_path = self.filenames_list[idx]
        image = cv2.imread(img_path)
        image = image/255.0
        if self.transform:
            image = self.transform(image=image)["image"]
        return image, img_path