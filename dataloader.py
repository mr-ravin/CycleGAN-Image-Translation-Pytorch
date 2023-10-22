import torch
import glob
import cv2
import json
from torch.utils.data import Dataset


class CustomImageDataset(Dataset):
    def __init__(self, transform=None, mode="train"):
        self.path_A = "dataset/"+mode+"/A/"
        self.path_B = "dataset/"+mode+"/B/"
        self.filenames_list_A = glob.glob(self.path_A+"*.jpg")
        self.filenames_list_B = glob.glob(self.path_B+"*.jpg")
        self.transform = transform

    def __len__(self):
        return len(self.filenames_list_A)

    def __getitem__(self, idx):
        img_path_A = self.filenames_list_A[idx]
        image_A = cv2.imread(img_path_A)
        image_A = image_A/255.0
        img_path_B = self.filenames_list_B[idx]
        image_B = cv2.imread(img_path_B)
        image_B = image_B/255.0
        if self.transform:
            image_A = self.transform(image=image_A)["image"]
            image_B = self.transform(image=image_B)["image"]
            
        return image_A, image_B, img_path_A, img_path_B