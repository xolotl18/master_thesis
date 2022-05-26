import os
import time
import sys
import argparse
import numpy as np
import onnx
import onnxruntime
import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchmetrics import JaccardIndex
from statistics import mean
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from statistics import mean
from tqdm import tqdm

c_dir = os.getcwd()
mt_dir = os.path.dirname(c_dir)
models_path = os.path.join(mt_dir, "models")
sys.path.insert(1, models_path)
from fast_scnn import FastSCNN
from small_scnn import SmallSCNN
from super_small_scnn import SuperSmallSCNN

class PackagesInferenceDataset(Dataset):
    def __init__(self, images_filenames, images_directory, masks_directory, transform=None,):
        self.images_filenames = images_filenames
        self.images_directory = images_directory
        self.masks_directory = masks_directory
        self.transform = transform

    def __len__(self):
        return len(self.images_filenames)

    def __getitem__(self, idx):
        image_filename = self.images_filenames[idx]
        image = cv2.imread(os.path.join(self.images_directory, image_filename))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(
            os.path.join(self.masks_directory, image_filename), cv2.IMREAD_UNCHANGED,
        )
        mask = mask.astype(np.float32)
        mask[mask == 0.0] = 0.0
        mask[mask == 255.0] = 1.0
        original_size = tuple(image.shape[:2])
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        return image, mask, original_size

test_transform = A.Compose(
    [
        A.Resize(512, 512),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)


test_images_path = os.path.join(mt_dir, "full_dataset/test/images")
test_label_path = os.path.join(mt_dir, "full_dataset/test/labels")
test_images_filenames = [item for item in os.listdir(test_images_path) if item.endswith(".png")]
test_dataset = PackagesInferenceDataset(
    images_filenames=test_images_filenames,
    images_directory=test_images_path, 
    masks_directory=test_label_path, 
    transform=test_transform
)

model_path = os.path.join(mt_dir, "model_checkpoints/fast_scnn40e_noinit.onnx")
#load onnx model
onnx_model = onnx.load(model_path)
onnx.checker.check_model(onnx_model)

#inference with onnx
n_runs = 1
ort_session = onnxruntime.InferenceSession(model_path)

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

#inference over the entire test dataset one image at a time
latency_onnx = []
i = 0
for run in tqdm(range(n_runs)):
    for image, mask, (height, width) in test_dataset:
        image = image.unsqueeze(0)
        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(image)}
        i+=1
        start = time.time()
        ort_outs = ort_session.run(None, ort_inputs)
        lat = time.time()-start
        latency_onnx.append(lat)
        #print(f"inference latency on image {i} is {lat}")

print(f"The mean latency in onnx is {mean(latency_onnx)}")