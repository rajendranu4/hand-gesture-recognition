import torch
import torchvision
import os
import global_constants as gc
from label_rename import convert_label_to_idx
import torch.nn.functional as tnf
from torchvision.utils import save_image
import cv2
import argparse


def create_pixel_array(folder, mode):
    if os.path.isfile(gc.BASE_PATH + "\\" + gc.INPUT_FEATURES_ARRAY + "_" + mode + ".pt"):
        return

    else:
        printl = True
        X = []
        y = []

        folder_base = gc.BASE_PATH + folder
        data_dirs = [data_dir[0] for data_dir in os.walk(folder_base)][1:]

        resize_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((128, 128))
        ])

        i = 0

        for data_dir in data_dirs:
            label = data_dir.split('\\')[-1]
            for file in os.listdir(data_dir):
                tensor_img = torchvision.io.read_image(data_dir + '\\' + file)
                tensor_img = resize_transforms(tensor_img)
                tensor_img = tensor_img / 255

                y_img = label

                X.append(tensor_img)
                y.append(y_img)

        X = torch.stack(X)
        y = convert_label_to_idx(y)
        y = torch.tensor(y)
        torch.save(X, gc.BASE_PATH + "\\" + gc.INPUT_FEATURES_ARRAY + "_" + mode + ".pt")
        torch.save(y, gc.BASE_PATH + "\\" + gc.LABEL_ARRAY + "_" + mode + ".pt")


def create_pixel_array_single(img, label):
    resize_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((128, 128)),
        #torchvision.transforms.Grayscale(num_output_channels=1)
    ])

    tensor_img = torchvision.io.read_image(img)
    tensor_img = resize_transforms(tensor_img)
    tensor_img = tensor_img / 255

    tensor_label = torch.tensor(label)

    return tensor_img, tensor_label
