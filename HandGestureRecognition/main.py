from img_to_pixel import create_pixel_array, create_pixel_array_single
from metadata_file import FileMetaData
from dataloader import prepare_dataloader
import global_constants as gc
from knowledge import Learning
import torch
import torch.nn as tnn
import torch.nn.functional as tnf
from classification_cnn import CNN_2D
from capture_image import capture
from capture_image_2 import capture_2
import cv2
import os
import warnings
warnings.filterwarnings('ignore')

from dummy import run_dummy


if __name__ == '__main__':

    #run_dummy()

    #file_metadata = FileMetaData(gc.BASE_PATH, gc.TRAIN_FOLDER, gc.TEST_FOLDER, gc.TARGET_FOLDER)
    #file_metadata.process_metadata()

    create_pixel_array(gc.TRAIN_FOLDER, 'train')
    create_pixel_array(gc.TEST_FOLDER, 'test')

    X_train = torch.load(gc.BASE_PATH + "\\" + gc.INPUT_FEATURES_ARRAY + "_train.pt")
    y_train = torch.load(gc.BASE_PATH + "\\" + gc.LABEL_ARRAY + "_train.pt")
    #y_train = tnf.one_hot(y_train, num_classes=len(gc.LABEL_TO_IDX))

    X_test = torch.load(gc.BASE_PATH + "\\" + gc.INPUT_FEATURES_ARRAY + "_test.pt")
    y_test = torch.load(gc.BASE_PATH + "\\" + gc.LABEL_ARRAY + "_test.pt")
    #y_test = tnf.one_hot(y_test, num_classes=len(gc.LABEL_TO_IDX))

    print(X_train.shape, y_train.shape)
    train_dataloader = prepare_dataloader(X_train, y_train)
    test_dataloader = prepare_dataloader(X_test, y_test)

    cnn_model = Learning(CNN_2D())

    cnn_model.train(train_dataloader)

    if gc.VALIDATE_TRAINING:
        cnn_model.test(test_dataloader)

    capture(cnn_model)

    if gc.TEST_SINGLE:
        single_img, single_img_label = create_pixel_array_single(gc.TEST_IMG, gc.TEST_IMG_LABEL)
        single_img_label = tnf.one_hot(single_img_label, num_classes=len(gc.LABEL_TO_IDX))
        cnn_model.test_single_known(single_img, single_img_label)