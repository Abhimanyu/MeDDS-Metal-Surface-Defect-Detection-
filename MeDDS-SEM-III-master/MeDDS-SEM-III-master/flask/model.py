
# # IMPORTS FOR SETUP
# import os
# import json
# import shutil
# import sys

# # IMPORTS FOR PROCESS
# import pdb
import os
import cv2
# import torch
# import pandas as pd
import numpy as np
# from tqdm import tqdm
# import torch.backends.cudnn as cudnn
# from torch.utils.data import DataLoader, Dataset
# from albumentations import (Normalize, Compose)
# from albumentations.pytorch import ToTensor
# import torch.utils.data as data

# # IMPORT MATPLOTLIB
# import matplotlib.pyplot as plt
# # %matplotlib inline

# # def setup():
# #   # FILE SYSTEM
# #   token = {"username":"abhinavrob","key":"8fe52d7b61ab4ffe3beca0e8687385ad"}
# #   if not os.path.exists('/content/.kaggle'):
# #     os.mkdir('/content/.kaggle')
# #   with open('/content/.kaggle/kaggle.json', 'w') as file:
# #     json.dump(token, file)
# #   !chmod 600 /content/.kaggle/kaggle.json
# #   if not os.path.exists('~/.kaggle'):
# #     os.mkdir('/root/.kaggle')
# #   !cp /content/.kaggle/kaggle.json ~/.kaggle/
# #   !kaggle config set -n path -v /content

# #   # DOWNLOAD DATASETS
# #   !kaggle datasets download -d gontcharovd/resnetmodels
# #   !kaggle datasets download -d gontcharovd/resnetunetmodelcode
# #   !kaggle datasets download -d gontcharovd/senetmodels
# #   !kaggle datasets download -d gontcharovd/senetunetmodelcode
# #   !kaggle competitions download -c severstal-steel-defect-detection
# #   !kaggle datasets download -d lightforever/mlcomp
# #   !git clone https://github.com/Cadene/pretrained-models.pytorch.git

# #   # READY FOR EXTRACTING
# #   !mkdir input
# #   shutil.move("/content/datasets/gontcharovd/resnetmodels", "/content/input/resnetmodels")
# #   shutil.move("/content/datasets/gontcharovd/resnetunetmodelcode", "/content/input/resnetunetmodelcode")
# #   shutil.move("/content/datasets/gontcharovd/senetmodels", "/content/input/senetmodels")
# #   shutil.move("/content/datasets/gontcharovd/senetunetmodelcode", "/content/input/senetunetmodelcode")
# #   shutil.move("/content/competitions/severstal-steel-defect-detection", "/content/input/severstal-steel-defect-detection")
# #   shutil.move("/content/pretrained-models.pytorch/pretrainedmodels", "/content/input/pretrainedmodels")

# #   # UNZIP
# #   !unzip /content/input/resnetmodels/resnetmodels.zip -d /content/input/resnetmodels
# #   !unzip /content/input/resnetunetmodelcode/resnetunetmodelcode.zip -d /content/input/resnetunetmodelcode
# #   !unzip /content/input/senetmodels/senetmodels.zip -d /content/input/senetmodels
# #   !unzip /content/input/senetunetmodelcode/senetunetmodelcode.zip -d /content/input/senetunetmodelcode
# #   !unzip /content/input/severstal-steel-defect-detection/test_images.zip -d /content/input/severstal-steel-defect-detection/test_images
# #   !unzip /content/input/severstal-steel-defect-detection/train_images.zip -d /content/input/severstal-steel-defect-detection/train_images
# #   !unzip /content/input/severstal-steel-defect-detection/train.csv.zip -d /content/input/severstal-steel-defect-detection
# #   !unzip /content/datasets/lightforever/mlcomp/mlcomp.zip -d /content/datasets/lightforever/mlcomp

# #   # INSTALL CUSTOM LIBRARIES
# #   !python datasets/lightforever/mlcomp/mlcomp/mlcomp/setup.py

# #   # PATH SETUP
# #   package_path = '/content/input/senetunetmodelcode'
# #   package_path_2 = '/content/input'
# #   sys.path.append(package_path)
# #   sys.path.append(package_path_2)

# #   # SHOW FILE SYSTEM (DIAGNOSTICS)
# #   os.listdir('/content/input')
# #   !ls /content/input/pretrainedmodels
# #   !ls /content/input/resnetunetmodelcode
# #   !ls /content/input/resnetmodels
# #   !ls /content/input/senetunetmodelcode
# #   !ls /content/input/senetmodels

# # setup()
# print("SETUP SUCCESS!")

# sample_submission_path = '/content/input/severstal-steel-defect-detection/sample_submission.csv'
# test_data_folder = "/content/input/severstal-steel-defect-detection/test_images"
# img_folder = '/content/input/severstal-steel-defect-detection/test_images'

# def mask2rle(img):
#     '''
#     img: numpy array, 1 - mask, 0 - background
#     Returns run length as string formated
#     '''
#     pixels= img.T.flatten()
#     pixels = np.concatenate([[0], pixels, [0]])
#     runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
#     runs[1::2] -= runs[::2]
#     return ' '.join(str(x) for x in runs)

# def post_process(probability, threshold, min_size):
#     '''Post processing of each predicted mask, components with lesser number of pixels
#     than `min_size` are ignored'''
#     mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]
#     num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
#     predictions = np.zeros((256, 1600), np.float32)
#     num = 0
#     for c in range(1, num_component):
#         p = (component == c)
#         if p.sum() > min_size:
#             predictions[p] = 1
#             num += 1
#     return predictions, num

# class TestDataset(Dataset):
#     '''Dataset for test prediction'''
#     def __init__(self, root, df, mean, std):
#         self.root = root
#         df['ImageId'] = df['ImageId_ClassId'].apply(lambda x: x.split('_')[0])
#         self.fnames = df['ImageId'].unique().tolist()
#         self.num_samples = len(self.fnames)
#         self.transform = Compose(
#             [
#                 Normalize(mean=mean, std=std, p=1),
#                 ToTensor(),
#             ]
#         )

#     def __getitem__(self, idx):
#         fname = self.fnames[idx]
#         path = os.path.join(self.root, fname)
#         image = cv2.imread(path)
#         images = self.transform(image=image)["image"]
#         return fname, images

#     def __len__(self):
#         return self.num_samples

# # initialize test dataloader
# best_threshold = 0.5
# num_workers = 2
# batch_size = 4
# print('best_threshold', best_threshold)
# min_size = 3500
# mean = (0.485, 0.456, 0.406)
# std = (0.229, 0.224, 0.225)
# df = pd.read_csv(sample_submission_path)
# testset = DataLoader(
#     TestDataset(test_data_folder, df, mean, std),
#     batch_size=batch_size,
#     shuffle=False,
#     num_workers=num_workers,
#     pin_memory=True
# )

# from senet_unet_model_code import Unet
# # Initialize mode and load trained weights
# # ckpt_path = "../input/resnetmodels/resnet18_20_epochs.pth"            OTHER AVAILABLE MODELS
# # ckpt_path = "../input/senetmodels/senet50_20_epochs.pth"                     ^^^
# ckpt_path = "/content/input/senetmodels/senext50_30_epochs.pth"
# device = torch.device("cuda")
# # change the encoder name in the Unet() call.
# model = Unet('se_resnext50_32x4d', encoder_weights=None, classes=4, activation=None)
# model.to(device)
# model.eval()
# state = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
# model.load_state_dict(state["state_dict"])

# # start prediction
# predictions = []
# # for i, batch in enumerate(tqdm(testset)):
# #     fnames, images = batch
# #     batch_preds = torch.sigmoid(model(images.to(device)))
# #     batch_preds = batch_preds.detach().cpu().numpy()
# #     for fname, preds in zip(fnames, batch_preds):
# #         for cls, pred in enumerate(preds):
# #             pred, num = post_process(pred, best_threshold, min_size)
# #             rle = mask2rle(pred)
# #             name = fname + f"_{cls+1}"
# #             predictions.append([name, rle])

# # save predictions to submission.csv
# df = pd.DataFrame(predictions, columns=['ImageId_ClassId', 'EncodedPixels'])
# df.to_csv("submission.csv", index=False)

# df.head()

# from mlcomp.contrib.transform.rle import rle2mask, mask2rle
# df['Image'] = df['ImageId_ClassId'].map(lambda x: x.split('_')[0])
# df['Class'] = df['ImageId_ClassId'].map(lambda x: x.split('_')[1])
# df['empty'] = df['EncodedPixels'].map(lambda x: not x)
# df[df['empty'] == False]['Class'].value_counts()

# def plot_pictures():
#   df = pd.read_csv('submission.csv')[:500]
#   df['Image'] = df['ImageId_ClassId'].map(lambda x: x.split('_')[0])
#   df['Class'] = df['ImageId_ClassId'].map(lambda x: x.split('_')[1])

#   for row in df.itertuples():
#       img_path = os.path.join(img_folder, row.Image)
#       img = cv2.imread(img_path)
#       mask = rle2mask(row.EncodedPixels, (1600, 256)) \
#           if isinstance(row.EncodedPixels, str) else np.zeros((256, 1600))
#       if mask.sum() == 0:
#           continue
      
#       fig, axes = plt.subplots(1, 2, figsize=(20, 60))
#       axes[0].imshow(img/255)
#       axes[1].imshow(mask*60)
#       axes[0].set_title(row.Image)
#       axes[1].set_title(row.Class)
#       plt.show()

# plot_pictures()

def mask(imagename,path,lst):
    img = np.zeros(1600*256)
    for i in lst:
        img[i-1]=255
    img = np.reshape(img,(1600,256)).T
    path = path+"/"+imagename
    cv2.imwrite(path, img)
    return img