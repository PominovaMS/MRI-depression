import os
import copy
import numpy as np
import pandas as pd
import nibabel as nib
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.utils.data as data
from utils import *

class LA5studyMRI(data.Dataset):
    """
    Arguments:
        paths: paths to data folders
        target_path: path to file with targets and additional information
        load_online (bool): if True, load mri images online. Else, preload everything during initialization
        mri_type (str): sMRI or fMRI
        mri_file_suffix (str): substring in file name to find required mri files
        coord_min, img_shape: 
            cut from mri image array sub-array of shape img_shape
            starting from coord_min point
        start_pos, seq_len (for fMRI only): 
            cut from fMRI sequence subseqence of length seq_len (full sequence if None)
            starting from start_pos time point (selected randomly if None)
    """
    
    def __init__(self, paths, target_path, load_online=False, mri_type="sMRI", mri_file_suffix="",
                coord_min=(20, 20, 20,), img_shape=(152, 188, 152,), start_pos=None, seq_len=None):
        self.mri_paths = {
            "participant_id" : [],
            "path" : [],
        }
        self.paths = paths
        self.target = pd.read_csv(target_path)
        self.load_online = load_online
        
        self.mri_type = mri_type
        if self.mri_type == "sMRI":
            self.type = "anat" 
        elif self.mri_type == "fMRI":
            self.type = "func"
        else:
            raise ValueError("Select sMRI or fMRI mri type.")
        self.mri_file_suffix = mri_file_suffix
    
        self.coord_min = coord_min
        self.img_shape = img_shape
        self.start_pos = start_pos
        self.seq_len = seq_len
        
        for path_to_folder in self.paths:
            for patient_folder_name in os.listdir(path_to_folder):
                if 'sub-' in patient_folder_name and os.path.isdir(path_to_folder + patient_folder_name) and \
                    self.type in os.listdir(path_to_folder + patient_folder_name):        
                    temp_path = path_to_folder + patient_folder_name + "/" + self.type + "/"
                    for filename in os.listdir(temp_path):
                        if self.mri_file_suffix in filename:
                            self.mri_paths["participant_id"].append(patient_folder_name)
                            full_path = temp_path + filename
                            self.mri_paths["path"].append(full_path)
        self.mri_paths = pd.DataFrame(self.mri_paths)
        
        self.target = self.target.merge(self.mri_paths, on="participant_id")
        self.mri_files = self.target.path.tolist()
        if not self.load_online:
            self.mri_files = [self.get_image(f) for f in tqdm(self.mri_files)]
#             self.mri_files = list(map(self.get_image, self.mri_files))

            
    def reshape_image(self, mri_img, coord_min, img_shape):
        if self.mri_type == "sMRI":
            return mri_img[coord_min[0]:coord_min[0] + img_shape[0],
                           coord_min[1]:coord_min[1] + img_shape[1],
                           coord_min[2]:coord_min[2] + img_shape[2]].reshape((1,) + img_shape)
        if self.mri_type == "fMRI":
            seq_len = mri_img.shape[-1]
            return mri_img[coord_min[0]:coord_min[0] + img_shape[0],
                           coord_min[1]:coord_min[1] + img_shape[1],
                           coord_min[2]:coord_min[2] + img_shape[2], :].reshape((1,) + img_shape + (seq_len,))
        
    def get_image(self, mri_file, start_pos=None, seq_len=None):
        if "nii" in mri_file:
            img = load_nii_to_array(mri_file)
        else:
            img = np.load(mri_file)
        img = self.reshape_image(img, self.coord_min, self.img_shape)
        
        if self.mri_type == "sMRI":
            if self.transform is not None:
                img = self.transform(img)
            return img
        
        if self.mri_type == "fMRI":
            if seq_len is None:
                seq_len = img.shape[-1]
            if start_pos is None:
                start_pos = np.random.choice(img.shape[-1] - seq_len)
            if seq_len == 1:
                img = img[:, :, :, :, start_pos]
            else:
                img = img[:, :, :, :, start_pos:start_pos + seq_len]
            if self.transform is not None:
                img = self.transform(img)
            return img
    
    def __getitem__(self, index):
        if not self.load_online:
            return self.mri_files[index]
        return self.get_image(self.mri_files[index], self.start_pos, self.seq_len)
    
    def __len__(self):
        return len(self.mri_files)


# [TODO] add Gueht data loader and correct labels and paths merging