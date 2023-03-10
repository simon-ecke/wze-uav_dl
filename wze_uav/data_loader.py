import torch
import os
import findatree_roi.io as io
import findatree_roi.exporter as exporter
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, List, Tuple

class CustomDataset(Dataset):
    def __init__(self, data, labels, class_names=List[str], class_idx=dict, species=None, transform=None):
        self.data = data
        self.labels = torch.from_numpy(labels)
        self.species = torch.from_numpy(species)
        self.class_names = class_names
        self.class_idx = class_idx
        self.transform = transform
    
    # overwrite __len__()
    def __len__(self) -> int:
        "Returns total number of samples"
        return len(self.data)
    
    # overwrite __getitem__() -> returns image and label 
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, float]:
        img = self.data[index]
        label = self.labels[index]
        label = label.squeeze(dim=0) # transform label into 1-dimensions
        label = int(label)
        
        if self.transform:
            return self.transform(img), label # returns image (transformed), label
        else:
            return img, label # returns image, label
        
class CustomDataset3Classes(Dataset):
    def __init__(self, data, labels, class_names=List[str], class_idx=dict, species=None, transform=None):
        self.data = data
        self.labels = torch.from_numpy(labels) #important: these are the original labels! they are changed to adapt to 3 classes below
        self.species = torch.from_numpy(species)
        self.class_names = class_names
        self.class_idx = class_idx
        self.transform = transform
    
    # overwrite __len__()
    def __len__(self) -> int:
        "Returns total number of samples"
        return len(self.data)
    
    # overwrite __getitem__() -> returns image and label 
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, float]:
        img = self.data[index]
        label = self.labels[index]
        label = label.squeeze(dim=0) # transform label into 1-dimensions
        label = int(label)
        
        #adjust labels to 3 classes
        if label <= 1: # healthy trees with labels 0 and 1
            label = 0 
        elif label > 1 and label < 4: # stressed trees with labels 2 and 3
            label = 1
        else: # dead trees with label 4
            label = 2
        
        if self.transform:
            return self.transform(img), label # returns image (transformed), label
        else:
            return img, label # returns image, label

        
def hdf5_to_img(data_path, load_sets=None):
    d = {}
    fn_list = os.listdir(data_path)
    count = 0
    for fn in fn_list:
        rois, params_rois = exporter.load_rois_from_hdf5(data_path + "\\" + fn, load_sets=load_sets) # load hdf5 data from path
        d["images{0}".format(count)] = rois["images_masked"] # assigns all crown arrays from the different hdf5 files to dictionary
        images = d["images" + str(count)].transpose(3,0,1,2) # array transpose shape from (H, W, C, n) to (n, H, W, C)
        if count >= 1:
            image_set = np.concatenate((image_set, images), axis=0)  # concatenate all crown arrays to one image_set
            count = count + 1
        else:
            image_set = images
            count = count + 1
    del d, rois, params_rois
    return image_set

def hdf5_to_label(data_path):
    d = {}
    s = {}
    fn_list = os.listdir(data_path)
    count = 0
    for fn in fn_list:
        data, params_data = io.load_hdf5(data_path + "\\" + fn) # load hdf5 data from path
        crowns_human = data['crowns_human'] #get crown info with all terrestrial features
        d["labels{0}".format(count)] = crowns_human['features']['terrestrial']['sst'] # get stress level (0, 1, 2, 3, 4)
        s["species{0}".format(count)] = crowns_human['features']['terrestrial']['ba'] # get tree species
        labels = d["labels" + str(count)].copy() # copy to avoid memory error
        species = s["species" + str(count)].copy()
        labels = labels.reshape(len(labels),1) # reshape from (samples,) to (samples,1)
        species = species.reshape(len(species),1)
        if count >= 1:
            label_set = np.concatenate((label_set, labels), axis=0)  # concatenate all crown arrays to one label_set
            species_set = np.concatenate((species_set, species), axis=0)
            count = count + 1
        else:
            label_set = labels # define label_set with the first sets of labels
            species_set = species
            count = count + 1 
    del d, s, data, params_data, crowns_human
    return label_set, species_set


def hdf5_to_img_label(data_path, load_sets=None):
    d = {}
    l = {}
    s = {}
    fn_list = os.listdir(data_path)
    count = 0
    for fn in fn_list:
        # load hdf5 data from path
        rois, params_rois, data, params_data = exporter.load_rois_from_hdf5_v2(data_path + "\\" + fn,
                                                                               load_sets=load_sets) 
        # fetch images
        d["images{0}".format(count)] = rois["images_masked"] # assigns all crown arrays from the different hdf5 files to dictionary
        images = d["images" + str(count)].transpose(3,0,1,2) # array transpose shape from (H, W, C, n) to (n, H, W, C)
        
        # fetch features such as labels and species
        crowns_human = data['crowns_human'] #get crown info with all terrestrial features
        l["labels{0}".format(count)] = crowns_human['features']['terrestrial']['sst'] # get stress level (0, 1, 2, 3, 4)
        s["species{0}".format(count)] = crowns_human['features']['terrestrial']['ba'] # get tree species
        labels = l["labels" + str(count)].copy() # copy to avoid memory error
        species = s["species" + str(count)].copy()
        labels = labels.reshape(len(labels),1) # reshape from (samples,) to (samples,1)
        species = species.reshape(len(species),1)
        if count >= 1:
            image_set = np.concatenate((image_set, images), axis=0)  # concatenate all crown arrays to one image_set
            label_set = np.concatenate((label_set, labels), axis=0)  # concatenate all crown arrays to one label_set
            species_set = np.concatenate((species_set, species), axis=0)
            count = count + 1
        else:
            image_set = images
            label_set = labels # define label_set with the first sets of labels
            species_set = species
            count = count + 1
    del d, s, l, rois, params_rois, data, params_data, crowns_human
    return image_set, label_set, species_set