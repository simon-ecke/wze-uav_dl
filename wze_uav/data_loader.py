import torch
import os
import findatree_roi.io as io
import findatree_roi.exporter as exporter
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, List, Tuple

class CustomDataset5Classes(Dataset):
    def __init__(self, data, labels, class_names=List[str], class_idx=dict, species=None, kkl=None, transform=None):
        self.data = data
        self.labels = torch.from_numpy(labels)
        self.species = species
        self.kkl = kkl
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
        #adjust labels to 5 classes
        if label <= 10: # healthy trees with labels 0 - 10 NBV
            label = 0 
        elif label > 10 and label <= 25: # stressed trees with labels 15 - 25 NBV
            label = 1
        elif label > 25 and label <= 60:
            label = 2
        elif label > 60 and label < 99:
            label = 3
        else: # dead trees with label 99-100 NBV
            label = 4
        
        if self.transform:
            return self.transform(img), label # returns image (transformed), label
        else:
            return img, label # returns image, label
        
class CustomDataset3Classes(Dataset):
    def __init__(self, data, labels, class_names=List[str], class_idx=dict, species=None, kkl=None, transform=None):
        self.data = data
        self.labels = torch.from_numpy(labels) #important: these are the original labels! they are changed to adapt to 3 classes below
        self.species = species
        self.kkl = kkl
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
        if label <= 45: # healthy trees with labels 0 - 45 NBV
            label = 0 
        elif label > 45 and label < 99: # stressed trees with labels 50 - 98 NBV
            label = 1
        else: # dead trees with label 99-100 NBV
            label = 2
        
        if self.transform:
            return self.transform(img), label # returns image (transformed), label
        else:
            return img, label # returns image, label
        
class CustomDataset4Classes(Dataset):
    def __init__(self, data, labels, class_names=List[str], class_idx=dict, species=None, kkl=None, transform=None):
        self.data = data
        self.labels = torch.from_numpy(labels) #important: these are the original labels! they are changed to adapt to 3 classes below
        self.species = species
        self.kkl = kkl
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
        if label <= 25: # healthy trees with labels 0 - 25 NBV
            label = 0 
        elif label > 25 and label <= 60: # moderately stressed trees with labels 30 - 60 NBV
            label = 1
        elif label > 60 and label < 99: #  highly stressed trees with labels 65-95 NBV
            label = 2
        else: # dead trees with labels 99-100 NBV
            label = 3
        
        if self.transform:
            return self.transform(img), label # returns image (transformed), label
        else:
            return img, label # returns image, label


def hdf5_to_img_label(data_path, load_sets=None):
    d = {} # empty dict for the imagery
    l = {} # empty dict for the labels (needle or leafloss)
    s = {} # empty dict for the tree species
    k = {} # empty dict for the kraftsche class (e.g. if a tree is dominant or not) 
    b = {} # empty dict for the tree class
    fn_list = os.listdir(data_path)
    count = 0
    for fn in fn_list:
        # load hdf5 data from path
        rois, params_rois, data, params_data = exporter.load_rois_from_hdf5_v2(data_path + "\\" + fn,
                                                                               load_sets=load_sets) 
        # fetch images
        d["images{0}".format(count)] = rois["images_masked"] # assigns all crown arrays from the different hdf5 files to dictionary
        images = d["images" + str(count)].transpose(3,0,1,2) # array transpose shape from (H, W, C, n) to (n, H, W, C)
        
        # fetch terrestrial features (such as labels and species etc.)
        crowns_human = data['crowns_human'] #get crown info with all terrestrial features
        l["labels{0}".format(count)] = crowns_human['features']['terrestrial']['nbv'] # get stress level (0, 1, 2, 3, 4)
        s["species{0}".format(count)] = crowns_human['features']['terrestrial']['ba'] # get tree species
        k["kkl{0}".format(count)] = crowns_human['features']['terrestrial']['kkl']
        b["bk{0}".format(count)] = crowns_human['features']['terrestrial']['bk']
        labels = l["labels" + str(count)].copy() # copy to avoid memory error
        species = s["species" + str(count)].copy()
        kkl = k["kkl" + str(count)].copy()
        bk = b["bk" + str(count)].copy()
        labels = labels.reshape(len(labels),1) # reshape from (samples,) to (samples,1)
        species = species.reshape(len(species),1)
        kkl = kkl.reshape(len(kkl),1)
        bk = bk.reshape(len(bk),1)
        if count >= 1:
            image_set = np.concatenate((image_set, images), axis=0)  # concatenate all crown arrays to one image_set
            label_set = np.concatenate((label_set, labels), axis=0)  # concatenate all crown arrays to one label_set
            species_set = np.concatenate((species_set, species), axis=0)
            kkl_set = np.concatenate((kkl_set, kkl), axis=0)
            bk_set = np.concatenate((bk_set, bk), axis=0)
            count = count + 1
        else:
            image_set = images
            label_set = labels # define label_set with the first sets of labels
            species_set = species
            kkl_set = kkl
            bk_set = bk
            count = count + 1
    
    # filter data depending on terrestrial features
    np_filter = []
    for i in range(0, len(bk_set)):
        if kkl_set[i] > 3:
            np_filter.append(False) 
        elif bk_set[i] <= 1:
            np_filter.append(True)
        elif bk_set[i] >= 320 and bk_set[i] <= 340:
            np_filter.append(True)
        else:
            np_filter.append(False)
        
    image_set = image_set[np_filter]
    label_set = label_set[np_filter]
    species_set = species_set[np_filter]
    kkl_set = kkl_set[np_filter]
    bk_set = bk_set[np_filter]
        
    del d, s, l, k, b, rois, params_rois, data, params_data, crowns_human
    return image_set, label_set, species_set, kkl_set, bk_set