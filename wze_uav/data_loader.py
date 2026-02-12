"""data_loader.py - Helper functions for importing and filtering images, lables and other features from hdf5 files as ndarrays, defining custom PyTorch Datasets, creating unique tree IDs or getting the plot IDs from the data and for defining a custom sampler for oversampling the minority classes.
"""

# Author: Simon Ecke
# Date: 15.03.2023
# Code Repository: https://github.com/simon-ecke/wze-uav_dl
#

import torch
import os
import findatree_roi.io as io
import findatree_roi.exporter as exporter
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, List, Tuple
from torch.utils.data import WeightedRandomSampler
from tqdm.auto import tqdm
from collections import defaultdict
import hashlib


######################################
####### PyTorch Custom Dataset #######
######################################

# Classes for train and validation datasets
class CustomDataset(Dataset):
    def __init__(self, data, labels, class_names=None, species=None, kkl=None, transform=None):
        self.data = data
        #important: these are the original labels representing the needle or leafloss! They are changed below to 3 classes below!
        self.labels = labels
        self.species = species
        self.kkl = kkl
        self.class_names = class_names
        self.transform = transform

    # overwrite __len__()
    def __len__(self) -> int:
        "Returns total number of samples"
        return len(self.data)
    
    # overwrite __getitem__() -> returns image and label 
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, float]:
        img = self.data[index]
        label = self.labels[index]
        #label = label.squeeze(dim=0)
        label = int(label)
  
        if self.transform:
            return self.transform(img), label # returns image (transformed), label
        else:
            return img, label # returns image, label
        

# Class to be used for the 11 classes example. The dataset is filtered so that the exact same images are used here for better comparison
class FilteredDataset(Dataset):
    def __init__(self, data, labels, species=None, class_names=None, kkl=None, transform=None):
        self.data = data
        self.species = species
        self.class_names = class_names
        self.kkl = kkl
        self.transform = transform

        self.filtered_indices, self.filtered_labels = self._filter_indices_and_labels(labels)
        
    def _filter_indices_and_labels(self, labels):
        filtered_indices = []
        filtered_labels = []
        for i, label in enumerate(labels):
            if label not in [10, 11, 12]:
                filtered_indices.append(i)
                if label == 13:
                    filtered_labels.append(10)
                else:
                    filtered_labels.append(label)

        return filtered_indices, np.array(filtered_labels, dtype=object)
    
    
    def __len__(self) -> int:
        return len(self.filtered_indices)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, int]:
        filtered_index = self.filtered_indices[index]
        img = self.data[filtered_index]
        label = self.filtered_labels[index]
        species = self.species[filtered_index]
        label = int(label)
        species = int(species)

        if self.transform:
            return self.transform(img), label
        else:
            return img, label

    @property
    def labels(self):
        return np.array(self.filtered_labels.tolist(), dtype=np.int32).reshape(-1, 1)


# classes for test dataset
class CustomTestDataset(Dataset):
    def __init__(self, data, labels, species, class_names=None, kkl=None, transform=None):
        self.data = data
        #important: these are the original labels representing the needle or leafloss! They are changed below to 3 classes below!
        self.labels = labels
        self.species = species
        self.kkl = kkl
        self.class_names = class_names
        self.transform = transform

    # overwrite __len__()
    def __len__(self) -> int:
        "Returns total number of samples"
        return len(self.data)
    
    # overwrite __getitem__() -> returns image and label 
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, float]:
        img = self.data[index]
        label = self.labels[index]
        #label = label.squeeze(dim=0)
        label = int(label)
        species = self.species[index]
        species = int(species)
  
        if self.transform:
            return self.transform(img), label, species # returns image (transformed), label and species
        else:
            return img, label, species # returns image, label and species
        
        
class FilteredTestDataset(Dataset):
    def __init__(self, data, labels, species=None, class_names=None, kkl=None, transform=None):
        self.data = data
        self.species = species
        self.class_names = class_names
        self.kkl = kkl
        self.transform = transform

        self.filtered_indices, self.filtered_labels = self._filter_indices_and_labels(labels)

    def _filter_indices_and_labels(self, labels):
        filtered_indices = []
        filtered_labels = []
        for i, label in enumerate(labels):
            if label not in [10, 11, 12]:
                filtered_indices.append(i)
                if label == 13:
                    filtered_labels.append(10)
                else:
                    filtered_labels.append(label)

        return filtered_indices, np.array(filtered_labels, dtype=object)

    def __len__(self) -> int:
        return len(self.filtered_indices)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, int]:
        filtered_index = self.filtered_indices[index]
        img = self.data[filtered_index]
        label = self.filtered_labels[index]
        species = self.species[filtered_index]
        label = int(label)
        species = int(species)

        if self.transform:
            return self.transform(img), label, species
        else:
            return img, label, species

    @property
    def labels(self):
        return np.array(self.filtered_labels.tolist(), dtype=np.int32).reshape(-1, 1)

        
###############################        
###### Import hdf5 files ######        
###############################        

def hdf5_to_img_label(path_list, hashID_dict:dict, load_sets=None):
    count = 0
    for fn in tqdm(path_list, desc="Processing hdf5 datasets"):
        # load hdf5 data from path
        rois, params_rois, data, params_data = exporter.load_rois_from_hdf5_v2(fn,
                                                                               load_sets=load_sets) 
        # fetch images
        images = rois["images_masked"] # assigns all crown arrays from the different hdf5 files to dictionary
        images = images.transpose(3,0,1,2) # array transpose shape from (H, W, C, n) to (n, H, W, C)
        
        # fetch terrestrial features 
        crowns_human = data['crowns_human'] #get crown info with all terrestrial features
        labels = crowns_human['features']['terrestrial']['nbv'] # nbv (Nadel- und Blattverlust = defolition percentage)
        species = crowns_human['features']['terrestrial']['ba'] # ba (Baumart = tree species) 
        kkl = crowns_human['features']['terrestrial']['kkl'] # classes accoring to Kraft
        bk = crowns_human['features']['terrestrial']['bk'] # classes according to the ground-based assessment
        labels = labels.copy() # copy to avoid memory error
        species = species.copy()
        kkl = kkl.copy()
        bk = bk.copy()
        labels = labels.reshape(len(labels),1) # reshape from (samples,) to (samples,1)
        species = species.reshape(len(species),1)
        kkl = kkl.reshape(len(kkl),1)
        bk = bk.reshape(len(bk),1)
        if count >= 1:
            image_set = np.concatenate((image_set, images), axis=0)  # concatenate all crown arrays to one image_set
            label_set = np.concatenate((label_set, labels), axis=0)  
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
    
    # get hash_id
    hash_id = []        
    for key, value in hashID_dict.items():
        hash_id.append(value)
        
    hash_id = np.array(hash_id)  
    #hash_id = hash_id.reshape((-1, 1)) # from (samples,) to (samples,1)
    
    # filter data depending on terrestrial features
    np_filter = []
    for i in range(0, len(bk_set)):
        if kkl_set[i] <= 3 and (bk_set[i] <= 1 or (bk_set[i] >= 320 and bk_set[i] <= 340)):
            np_filter.append(True)
        else:
            np_filter.append(False)
        
    image_set = image_set[np_filter]
    label_set = label_set[np_filter]
    species_set = species_set[np_filter]
    kkl_set = kkl_set[np_filter]
    bk_set = bk_set[np_filter]
    hash_id = hash_id[np_filter]
        
    del rois, params_rois, data, params_data, crowns_human
    return image_set, label_set, species_set, kkl_set, bk_set, hash_id


##########################################################
###### Create a unique ID for every individual tree ######
##########################################################


def get_unique_treeID(path_list: str):
    enr = {}
    bnr = {}
    sat = {}
    all_id = {}
    count = 0
    for fn in tqdm(path_list, desc="Creating unique tree IDs..."):
        # load hdf5 data from path
        rois, params_rois, data, params_data = exporter.load_rois_from_hdf5_v2(fn,
                                                                               load_sets=["images_masked"])
       
        # fetch terrestrial features (such as labels and species etc.)
        tnr = fn.rsplit("\\",1)[1].split("_",1)[0].split("r",1)[1]
        crowns_human = data['crowns_human'] #get crown info with all terrestrial features
        enr = crowns_human['features']['terrestrial']['enr']
        sat = crowns_human['features']['terrestrial']['sat'] 
        bnr = crowns_human['features']['terrestrial']['bnr'] 
        year = fn.rsplit("\\", 2)[1]
        for i in range(len(enr)):
            all_id[count] = {'tnr': tnr, 'enr': enr[i], 'sat': sat[i], 'bnr': bnr[i], 'year': year}
            count += 1
            
      # First, group the data by the values of tnr, enr, sat, and bnr
        grouped_data = defaultdict(list)
        for idx, item in all_id.items():
            key = (item['tnr'], item['enr'], item['sat'], item['bnr'])
            grouped_data[key].append((idx, item))
        
        # Then, create the final dictionary and calculate the hash IDs
        final_dict = {}
        for key, group in grouped_data.items():
            hash_id = hashlib.sha1(str(key).encode('utf-8')).hexdigest()[:8]
            final_dict[hash_id] = [(idx, item['year']) for idx, item in group]
    
    # Update the new_dict dictionary comprehension to include the year
    new_dict = {idx: (final_dict[hash_id], final_dict[hash_id][idx][1]) for hash_id in final_dict for idx in range(len(final_dict[hash_id]))}
    hashID_dict = dict(new_dict.items())

    for hashID, index_year_list in final_dict.items():
        for index, year in index_year_list:
            hashID_dict[index] = (hashID, year)
        
    hashID_dict = dict(sorted(hashID_dict.items(), key=lambda x: x[0]))
    
    return hashID_dict


def get_plotID(path_list: str):
    plotID_dict = {}
    count = 0
    for fn in tqdm(path_list, desc="Creating unique tree IDs..."):
        # load hdf5 data from path
        rois, params_rois, data, params_data = exporter.load_rois_from_hdf5_v2(fn,
                                                                               load_sets=["images_masked"])
       
        # fetch terrestrial features (such as labels and species etc.)
        tnr = fn.rsplit("\\",1)[1].split("_",1)[0].split("r",1)[1]
        crowns_human = data['crowns_human'] #get crown info with all terrestrial features
        enr = crowns_human['features']['terrestrial']['enr']
        year = fn.rsplit("\\", 2)[1]
        for i in range(len(enr)):
            plotID_dict[count] = (tnr, year)
            count += 1
    
    return plotID_dict


#############################################################
###### Define a sampler to account for class imbalance ######
#############################################################

# sampler for oversampling of the minority classes and used in the dataloader
def data_sampler(dataset: Dataset, class_names: List[str]):
    
    labels_list = dataset.labels.tolist()
    label_list = []
    for label in labels_list:
        label = label[0]
        label_list.append(label)
    
    # count the number of samples per class
    class_sample_counts = [label_list.count(i) for i in range(len(class_names))]
    
    # compute the effective number of samples
    total_samples = sum(class_sample_counts)
    effective_num = [(1.0 - (count / total_samples)) / (len(class_names) - 1) for count in class_sample_counts]
    
    # calculate the weight of each class as inverse of frequency
    class_weights = torch.FloatTensor(effective_num)
    class_weights /= torch.sum(class_weights)
    
    # create a list of weights for each data sample
    train_weights = [class_weights[label] for label in label_list]
    
    # create a sampler using the weights
    sampler = WeightedRandomSampler(train_weights, len(dataset), replacement=True)
 
    return sampler