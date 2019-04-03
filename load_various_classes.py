from __future__ import print_function
import argparse
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision.models as models
from torch.utils.data.sampler import Sampler,SubsetRandomSampler
from deep_fool import deepfool
from vgg import content_encoder
import random
import sys
import cifar10_subset as nl


import os
# classDict = {'plane':0, 'car':1, 'bird':2, 'cat':3, 'deer':4, 'dog':5, 'frog':6, 'horse':7, 'ship':8, 'truck':9}
# Let's choose cats (class 3 of CIFAR) and dogs (class 5 of CIFAR) as trainset/testset
plane_trainset = \
    nl.DatasetMaker(
        [nl.get_class_i(nl.x_train, nl.y_train, nl.classDict['plane'])],
        nl.transform_with_aug
    )

car_trainset = \
    nl.DatasetMaker(
        [nl.get_class_i(nl.x_train, nl.y_train, nl.classDict['car'])],
        nl.transform_with_aug
    )

bird_trainset = \
    nl.DatasetMaker(
        [nl.get_class_i(nl.x_train, nl.y_train, nl.classDict['bird'])],
        nl.transform_with_aug
    )

cat_trainset = \
    nl.DatasetMaker(
        [nl.get_class_i(nl.x_train, nl.y_train, nl.classDict['cat'])],
        nl.transform_with_aug
    )

deer_trainset = \
    nl.DatasetMaker(
        [nl.get_class_i(nl.x_train, nl.y_train, nl.classDict['deer'])],
        nl.transform_with_aug
    )

dog_trainset = \
    nl.DatasetMaker(
        [nl.get_class_i(nl.x_train, nl.y_train, nl.classDict['dog'])],
        nl.transform_with_aug
    )
    
frog_trainset = \
    nl.DatasetMaker(
        [nl.get_class_i(nl.x_train, nl.y_train, nl.classDict['frog'])],
        nl.transform_with_aug
    )
    
horse_trainset = \
    nl.DatasetMaker(
        [nl.get_class_i(nl.x_train, nl.y_train, nl.classDict['horse'])],
        nl.transform_with_aug
    )
    
ship_trainset = \
    nl.DatasetMaker(
        [nl.get_class_i(nl.x_train, nl.y_train, nl.classDict['ship'])],
        nl.transform_with_aug
    )
    
truck_trainset = \
    nl.DatasetMaker(
        [nl.get_class_i(nl.x_train, nl.y_train, nl.classDict['truck'])],
        nl.transform_with_aug
    )
    

kwargs = {'num_workers': 2, 'pin_memory': False}

query = 50
data_mask  = list(range(0,5000))
selected_data_mask = np.random.choice(data_mask, 2*query)

# Create datasetLoaders from trainset and testset
# classDict = {'plane':0, 'car':1, 'bird':2, 'cat':3, 'deer':4, 'dog':5, 'frog':6, 'horse':7, 'ship':8, 'truck':9}
plane_data   = torch.utils.data.DataLoader(plane_trainset, batch_size=1, sampler = SubsetRandomSampler(selected_data_mask), shuffle=False , **kwargs)
car_data   = torch.utils.data.DataLoader(car_trainset, batch_size=1, sampler = SubsetRandomSampler(selected_data_mask),  shuffle=False , **kwargs)
bird_data   = torch.utils.data.DataLoader(bird_trainset, batch_size=1, sampler = SubsetRandomSampler(selected_data_mask), shuffle=False , **kwargs)
cat_data   = torch.utils.data.DataLoader(cat_trainset, batch_size=1, sampler = SubsetRandomSampler(selected_data_mask), shuffle=False , **kwargs)
deer_data   = torch.utils.data.DataLoader(deer_trainset, batch_size=1, sampler = SubsetRandomSampler(selected_data_mask), shuffle=False , **kwargs)
dog_data   = torch.utils.data.DataLoader(dog_trainset, batch_size=1, sampler = SubsetRandomSampler(selected_data_mask), shuffle=False , **kwargs)
frog_data   = torch.utils.data.DataLoader(frog_trainset, batch_size=1, sampler = SubsetRandomSampler(selected_data_mask), shuffle=False , **kwargs)
horse_data   = torch.utils.data.DataLoader(horse_trainset, batch_size=1, sampler = SubsetRandomSampler(selected_data_mask), shuffle=False , **kwargs)
ship_data   = torch.utils.data.DataLoader(ship_trainset, batch_size=1, sampler = SubsetRandomSampler(selected_data_mask), shuffle=False , **kwargs)
truck_data   = torch.utils.data.DataLoader(truck_trainset, batch_size=1, sampler = SubsetRandomSampler(selected_data_mask), shuffle=False , **kwargs)

all_data_list = [plane_data , car_data, bird_data, car_data, deer_data, frog_trainset, horse_data, ship_data, truck_data]

# import pdb;pdb.set_trace()