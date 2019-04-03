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

# Create datasetLoaders from trainset and testset
# classDict = {'plane':0, 'car':1, 'bird':2, 'cat':3, 'deer':4, 'dog':5, 'frog':6, 'horse':7, 'ship':8, 'truck':9}
plane_data   = nl.DataLoader(plane_trainset, batch_size=32, shuffle=True , **kwargs)
car_data   = nl.DataLoader(car_trainset, batch_size=32, shuffle=True , **kwargs)
bird_data   = nl.DataLoader(bird_trainset, batch_size=32, shuffle=True , **kwargs)
cat_data   = nl.DataLoader(cat_trainset, batch_size=32, shuffle=True , **kwargs)
deer_data   = nl.DataLoader(deer_trainset, batch_size=32, shuffle=True , **kwargs)
dog_data   = nl.DataLoader(dog_trainset, batch_size=32, shuffle=True , **kwargs)
frog_data   = nl.DataLoader(frog_trainset, batch_size=32, shuffle=True , **kwargs)
horse_data   = nl.DataLoader(horse_trainset, batch_size=32, shuffle=True , **kwargs)
ship_data   = nl.DataLoader(ship_trainset, batch_size=32, shuffle=True , **kwargs)
truck_data   = nl.DataLoader(truck_trainset, batch_size=32, shuffle=True , **kwargs)
