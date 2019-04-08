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
import load_various_classes as dt
from itertools import chain

import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

labelled_mask  = list(range(0,500))
unlabelled_mask = list(range(2000, 50000))
prev_avg_pert_norms = []

list_unlab_data_mask = []
for i in range(10):
    unlab_data_mask  = list(range(51,5000))
    list_unlab_data_mask.append([unlab_data_mask])

list_lab_data_mask = []
for i in range(10):
    lab_data_mask  = list(range(1,50))
    list_lab_data_mask.append([lab_data_mask])

list_selected_mask = []

query = 500
class_query = 50
lm = len(labelled_mask)
um = len(unlabelled_mask)
print('len of labelled_mask: ',lm)
print('len of unlabelled_mask: ',um)
test_accs = []


def train(args, model, device, all_lab_data_list, optimizer, epoch):

    model.train()
    for j in range(10):

        for batch_idx, (data, target) in enumerate(all_lab_data_list[j]):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
        
            output = model(data)

            output2 = output.reshape(output.shape[0],output.shape[1])
            # output_softmax = F.softmax(output)
            
            loss = F.cross_entropy(output2, target)
            loss.backward()
            optimizer.step()

def calc_perturbation(list_list_pert,avg_pert_norms,all_unlab_data_list,model, device):

    for i in range(10):
        print(i)
        pert_norms_list = []
        pert_sum = 0; 
        for batch_idx, (data, target) in enumerate(all_unlab_data_list[i]):
            print(batch_idx)
            data, target = data.to(device), target.to(device)
            rdata = np.reshape(data,(3,64,64))

            r, loop_i, label_orig, label_pert, pert_image = deepfool(rdata, model)
            #r is a matrix, linalg.norm gets the l2 norm of matrix
            temp_val = np.linalg.norm(r)
            pert_sum += temp_val
            pert_norms_list.append(temp_val)

        pert_sum = pert_sum/(2*class_query)
        avg_pert_norms.append(pert_sum)
        list_list_pert.append([pert_norms_list])



def active_learn_hier(all_unlab_data_list,model,active_learn_iter, device):

    global list_unlab_data_mask
    global list_lab_data_mask
    global list_selected_mask
    global prev_avg_pert_norms
    query = 500
    class_query = 50
    list_list_pert = []
    avg_pert_norms = []

    calc_perturbation(list_list_pert,avg_pert_norms,all_unlab_data_list,model, device)
    print(avg_pert_norms)

    if active_learn_iter==0:

        prev_avg_pert_norms = avg_pert_norms

        for j in range(10):
            jth_list = np.array(list_list_pert[j])
            min_norms = jth_list[0].argsort()[:class_query]
            print("query size ",class_query)
            print("min_norms size ", np.size(min_norms))
            
            tmp_list = list(chain.from_iterable(list_selected_mask[j]))
            tmp_arr = np.asarray(tmp_list)
            
            # add_labels = [tmp_arr[i] for i in min_norms]
            add_labels = np.take(tmp_arr,min_norms)
            add_labels_l = add_labels.tolist()
            list_lab_data_mask[j][0] = list_lab_data_mask[j][0] + add_labels_l
            list_unlab_data_mask[j][0] = [x for x in list_unlab_data_mask[j][0] if x not in add_labels_l]

    else:

        ec = np.array(avg_pert_norms)
        ep = np.array(prev_avg_pert_norms)
        prev_avg_pert_norms = avg_pert_norms
        change_avg_pert_norms = ec - ep 
        arr_sum = np.sum(change_avg_pert_norms)
        arr_sum = arr_sum.astype(float)
        change_avg_pert_norms = change_avg_pert_norms/arr_sum

        for j in range(10):
            jth_list = np.array(list_list_pert[j])
            class_query = np.floor(query * change_avg_pert_norms[j])
            class_query = class_query.astype(int)
            print("query from class: ",j, "is: ",class_query)

            min_norms = jth_list[0].argsort()[:class_query]
            
            tmp_list = list(chain.from_iterable(list_selected_mask[j]))
            tmp_arr = np.asarray(tmp_list)
            
            # add_labels = [tmp_arr[i] for i in min_norms]
            add_labels = np.take(tmp_arr,min_norms)
            add_labels_l = add_labels.tolist()
            print(np.size(add_labels_l))
            if np.size(add_labels_l)!=0:

                list_lab_data_mask[j][0] = list_lab_data_mask[j][0] + add_labels_l
                print("Modified lab data list size ",np.size(list_lab_data_mask[j][0]))
                list_unlab_data_mask[j][0] = [x for x in list_unlab_data_mask[j][0] if x not in add_labels_l]



def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            output2 = output.reshape(output.shape[0],output.shape[1])
            # output_softmax = F.softmax(output)
            test_loss += F.cross_entropy(output2, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_acc = 100. * correct / len(test_loader.dataset)
    test_accs.append(test_acc)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    # Training settings
    # np.random.seed(2)

    # i = sys.argv[2]
    # jj = int(i,10)
    # print(jj*10)
    global list_unlab_data_mask
    global list_lab_data_mask
    global list_selected_mask
    global prev_avg_pert_norms
    jj = 1

    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=2, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--iterNum', type=int, default=1, metavar='S',
                        help='iter num')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    # torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    
    testset = datasets.CIFAR10('../data', train=False, download=True,
                   transform=transforms.Compose([
                       transforms.Resize((64,64), interpolation=2),
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))



    active_learn_iter = 0
    print("Starting Active learning")

    while active_learn_iter<30:


        print('Active learning Iter: ', active_learn_iter)

        # kwargs = {'num_workers': 2, 'pin_memory': False}

        #labelled training data
        lab_plane_data   = torch.utils.data.DataLoader(dt.plane_trainset, batch_size=32, sampler = SubsetRandomSampler(list(chain.from_iterable(list_lab_data_mask[0]))), shuffle=False , **kwargs)
        lab_car_data   = torch.utils.data.DataLoader(dt.car_trainset, batch_size=32, sampler = SubsetRandomSampler(list(chain.from_iterable(list_lab_data_mask[1]))),  shuffle=False , **kwargs)
        lab_bird_data   = torch.utils.data.DataLoader(dt.bird_trainset, batch_size=32, sampler = SubsetRandomSampler(list(chain.from_iterable(list_lab_data_mask[2]))), shuffle=False , **kwargs)
        lab_cat_data   = torch.utils.data.DataLoader(dt.cat_trainset, batch_size=32, sampler = SubsetRandomSampler(list(chain.from_iterable(list_lab_data_mask[3]))), shuffle=False , **kwargs)
        lab_deer_data   = torch.utils.data.DataLoader(dt.deer_trainset, batch_size=32, sampler = SubsetRandomSampler(list(chain.from_iterable(list_lab_data_mask[4]))), shuffle=False , **kwargs)
        lab_dog_data   = torch.utils.data.DataLoader(dt.dog_trainset, batch_size=32, sampler = SubsetRandomSampler(list(chain.from_iterable(list_lab_data_mask[5]))), shuffle=False , **kwargs)
        lab_frog_data   = torch.utils.data.DataLoader(dt.frog_trainset, batch_size=32, sampler = SubsetRandomSampler(list(chain.from_iterable(list_lab_data_mask[6]))), shuffle=False , **kwargs)
        lab_horse_data   = torch.utils.data.DataLoader(dt.horse_trainset, batch_size=32, sampler = SubsetRandomSampler(list(chain.from_iterable(list_lab_data_mask[7]))), shuffle=False , **kwargs)
        lab_ship_data   = torch.utils.data.DataLoader(dt.ship_trainset, batch_size=32, sampler = SubsetRandomSampler(list(chain.from_iterable(list_lab_data_mask[8]))), shuffle=False , **kwargs)
        lab_truck_data   = torch.utils.data.DataLoader(dt.truck_trainset, batch_size=32, sampler = SubsetRandomSampler(list(chain.from_iterable(list_lab_data_mask[9]))), shuffle=False , **kwargs)

        all_lab_data_list = [lab_plane_data , lab_car_data, lab_bird_data, lab_cat_data, lab_deer_data, lab_dog_data, lab_frog_data, lab_horse_data, lab_ship_data, lab_truck_data, ]
        list_selected_mask = []
        for i in range(10):
            selected_data_mask = np.random.choice(list(chain.from_iterable(list_unlab_data_mask[i])), 2*class_query)
            list_selected_mask.append([selected_data_mask])

        # Create datasetLoaders from trainset and testset
        # classDict = {'plane':0, 'car':1, 'bird':2, 'cat':3, 'deer':4, 'dog':5, 'frog':6, 'horse':7, 'ship':8, 'truck':9}
        plane_data   = torch.utils.data.DataLoader(dt.plane_trainset, batch_size=1, sampler = SubsetRandomSampler(list(chain.from_iterable(list_selected_mask[0]))), shuffle=False , **kwargs)
        car_data   = torch.utils.data.DataLoader(dt.car_trainset, batch_size=1, sampler = SubsetRandomSampler(list(chain.from_iterable(list_selected_mask[1]))),  shuffle=False , **kwargs)
        bird_data   = torch.utils.data.DataLoader(dt.bird_trainset, batch_size=1, sampler = SubsetRandomSampler(list(chain.from_iterable(list_selected_mask[2]))), shuffle=False , **kwargs)
        cat_data   = torch.utils.data.DataLoader(dt.cat_trainset, batch_size=1, sampler = SubsetRandomSampler(list(chain.from_iterable(list_selected_mask[3]))), shuffle=False , **kwargs)
        deer_data   = torch.utils.data.DataLoader(dt.deer_trainset, batch_size=1, sampler = SubsetRandomSampler(list(chain.from_iterable(list_selected_mask[4]))), shuffle=False , **kwargs)
        dog_data   = torch.utils.data.DataLoader(dt.dog_trainset, batch_size=1, sampler = SubsetRandomSampler(list(chain.from_iterable(list_selected_mask[5]))), shuffle=False , **kwargs)
        frog_data   = torch.utils.data.DataLoader(dt.frog_trainset, batch_size=1, sampler = SubsetRandomSampler(list(chain.from_iterable(list_selected_mask[6]))), shuffle=False , **kwargs)
        horse_data   = torch.utils.data.DataLoader(dt.horse_trainset, batch_size=1, sampler = SubsetRandomSampler(list(chain.from_iterable(list_selected_mask[7]))), shuffle=False , **kwargs)
        ship_data   = torch.utils.data.DataLoader(dt.ship_trainset, batch_size=1, sampler = SubsetRandomSampler(list(chain.from_iterable(list_selected_mask[8]))), shuffle=False , **kwargs)
        truck_data   = torch.utils.data.DataLoader(dt.truck_trainset, batch_size=1, sampler = SubsetRandomSampler(list(chain.from_iterable(list_selected_mask[9]))), shuffle=False , **kwargs)

        all_unlab_data_list = [plane_data , car_data, bird_data, cat_data, deer_data, dog_data, frog_data, horse_data, ship_data, truck_data]

        test_data = torch.utils.data.DataLoader(testset, batch_size=10,
                                          sampler = None, shuffle=False, num_workers=2)


        model = content_encoder(10).to(device)
        model.cuda()

        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

        for epoch in range(1, args.epochs + 1):
            print('epochs: ',epoch)
            train(args, model, device, all_lab_data_list, optimizer, epoch)
        test(args, model, device, test_data)

        active_learn_hier(all_unlab_data_list,model,active_learn_iter,device)

        #some check prints
        for kk in range(10):
            list_unlab_data_mask[kk][0] = [x for x in list_unlab_data_mask[kk][0] if x not in list_lab_data_mask[kk][0]]
            print(np.size(list_lab_data_mask[kk]))
            print(np.size(list_unlab_data_mask[kk]))

        # if (args.save_model):
        #     torch.save(model.state_dict(),"cifar_resnet.pt")

        active_learn_iter = active_learn_iter + 1

        with open('results_hierar_8april%i.txt'%jj, 'w') as f:   

            for item in test_accs:
                f.write("%s\n"%item)

        
if __name__ == '__main__':
    main()