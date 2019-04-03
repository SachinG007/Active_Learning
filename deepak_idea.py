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


import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

labelled_mask  = list(range(0,500))
unlabelled_mask = list(range(2000, 50000))
unlabelled_mask_rand = []
query = 500
lm = len(labelled_mask)
um = len(unlabelled_mask)
print('len of labelled_mask: ',lm)
print('len of unlabelled_mask: ',um)
test_accs = []


def active_learn(unlabelled_data, model):
    print("active_learn")
    global labelled_mask
    global unlabelled_mask
    global unlabelled_mask_rand
    pert_norms = []
    for batch_idx, (data, target) in enumerate(unlabelled_data):
        # data, target = data.to(device), target.to(device)
        # import pdb;pdb.set_trace()
        rdata = np.reshape(data,(3,64,64))
        r, loop_i, label_orig, label_pert, pert_image = deepfool(rdata, model)
        #append the norm of the perturbation required to shift the image
        pert_norms.append(np.linalg.norm(r))
        # if(batch_idx%100==0):
        #     print(batch_idx)

    pert_norms = np.array(pert_norms)
    # print('len of total query deep fools ',len(pert_norms))
    min_norms = pert_norms.argsort()[:query]
    # print(min_norms)

    add_labels = [unlabelled_mask_rand[i] for i in min_norms]
    labelled_mask = labelled_mask + add_labels
    unlabelled_mask = [x for x in unlabelled_mask if x not in add_labels]
    # lm = len(labelled_mask)
    # um = len(unlabelled_mask)
    # print('len of labelled_mask: ',lm)
    # print('len of unlabelled_mask: ',um)

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output,last_features = model(data)

        output2 = output.reshape(output.shape[0],output.shape[1])
        # output_softmax = F.softmax(output)
        
        loss = F.cross_entropy(output2, target)
        loss.backward()
        optimizer.step()

def calc_perturbation(avg_pert_norms,all_data_list,model):

    for i in range(10):

        pert_norms_list = []
        pert_sum = 0; 
        for batch_idx, (data, target) in enumerate(all_data_list[i]):

            data, target = data.to(device), target.to(device)
            rdata = np.reshape(data,(3,64,64))

            r, loop_i, label_orig, label_pert, pert_image = deepfool(rdata, model)
            #r is a matrix, linalg.norm gets the l2 norm of matrix
            temp_val = np.linalg.norm(r)
            pert_sum += temp_val
            pert_norms_list.append(temp_val)

        pert_sum = pert_sum/50
        avg_pert_norms.append(pert_sum)



def active_learn_hier(all_data_list,model,active_learn_iter):

    list_of_pert_norms = []
    avg_pert_norms = []
    calc_perturbation(avg_pert_norms,all_data_list,model)
    print(avg_pert_norms)

    if active_learn_iter==0:


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output,last_features = model(data)
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
    jj = 1

    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
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
    
    trainset = datasets.CIFAR10('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.Resize((64,64), interpolation=2),
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))
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
        labelled_data = torch.utils.data.DataLoader(trainset, batch_size=32,
                                              sampler = SubsetRandomSampler(labelled_mask), shuffle=False, num_workers=2)
        global unlabelled_mask_rand
        # unlabelled_mask_rand = random.sample(unlabelled_mask, 2*query)
        unlabelled_mask_rand = np.random.choice(unlabelled_mask, 2*query)
        
        # unlabelled_data = torch.utils.data.DataLoader(trainset, batch_size=1,
        #                                       sampler = SubsetRandomSampler(unlabelled_mask_rand), shuffle=False, num_workers=2)

        test_data = torch.utils.data.DataLoader(testset, batch_size=10,
                                          sampler = None, shuffle=False, num_workers=2)


        model = content_encoder(10).to(device)
        model.cuda()

        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

        for epoch in range(1, args.epochs + 1):
            print('epochs: ',epoch)
            train(args, model, device, labelled_data, optimizer, epoch)
        test(args, model, device, test_data)

        active_learn_hier(dt.all_data_list,model,active_learn_iter)
        # active_learn(unlabelled_data,model)
        # coreset(args, model, device, labelled_data, complete_unlabelled_data, optimizer)
        # rand_samp()

        #some check prints
        print("active_learn over")
        lm = len(labelled_mask)
        um = len(unlabelled_mask)
        print('len of labelled_mask: ',lm)
        print('len of unlabelled_mask: ',um)

        # if (args.save_model):
        #     torch.save(model.state_dict(),"cifar_resnet.pt")

        active_learn_iter = active_learn_iter + 1

        with open('results_coreset_cifar_3april_%i*.txt'%jj, 'w') as f:   

            for item in test_accs:
                f.write("%s\n"%item)

        
if __name__ == '__main__':
    main()