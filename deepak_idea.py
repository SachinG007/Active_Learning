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
        # if batch_idx % args.log_interval == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(data), len(train_loader.dataset),
        #         100. * batch_idx / len(train_loader), loss.item()))

def coreset(args, model, device, labelled_data, unlabelled_data, optimizer):

    global labelled_mask
    global unlabelled_mask

    dist_mat_sum = np.zeros((48500,48500))
    for itern in range(64):
        print(itern)

        # dist = torch.tensor([])
        # dist = dist.to(device)
        dist = []


        for batch_idx, (data, target) in enumerate(labelled_data):
            data, target = data.to(device), target.to(device)
            # optimizer.zero_grad()
            output,last_features = model(data)
            #extracting last layer features
            encoding = last_features[:,511,:,:]
            x,y,z = np.shape(encoding)
            #64 encodings of 32 (batch size) images 
            img_encoding = encoding.reshape(x,y*z)

            current_img_encoding = img_encoding[:,itern]
            new_data = (current_img_encoding.data).cpu().numpy()
            # dist = torch.cat([dist,current_img_encoding])
            # import pdb; pdb.set_trace()
            dist = np.append(dist,new_data)
            # import pdb; pdb.set_trace()

        x = np.shape(dist)
        num_labelled = x[0]
        print("labelled done")

        for batch_idx, (data, target) in enumerate(unlabelled_data):
            data, target = data.to(device), target.to(device)
            # optimizer.zero_grad()
            output,last_features = model(data)
            #extracting last layer features
            encoding = last_features[:,511,:,:]
            x,y,z = np.shape(encoding)
            #64 encodings of 32 (batch size) images 
            img_encoding = encoding.reshape(x,y*z)

            current_img_encoding = img_encoding[:,itern]
            new_data = (current_img_encoding.data).cpu().numpy()
            # dist = torch.cat([dist,current_img_encoding])
            # import pdb; pdb.set_trace()
            dist = np.append(dist,new_data)
            # import pdb; pdb.set_trace()
        print("unlabelled done")

        # import pdb; pdb.set_trace()
        x = np.shape(dist)
        dist_a = np.asarray(dist)
        dist_vec = np.reshape(dist_a,(x[0],1))
        dist_mat = np.matmul(dist_vec,dist_vec.transpose())
        x,y = np.shape(dist_mat)
        sq = np.array(dist_mat.diagonal()).reshape(x,1)
        dist_mat *= -2
        dist_mat+=sq
        dist_mat+=sq.transpose()
        print("debug 6")
        
        dist_mat_sum = np.add(dist_mat_sum,dist_mat)
        print("done this iter")

    
    import pdb; pdb.set_trace()
    xx,yy = np.shape(dist_mat_sum)

    useful_dist = dist_mat_sum[0:num_labelled,num_labelled:]
    b = np.amin(useful_dist, axis=0)
    import pdb; pdb.set_trace()
    min_norms = b.argsort()[:query]
    # print(min_norms)

    add_labels = [unlabelled_mask[i] for i in min_norms]
    labelled_mask = labelled_mask + add_labels
    unlabelled_mask = [x for x in unlabelled_mask if x not in add_labels]


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
        print('Random Numer 1 for test ', unlabelled_mask_rand[0])
        
        unlabelled_data = torch.utils.data.DataLoader(trainset, batch_size=1,
                                              sampler = SubsetRandomSampler(unlabelled_mask_rand), shuffle=False, num_workers=2)
        
        complete_unlabelled_data = torch.utils.data.DataLoader(trainset, batch_size=32,
                                              sampler = SubsetRandomSampler(unlabelled_mask), shuffle=False, num_workers=2)      

        test_data = torch.utils.data.DataLoader(testset, batch_size=10,
                                          sampler = None, shuffle=False, num_workers=2)


        model = content_encoder(10).to(device)
        # model = models.resnet50(pretrained=True)
        # num_ftrs = model.fc.in_features
        # model.fc = nn.Linear(num_ftrs, 10)
        model.cuda()
        # if active_learn_iter != 0:
        #     model.load_state_dict(torch.load("cifar_resnet.pt"))

        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

        for epoch in range(1, args.epochs + 1):
            print('epochs: ',epoch)
            train(args, model, device, labelled_data, optimizer, epoch)
        test(args, model, device, test_data)

        # active_learn(unlabelled_data,model)
        coreset(args, model, device, labelled_data, complete_unlabelled_data, optimizer)
        # rand_samp()
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