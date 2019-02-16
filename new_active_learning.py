from __future__ import print_function
import argparse
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data.sampler import Sampler,SubsetRandomSampler
from deep_fool import deepfool
import random

import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

labelled_mask  = list(range(1000,1010))
unlabelled_mask = list(range(1010, 1300))
lm = len(labelled_mask)
um = len(unlabelled_mask)
print('len of labelled_mask: ',lm)
print('len of unlabelled_mask: ',um)
test_accs = []

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def rand_samp():
    
    global labelled_mask
    global unlabelled_mask
    add_labels = random.sample(unlabelled_mask, 20)
    labelled_mask = labelled_mask + add_labels
    unlabelled_mask = [x for x in unlabelled_mask if x not in add_labels]

def active_learn(unlabelled_data, model):
    print("active_learn")
    global labelled_mask
    global unlabelled_mask
    pert_norms = []
    for batch_idx, (data, target) in enumerate(unlabelled_data):
        # data, target = data.to(device), target.to(device)
        # import pdb;pdb.set_trace()
        rdata = np.reshape(data,(1,28,28))
        r, loop_i, label_orig, label_pert, pert_image = deepfool(rdata, model)
        #append the norm of the perturbation required to shift the image
        pert_norms.append(np.linalg.norm(r))

    pert_norms = np.array(pert_norms)
    min_norms = pert_norms.argsort()[:20]

    add_labels = [unlabelled_mask[i] for i in min_norms]
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
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        # if batch_idx % args.log_interval == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(data), len(train_loader.dataset),
        #         100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
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
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    
    trainset = datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))
    testset = datasets.MNIST('../data', train=False, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))

    active_learn_iter = 0

    while active_learn_iter<10:


        labelled_data = torch.utils.data.DataLoader(trainset, batch_size=10,
                                              sampler = SubsetRandomSampler(labelled_mask), shuffle=False, num_workers=2)
        unlabelled_data = torch.utils.data.DataLoader(trainset, batch_size=1,
                                              sampler = SubsetRandomSampler(unlabelled_mask), shuffle=False, num_workers=2)
        test_data = torch.utils.data.DataLoader(testset, batch_size=10,
                                          sampler = None, shuffle=False, num_workers=2)


        model = Net().to(device)
        if active_learn_iter != 0:
            model.load_state_dict(torch.load("mnist_cnn.pt"))

        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

        for epoch in range(1, args.epochs + 1):
            train(args, model, device, labelled_data, optimizer, epoch)
        test(args, model, device, test_data)

        # active_learn(unlabelled_data,model)
        rand_samp()
        print("active_learn over")
        lm = len(labelled_mask)
        um = len(unlabelled_mask)
        print('len of labelled_mask: ',lm)
        print('len of unlabelled_mask: ',um)

        if (args.save_model):
            torch.save(model.state_dict(),"mnist_cnn.pt")

        active_learn_iter = active_learn_iter + 1

    with open('results_adversarial.txt', 'w') as f:   

        for item in test_accs:
            f.write("%s\n"%item)
        
if __name__ == '__main__':
    main()