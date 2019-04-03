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

def conv_2d(ni, nf, stride=1, ks=3):
    return nn.Conv2d(in_channels=ni, out_channels=nf, 
                     kernel_size=ks, stride=stride, 
                     padding=ks//2, bias=False)

def bn_relu_conv(ni, nf):
    return nn.Sequential(nn.BatchNorm2d(ni), 
                         nn.ReLU(inplace=True), 
                         conv_2d(ni, nf))

class BasicBlock(nn.Module):
    def __init__(self, ni, nf, stride=1):
        super(BasicBlock,self).__init__()
        self.bn = nn.BatchNorm2d(ni)
        self.conv1 = conv_2d(ni, nf, stride)
        self.conv2 = bn_relu_conv(nf, nf)
        self.shortcut = lambda x: x
        if ni != nf:
            self.shortcut = conv_2d(ni, nf, stride, 1)
    
    def forward(self, x):
        x = F.relu(self.bn(x), inplace=True)
        r = self.shortcut(x)
        x = self.conv1(x)
        x = self.conv2(x) * 0.2
        return x.add_(r)

def make_group(N, ni, nf, stride):
    start = BasicBlock(ni, nf, stride)
    rest = [BasicBlock(nf, nf) for j in range(1, N)]
    return [start] + rest

class Flatten(nn.Module):
    def __init__(self): super(Flatten,self).__init__()
    def forward(self, x): return x.view(x.size(0), -1)

class WideResNet(nn.Module):
    def __init__(self, n_groups, N, n_classes, k=1, n_start=16):
        super(WideResNet,self).__init__()      
        # Increase channels to n_start using conv layer
        layers = [conv_2d(3, n_start)]
        n_channels = [n_start]
        
        # Add groups of BasicBlock(increase channels & downsample)
        for i in range(n_groups):
            n_channels.append(n_start*(2**i)*k)
            stride = 2 if i>0 else 1
            layers += make_group(N, n_channels[i], 
                                 n_channels[i+1], stride)
        
        # Pool, flatten & add linear layer for classification
        layers += [nn.BatchNorm2d(n_channels[3]), 
                   nn.ReLU(inplace=True), 
                   nn.AdaptiveAvgPool2d(1), 
                   Flatten(), 
                   nn.Linear(n_channels[3], n_classes)]
        
        self.features = nn.Sequential(*layers)
        
    def forward(self, x): return self.features(x)
    
def wrn_22(): 
    return WideResNet(n_groups=3, N=3, n_classes=10, k=6)

class CNNNet(nn.Module):
    def __init__(self):
        super(CNNNet, self).__init__()
        
        # Convolutional layers
                            #Init_channels, channels, kernel_size, padding) 
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        
        # Pooling layers
        self.pool = nn.MaxPool2d(2,2)
        
        # FC layers
        # Linear layer (64x4x4 -> 500)
        self.fc1 = nn.Linear(64 * 4 * 4, 500)
        
        # Linear Layer (500 -> 10)
        self.fc2 = nn.Linear(500, 10)
        
        # Dropout layer
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):
        x = self.pool(F.elu(self.conv1(x)))
        x = self.pool(F.elu(self.conv2(x)))
        x = self.pool(F.elu(self.conv3(x)))
        
        # Flatten the image
        x = x.view(-1, 64*4*4)
        x = self.dropout(x)
        x = F.elu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

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
    add_labels = np.random.choice(unlabelled_mask, 500)
    labelled_mask = labelled_mask + add_labels
    unlabelled_mask = [x for x in unlabelled_mask if x not in add_labels]

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
        output = model(data)
        # import pdb;pdb.set_trace()
        output2 = output.reshape(output.shape[0],output.shape[1])
        # output_softmax = F.softmax(output)
        
        loss = F.cross_entropy(output2, target)
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
    while active_learn_iter<50:

        print('Active learning Iter: ', active_learn_iter)
        labelled_data = torch.utils.data.DataLoader(trainset, batch_size=32,
                                              sampler = SubsetRandomSampler(labelled_mask), shuffle=False, num_workers=2)
        global unlabelled_mask_rand
        # unlabelled_mask_rand = random.sample(unlabelled_mask, 2*query)
        unlabelled_mask_rand = np.random.choice(unlabelled_mask, 2*query)
        print('Random Numer 1 for test ', unlabelled_mask_rand[0])
        
        unlabelled_data = torch.utils.data.DataLoader(trainset, batch_size=1,
                                              sampler = SubsetRandomSampler(unlabelled_mask_rand), shuffle=False, num_workers=2)
        
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

        active_learn(unlabelled_data,model)
        # rand_samp()
        print("active_learn over")
        lm = len(labelled_mask)
        um = len(unlabelled_mask)
        print('len of labelled_mask: ',lm)
        print('len of unlabelled_mask: ',um)

        # if (args.save_model):
        #     torch.save(model.state_dict(),"cifar_resnet.pt")

        active_learn_iter = active_learn_iter + 1


        with open('results_adversarial_cifar_2april.txt', 'w') as f:   

            for item in test_accs:
                f.write("%s\n"%item)

        
if __name__ == '__main__':
    main()