#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

class board_data(Dataset):
    def __init__(self, dataset): # dataset = np.array of (s, p, v)
        self.X = dataset[:,0]
        self.y_p, self.y_v = dataset[:,1], dataset[:,2]
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self,idx):
        return self.X[idx].transpose(2,0,1), self.y_p[idx], self.y_v[idx]

class ConvBlock(nn.Module):
    def __init__(self):
        super(ConvBlock, self).__init__()
        self.action_size = 8*8*73
        self.conv1 = nn.Conv2d(22, 128, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(128)

    def forward(self, s):
        #s: batch_size x channels X board_x x board_y
        #s = torch.from_numpy(s).float()
        #print(s.size())
        s = s.view(-1, 22, 8, 8)
        #print(s.size())                                     # batch_size x channels x board_x x board_y
        s = F.relu(self.bn1(self.conv1(s)))
        return s

class ResBlock(nn.Module):
    def __init__(self, inplanes=128, planes=128, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = F.relu(out)
        return out
    
class OutBlock(nn.Module):
    def __init__(self):
        super(OutBlock, self).__init__()
        self.conv = nn.Conv2d(128, 1, kernel_size=1) # value head
        self.bn = nn.BatchNorm2d(1)
        self.fc1 = nn.Linear(8*8, 64)
        self.fc2 = nn.Linear(64, 1)
        
        self.conv1 = nn.Conv2d(128, 77, kernel_size=1) # policy head
        self.bn1 = nn.BatchNorm2d(77)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.fc = nn.Linear(8*8*77, 8*8*73)
    
    def forward(self,s):
        v = F.relu(self.bn(self.conv(s))) # value head
        v = v.view(-1, 8*8)  # batch_size X channel X height X width
        v = F.relu(self.fc1(v))
        v = F.tanh(self.fc2(v))
        
        p = F.relu(self.bn1(self.conv1(s))) # policy head
        p = p.view(-1, 8*8*77)
        p = self.fc(p)
        p = self.logsoftmax(p).exp()
        return p, v
    
class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        self.conv = ConvBlock()
        for block in range(13):
            setattr(self, "res_%i" % block,ResBlock())
        self.outblock = OutBlock()
    
    def forward(self,s):
        s = self.conv(s)
        for block in range(13):
            s = getattr(self, "res_%i" % block)(s)
        s = self.outblock(s)
        return s
        

class AlphaLoss(torch.nn.Module):
    def __init__(self):
        super(AlphaLoss, self).__init__()

    def forward(self, y_value, value, y_policy, policy):
        value_error = (value - y_value) ** 2
        #print(policy[0,3509],y_policy[0,3509])
        policy_error = torch.sum((-policy* 
                                (1e-6 + y_policy.float()).float().log()), 1)
        total_error = (value_error.view(-1).float() + policy_error).mean()
        #print(value_error); print(policy_error); print(total_error)
        return total_error
    
def train(net, dataset, epoch_start=0, epoch_stop=500):
    cuda = torch.cuda.is_available()
    if cuda:
        net.cuda()
    net.train()
    criterion = AlphaLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,200,300,400], gamma=0.2)
    
    train_set = board_data(dataset)
    train_loader = DataLoader(train_set, batch_size=30, shuffle=True, num_workers=0, pin_memory=False)
    losses_per_epoch = []
    for epoch in range(epoch_start, epoch_stop):
        scheduler.step()
        total_loss = 0.0
        losses_per_batch = []
        for i,data in enumerate(train_loader,0):
            state, policy, value = data
            if cuda:
                state, policy, value = state.cuda().float(), policy.cuda(), value.cuda().float()
            optimizer.zero_grad()
            # state = torch.Size([batch, 22, 8, 8]); policy = torch.Size([batch, 4672]) # value = torch.Size([batch])
            policy_pred, value_pred = net(state) # policy_pred = torch.Size([batch, 4672]) value_pred = torch.Size([batch, 1])
            #print(value_pred); print(value); print(policy_pred); print(policy)
            loss = criterion(value_pred[:,0], value, policy_pred, policy)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if i % 10 == 9:    # print every 1000 mini-batches of size = batch_size
                print('[Epoch: %d, %5d/ %d points] total loss per batch: %.3f' %
                      (epoch + 1, (i + 1)*10, len(train_set), total_loss/10))
                losses_per_batch.append(total_loss/10)
                total_loss = 0.0

        losses_per_epoch.append(sum(losses_per_batch)/len(losses_per_batch))
    fig = plt.figure()
    ax = fig.add_subplot(222)
    ax.scatter([e for e in range(1,epoch_stop+1,1)], losses_per_epoch)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss per batch")
    ax.set_title("Loss vs Epoch")
    print('Finished Training')
        