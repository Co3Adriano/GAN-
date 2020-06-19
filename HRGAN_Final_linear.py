# HRGAN final 
# Conditional Generative Adversarial Nets
# 
# Base Environment: CUDA, Pytorch, MatplotLib 
# missing 
# PreProcessing: 
# DefImshow :  ImagePlotting
# Generator : 
# Plotting Tools : Analizing 

# imports
import torch
import torchvision as torchvision
import numpy as np
import math
import matplotlib.pyplot as plt
import argparse
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import os
import matplotlib.cm as cm
import matplotlib 
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, IterableDataset, DataLoader


# HyperParameters
batch_size = 11#  (number_of_samples / batch_size)*num_of_epochs = interations
num_epochs = 1000
lr = 1e-4 # LearningRate
load_model = False #CAUTION! setting to False OVERWRITES checkpoint.pth.tar
Z_dim = 456 # Noise Dimension
H_dim = 1824 #Hidden Layer

# device 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # using gpu if cuda is available 
print('Using device:', device)


# Root Dir
root_dir = r'E:/BachelorArbeit//KI_Regen//RadoLan_ASC_SET//sorted150//'


def get_data(root_dir):
    A = np.loadtxt(root_dir, skiprows = 6)
    return A

#class Dataset
class my_Dataset(torch.utils.data.Dataset):
    # Characterizes a dataset for PyTorch

    def __init__(self, root, loader = get_data, transform=torch.from_numpy):
        self.root = root
        self.files = os.listdir(self.root)
        self.loader = loader
        self.transform = transform

    def __len__(self):
        #print("files : " , len(self.files))
        return len(self.files)
    
    def __getitem__(self, index):
        #print  (self.transform(self.loader(os.path.join(self.root, self.files[index]))))
        #print(os.path.join(self.root, self.files[index]))
        return self.transform(self.loader(os.path.join(self.root, self.files[index])))

dataset = my_Dataset( root = root_dir)
trainloader = torch.utils.data.DataLoader(dataset, shuffle=False, batch_size=batch_size)

dataIter = iter(trainloader)
print(dataIter)
matrx= dataIter.next()
print("Shape : "  , matrx.shape)
print(matrx)
X_dim = (matrx.view(matrx.size(0), -1).size(1))

print("X-Dimension: ", X_dim)

# save function
def save_checkpoint(state, filename="checkpoint.pth.tar"):
    print("Saving Checkpoint")
    torch.save(state,filename)

# load function
def load_checkpoint(checkpoint):
    print("Loading Checkpoint")
    G.load_state_dict(checkpoint['state_dict_G'])
    D.load_state_dict(checkpoint['state_dict_D'])
    g_opt.load_state_dict(checkpoint['optimizer_g'])
    d_opt.load_state_dict(checkpoint['optimizer_d'])




# Generator
class Gen(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(Z_dim, H_dim),
            nn.ReLU(True),
            nn.Linear(H_dim, H_dim),
            nn.LeakyReLU(True),
            nn.Linear(H_dim, X_dim)

        )
          
    def forward(self, input):
        return self.model(input)


G = Gen().to(device=device)

# Diskriminator
class Dis(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(X_dim, H_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(H_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, input):
        return self.model(input)


D = Dis().to(device=device)


print(G)
print(D)

# Optimizer
g_opt = optim.Adam(G.parameters(), lr=lr)
d_opt = optim.Adam(D.parameters(), lr=lr)


# saving Progress
G_losses = []
D_losses = []

# load checkpoint if-check
if load_model:
    load_checkpoint(torch.load("checkpoint.pth.tar"))

# TrainingsLoop

for epoch in range(num_epochs):
    G_loss_run = 0.0
    D_loss_run = 0.0
    

    if epoch == 1000:
            checkpoint = {
            'state_dict_G' : G.state_dict(),
            'state_dict_D' : D.state_dict(),
            'optimizer_g' : g_opt.state_dict(),
            'optimizer_d' : d_opt.state_dict()}
            save_checkpoint(checkpoint)
    for i, data in enumerate(trainloader):
        X = data.float()
        X = X.view(X.size(0), -1).to(device=device)
        batch_size = X.size(0)
        
        one_labels = torch.ones(batch_size, 1, dtype=torch.float32).to(device=device)
        zero_labels = torch.zeros(batch_size, 1, dtype=torch.float32).to(device=device)
        
        z = torch.randn(batch_size, Z_dim).to(device=device)
        
        D_real = D(X)
        D_fake = D(G(z))
        
        D_real_loss = F.binary_cross_entropy(D_real, one_labels)
        D_fake_loss = F.binary_cross_entropy(D_fake, zero_labels)
        D_loss = D_real_loss + D_fake_loss
        
        d_opt.zero_grad()
        D_loss.backward()
        d_opt.step()
        
        z = torch.randn(batch_size, Z_dim).to(device=device)
        D_fake = D(G(z))
        G_loss = F.binary_cross_entropy(D_fake, one_labels)
        
        g_opt.zero_grad()
        G_loss.backward()
        g_opt.step()
        
        G_loss_run += G_loss.item()
        D_loss_run += D_loss.item()
        G_losses.append(G_loss_run)
        D_losses.append(D_loss_run)
        print('Epoch:{},   G_loss:{},    D_loss:{}'.format(epoch+1, G_loss_run/(i+1), D_loss_run/(i+1)))

# show results        
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()    

samples = G(z).detach()
samples = samples.view(samples.size(0),24, 19).cpu()  # matrix Size! 
print(samples.size(0))
print(samples)
print(samples.shape)
samples.numpy()
np.save('Results_numpyMatrix',samples)

# Save Progress

