import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST
from torchvision.datasets import FashionMNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from networks import *
from loss_functions import *
from datasets import *

os.makedirs('checkpoints_FMNIST', exist_ok = True)
#os.makedirs('checkpoints_MNIST', exist_ok = True)

#train_dataset = MNIST('./', train = True, download = True, transform = transforms.Compose([transforms.ToTensor()]))
#test_dataset = MNIST('./', train = False, download = True, transform = transforms.Compose([transforms.ToTensor()]))
train_dataset = FashionMNIST('./', train = True, download = True, transform = transforms.Compose([transforms.ToTensor()]))
test_dataset = FashionMNIST('./', train = False, download = True, transform = transforms.Compose([transforms.ToTensor()])) 

triplet_train_dataset = TripletMNIST(train_dataset) # Same can be used for "Fashion MNIST"
triplet_test_dataset = TripletMNIST(test_dataset) # Same can be used for "Fashion MNIST"

mnist_classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

batch_size = 512
train_loader = DataLoader(triplet_train_dataset, batch_size = batch_size, shuffle = True, num_workers = 0)
test_loader = DataLoader(triplet_test_dataset, batch_size = batch_size, shuffle = True, num_workers = 0)

no_of_training_batches = len(train_loader)/batch_size
no_of_test_batches = len(test_loader)/batch_size

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
epochs = 60

embeddingNetTriplet = EmbeddingNetTriplet()
optimizer = optim.Adam(embeddingNetTriplet.parameters(), lr = 0.001, betas = (0.9, 0.999), weight_decay = 0.0005)

def run_epoch(data_loader, model, optimizer, split = 'train', epoch_count = 0):

	model.to(device)

	if split == 'train':
		model.train()
	else:
		model.eval()

	running_loss = 0.0

	for batch_id, (anchor_imgs, positive_imgs, negative_imgs) in enumerate(train_loader):
		
		anchor_imgs = anchor_imgs.type(torch.FloatTensor)
		positive_imgs = positive_imgs.type(torch.FloatTensor)
		negative_imgs = negative_imgs.type(torch.FloatTensor)
		anchor_imgs = anchor_imgs.to(device)
		positive_imgs = positive_imgs.to(device)
		negative_imgs = negative_imgs.to(device)
		emb_anchor, emb_postive, emb_negative = model.triplet_get_embeddings(anchor_imgs, positive_imgs, negative_imgs)
		batch_loss = triplet_loss(emb_anchor, emb_postive, emb_negative, margin = 1)
		optimizer.zero_grad()
		
		if split == 'train':
			batch_loss.backward()
			optimizer.step()

		running_loss = running_loss + batch_loss.item()

	return running_loss

def fit(train_loader, test_loader, model, optimizer, n_epochs):

	print('Training Started\n')
	
	for epoch in range(n_epochs):
		
		loss = run_epoch(train_loader, model, optimizer, split = 'train', epoch_count = epoch)
		loss = loss/no_of_training_batches

		print('Loss after epoch ' + str(epoch + 1) + ' is:', loss)
		#torch.save({'state_dict': model.cpu().state_dict()}, 'checkpoints_MNIST/model_epoch_' + str(epoch + 1) + '.pth')
		torch.save({'state_dict': model.cpu().state_dict()}, 'checkpoints_FMNIST/model_epoch_' + str(epoch + 1) + '.pth')

fit(train_loader, test_loader, embeddingNetTriplet, optimizer = optimizer, n_epochs = epochs)