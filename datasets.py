import os
import numpy as np
from PIL import Image
from torch.utils import data
from torchvision import datasets, transforms

#np.random.seed(42)

class TripletMNIST(data.Dataset):

	def __init__(self, mnist_dataset):
		self.mnist_dataset = mnist_dataset
		# Flag variable that tell if it a train set (i.e., returns True) or test set (i.e., returns False)
		self.train = self.mnist_dataset.train 
		# Return the transforms that have been applied on the dataset
		self.transform = self.mnist_dataset.transform

		if self.train:
			self.train_data = self.mnist_dataset.data
			self.train_targets = self.mnist_dataset.targets
			self.labels_set = set(self.train_targets.numpy())
			self.labels_to_indices = {label: np.where(self.train_targets == label)[0] for label in self.labels_set}
		else:
			self.test_data = self.mnist_dataset.data
			self.test_targets = self.mnist_dataset.targets
			self.labels_set = set(self.test_targets.numpy())
			self.labels_to_indices = {label: np.where(self.test_targets == label)[0] for label in self.labels_set}

	def __getitem__(self, index):

		if self.train:
			anchor_img = self.train_data[index]
			anchor_class = self.train_targets[index].item()
			ind = np.random.choice(self.labels_to_indices[anchor_class],1)[0]
			postive_img = self.train_data[ind]
			new_set = self.labels_set - set([anchor_class])
			negative_label = np.random.choice(list(new_set), 1)
			ind = np.random.choice(self.labels_to_indices[negative_label[0]],1)[0]
			negative_img = self.train_data[ind]
		else:
			anchor_img = self.test_data[index]
			anchor_class = self.test_targets[index].item()
			ind = np.random.choice(self.labels_to_indices[anchor_class],1)[0]
			postive_img = self.test_data[ind]
			new_set = self.labels_set - set([anchor_class])
			negative_label = np.random.choice(list(new_set), 1)
			ind = np.random.choice(self.labels_to_indices[negative_label[0]],1)[0]
			negative_img = self.test_data[ind]	
		
		img1 = anchor_img.unsqueeze(0)
		img2 = postive_img.unsqueeze(0)
		img3 = negative_img.unsqueeze(0)
		img1 = img1/255.0
		img2 = img2/255.0
		img3 = img3/255.0	
		return img1, img2, img3

	def __len__(self):
		
		if self.train:
			return int(self.train_data.shape[0])
		else:
			return int(self.test_data.shape[0])