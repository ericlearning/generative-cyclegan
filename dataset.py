import os
import torch
import random
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image

class Dataset():
	def __init__(self, train_dir, basic_types = None, shuffle = True):
		self.train_dir = train_dir
		self.basic_types = basic_types
		self.shuffle = shuffle

	def get_loader(self, sz, bs, get_size = False, data_transform = None, num_workers = 1, audio_sample_num = None):
		if(self.basic_types == 'CycleGan'):
			input_transform = data_transform['input']
			target_transform = data_transform['target']

			train_dataset = CycleGan_Dataset(self.train_dir[0], self.train_dir[1], input_transform, target_transform)
			train_loader = DataLoader(train_dataset, batch_size = bs, shuffle = self.shuffle, num_workers = num_workers)

			returns = (train_loader)

		return returns

class CycleGan_Dataset():
	def __init__(self, input_dir, target_dir, input_transform, target_transform):
		self.input_dir = input_dir
		self.target_dir = target_dir
		self.input_transform = input_transform
		self.target_transform = target_transform

		self.A_image_name_list = []
		for file in os.listdir(input_dir):
			if(file.endswith('.png') or file.endswith('.jpeg') or file.endswith('.jpg') or file.endswith('.bmp')):
				self.A_image_name_list.append(file)

		self.B_image_name_list = []
		for file in os.listdir(target_dir):
			if(file.endswith('.png') or file.endswith('.jpeg') or file.endswith('.jpg') or file.endswith('.bmp')):
				self.B_image_name_list.append(file)

	def __len__(self):
		return len(self.A_image_name_list)

	def __getitem__(self, idx):
		input_img = Image.open(os.path.join(self.input_dir, self.A_image_name_list[idx]))
		target_img = Image.open(os.path.join(self.target_dir, self.B_image_name_list[random.randint(0, len(self.B_image_name_list) - 1)]))

		input_img = self.input_transform(input_img)
		target_img = self.target_transform(target_img)

		sample = (input_img, target_img)
		return sample


