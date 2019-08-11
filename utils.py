import os
import cv2
import glob
import torch
import imageio
import numpy as np
import pandas as pd
import seaborn as sn
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from scipy.io import wavfile
from PIL import Image

def set_lr(optimizer, lrs):
	if(len(lrs) == 1):
		for param in optimizer.param_groups:
			param['lr'] = lrs[0]
	else:
		for i, param in enumerate(optimizer.param_groups):
			param['lr'] = lrs[i]

def get_lr(optimizer):
	optim_param_groups = optimizer.param_groups
	if(len(optim_param_groups) == 1):
		return optim_param_groups[0]['lr']
	else:
		lrs = []
		for param in optim_param_groups:
			lrs.append(param['lr'])
		return lrs

def histogram_sizes(img_dir, h_lim = None, w_lim = None):
	hs, ws = [], []
	for file in glob.iglob(os.path.join(img_dir, '**/*.*')):
		try:
			with Image.open(file) as im:
				h, w = im.size
				hs.append(h)
				ws.append(w)
		except:
			print('Not an Image file')

	if(h_lim is not None and w_lim is not None):
		hs = [h for h in hs if h<h_lim]
		ws = [w for w in ws if w<w_lim]

	plt.figure('Height')
	plt.hist(hs)

	plt.figure('Width')
	plt.hist(ws)

	plt.show()

	return hs, ws

def generate_noise(bs, nz, device):
	noise = torch.randn(bs, nz, 1, 1, device = device)
	return noise

def plot_multiple_images(images, h, w):
	fig = plt.figure(figsize=(8, 8))
	for i in range(1, h*w+1):
		img = images[i-1]
		fig.add_subplot(h, w, i)
		if(img.shape[2] == 1):
			img = img.reshape(img.shape[0], img.shape[1])
		plt.imshow(img, cmap = 'gray')

	plt.show()
	return fig

def save(filename, netD, netG, optD, optG):
	state = {
		'netD' : netD.state_dict(),
		'netG' : netG.state_dict(),
		'optD' : optD.state_dict(),
		'optG' : optG.state_dict()
	}
	torch.save(state, filename)

def save_extra(filename, netD_A, netD_B, netG_A2B, netG_B2A, optD_A, optD_B, optG):
	state = {
		'netD_A' : netD_A.state_dict(),
		'netD_B' : netD_B.state_dict(),
		'netG_A2B' : netG_A2B.state_dict(),
		'netG_B2A' : netG_B2A.state_dict(),
		'optD_A' : optD_A.state_dict(),
		'optD_B' : optD_B.state_dict(),
		'optG' : optG.state_dict()
	}
	torch.save(state, filename)

def load(filename, netD, netG, optD, optG):
	state = torch.load(filename)
	netD.load_state_dict(state['netD'])
	netG.load_state_dict(state['netG'])
	optD.load_state_dict(state['optD'])
	optG.load_state_dict(state['optG'])

def load_extra(filename, netD_A, netD_B, netG_A2B, netG_B2A, optD_A, optD_B, optG):
	state = torch.load(filename)
	netD_A.load_state_dict(state['netD_A'])
	netD_B.load_state_dict(state['netD_B'])
	netG_A2B.load_state_dict(state['netG_A2B'])
	netG_B2A.load_state_dict(state['netG_B2A'])
	optD_A.load_state_dict(state['optD_A'])
	optD_B.load_state_dict(state['optD_B'])
	optG.load_state_dict(state['optG'])

def save_fig(filename, fig):
	fig.savefig(filename)

def rgb_to_ab(img):
	ab_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2LAB)[:, :, 1:]
	ab_img = (ab_img - 128.0) / 127.0
	ab_img = torch.from_numpy(ab_img.transpose(2, 0, 1)).float()
	return ab_img

def rgb_to_l(img):
	l_img = np.expand_dims(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2LAB)[:, :, 0], axis = 2)
	l_img = l_img * 2.0 / 100.0 - 1.0
	l_img = torch.from_numpy(l_img.transpose(2, 0, 1)).float()
	return l_img

def lab_to_rgb(img):
	l, a, b = (img[:, :, 0] + 1.0) * 100.0 / 2.0, img[:, :, 1] * 127.0 + 128.0, img[:, :, 2] * 127.0 + 128.0
	lab = np.dstack([l, a, b]).astype(np.uint8)
	rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
	return rgb

def get_sample_images_list(mode, inputs):
	if(mode == 'Cyclegan'):
		val_data, netG_A2B, netG_B2A, device = inputs[0], inputs[1], inputs[2], inputs[3]
		netG_A2B.eval()
		netG_B2A.eval()
		with torch.no_grad():
			A = val_data[0].to(device)
			B = val_data[1].to(device)

			sample_A_images = A.detach().cpu().numpy()
			sample_A_images_list = []

			sample_B_images = B.detach().cpu().numpy()
			sample_B_images_list = []

			sample_A2B_images = netG_A2B(A).detach()
			sample_A_Reconstruction_images = netG_B2A(sample_A2B_images).detach().cpu().numpy()
			sample_A2B_images = sample_A2B_images.cpu().numpy()
			sample_A2B_images_list = []
			sample_A_Reconstruction_images_list = []

			sample_B2A_images = netG_B2A(B).detach()
			sample_B_Reconstruction_images = netG_A2B(sample_B2A_images).detach().cpu().numpy()
			sample_B2A_images = sample_B2A_images.cpu().numpy()
			sample_B2A_images_list = []
			sample_B_Reconstruction_images_list = []

			for j in range(3):
				cur_img = (sample_A_images[j] + 1) / 2.0
				sample_A_images_list.append(cur_img.transpose(1, 2, 0))
				cur_img = (sample_B_images[j] + 1) / 2.0
				sample_B_images_list.append(cur_img.transpose(1, 2, 0))
				cur_img = (sample_A2B_images[j] + 1) / 2.0
				sample_A2B_images_list.append(cur_img.transpose(1, 2, 0))
				cur_img = (sample_A_Reconstruction_images[j] + 1) / 2.0
				sample_A_Reconstruction_images_list.append(cur_img.transpose(1, 2, 0))
				cur_img = (sample_B2A_images[j] + 1) / 2.0
				sample_B2A_images_list.append(cur_img.transpose(1, 2, 0))
				cur_img = (sample_B_Reconstruction_images[j] + 1) / 2.0
				sample_B_Reconstruction_images_list.append(cur_img.transpose(1, 2, 0))

			sample_images_list = []
			sample_images_list.extend(sample_A_images_list)
			sample_images_list.extend(sample_B_images_list)
			sample_images_list.extend(sample_A2B_images_list)
			sample_images_list.extend(sample_B2A_images_list)
			sample_images_list.extend(sample_A_Reconstruction_images_list)
			sample_images_list.extend(sample_B_Reconstruction_images_list)

		netG_A2B.train()
		netG_B2A.train()

	return sample_images_list