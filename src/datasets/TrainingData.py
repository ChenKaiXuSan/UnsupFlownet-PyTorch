import tensorflow as tf

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt

from components import *
from components.augmentation import *
from components.validPixelMask import *
from components.rgbToGray import *
from components.gradientFromGray import *

# import data_input
# from data_input.reader.DataReader import DataReader, Png
# from data_input.pre_processor.SharedCrop import SharedCrop
# from data_input.DataQueuer import DataQueuer


class TrainingData:
	"""
	handles queuing and preprocessing prior to and after batching
	"""

	def __init__(self, batchSize, instanceParams, shuffle=True):
		borderThicknessH = instanceParams["borderThicknessH"]
		borderThicknessW = instanceParams["borderThicknessW"]
		if instanceParams["dataset"] == "kitti2012" or instanceParams["dataset"] == "kitti2015":
			datasetRoot = "/workspace/UnsupFlownet-PyTorch/example_data/"
			frame0Path = datasetRoot+"datalists/train_im0.txt"
			frame1Path = datasetRoot+"datalists/train_im1.txt"
			desiredHeight = 320
			desiredWidth = 1152
		elif instanceParams["dataset"] == "sintel":
			datasetRoot = "/home/jjyu/datasets/Sintel/"
			frame0Path = datasetRoot+"datalists/train_raw_im0.txt"
			frame1Path = datasetRoot+"datalists/train_raw_im1.txt"
			desiredHeight = 384
			desiredWidth = 960
		else:
			assert False, "unknown dataset: " + instanceParams["dataset"]

		# create data readers
		# frame0Reader = Png(datasetRoot,frame0Path,3)
		# frame1Reader = Png(datasetRoot,frame1Path,3)

		# create croppers since kitti images are not all the same size
		# cropShape = [desiredHeight,desiredWidth] # 320, 1152
		# cropper = SharedCrop(cropShape,frame0Reader.data_out)

		# dataReaders = [frame0Reader,frame1Reader]
		# DataPreProcessors = [[cropper],[cropper]]
		# self.dataQueuer = DataQueuer(dataReaders,DataPreProcessors,n_threads=batchSize*4)

		# place data into batches, order of batches matches order of datareaders
		# batch = self.dataQueuer.queue.dequeue_many(batchSize)

		# Create the dataset
		dataset = dset.ImageFolder(root=datasetRoot + "data/",
								transform=transforms.Compose([
									transforms.Resize(224),
									transforms.CenterCrop(224),
									transforms.ToTensor(),
									transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
								]))
		# Create the dataloader
		dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchSize,
												shuffle=True, num_workers=2)

		# queuing complete
		return dataloader

		# ## async section done ##
		# #image augmentation
		# photoParam = photoAugParam(batchSize,0.7,1.3,0.2,0.9,1.1,0.7,1.5,0.00)
		# imData0aug = photoAug(img0raw,photoParam) - mean
		# imData1aug = photoAug(img1raw,photoParam) - mean

		# # artificial border augmentation
		# borderMask = validPixelMask(tf.stack([1, \
		# 	img0raw.get_shape()[1], \
		# 	img0raw.get_shape()[2], \
		# 	1]),borderThicknessH,borderThicknessW)

		# imData0aug *= borderMask
		# imData1aug *= borderMask

		# #LRN skipped
		# lrn0 = tf.nn.local_response_normalization(img0raw,depth_radius=2,alpha=(1.0/1.0),beta=0.7,bias=1)
		# lrn1 = tf.nn.local_response_normalization(img1raw,depth_radius=2,alpha=(1.0/1.0),beta=0.7,bias=1)

		# #gradient images
		# imData0Gray = rgbToGray(img0raw)
		# imData1Gray = rgbToGray(img1raw)

		# imData0Grad = gradientFromGray(imData0Gray)
		# imData1Grad = gradientFromGray(imData1Gray)

		# # ----------expose tensors-----------

		# self.frame0 = {
		# 	"rgb": imData0aug,
		# 	"rgbNorm": lrn0,
		# 	"grad": imData0Grad
		# }

		# self.frame1 = {
		# 	"rgb": imData1aug,
		# 	"rgbNorm": lrn1,
		# 	"grad": imData1Grad
		# }

		# self.validMask = borderMask


def example_dataset(batchSize, instanceParams):
	if instanceParams["dataset"] == "kitti2012" or instanceParams["dataset"] == "kitti2015":
		datasetRoot = "/workspace/UnsupFlownet-PyTorch/example_data/"
		desiredHeight = 320
		desiredWidth = 1152
	else:
		assert False, "unknown dataset: " + instanceParams["dataset"]

	# Create the dataset
	dataset = dset.ImageFolder(root=datasetRoot + "data/",
							transform=transforms.Compose([
								transforms.Resize(224),
								transforms.CenterCrop(224),
								transforms.ToTensor(),
								transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
							]))
	# Create the dataloader
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchSize,
											shuffle=True, num_workers=2)

	# todo uptonow pytorch should load the data.
	# queuing complete
	return dataloader


def image_augmentation(imgs, batchSize, instanceParams):

	borderThicknessH = instanceParams["borderThicknessH"]
	borderThicknessW = instanceParams["borderThicknessW"]

	img0raw = imgs[0].unsqueeze(dim=0)  # b, c, h, w
	img1raw = imgs[1].unsqueeze(dim=0)  # b, c, h, w
	mean = [[[[0.448553, 0.431021, 0.410602]]]]

	## async section done ##
	# image augmentation
	# photoParam = photoAugParam(batchSize,0.7,1.3,0.2,0.9,1.1,0.7,1.5,0.00)
	# imData0aug = photoAug(img0raw,photoParam) - mean
	# imData1aug = photoAug(img1raw,photoParam) - mean

	# artificial border augmentation
	borderMask = validPixelMask(
		torch.tensor([
		1,
		img0raw.size()[2],
		img0raw.size()[3],
		1], dtype=torch.int32),
		borderThicknessH, borderThicknessW).permute(0, 3, 1, 2)

	# imData0aug *= borderMask
	# imData1aug *= borderMask

	imData0aug = borderMask * img0raw
	imData1aug = borderMask * img1raw

	# LRN skipped, AlexNet LRN 
	lrn = nn.LocalResponseNorm(size=2, alpha=(1.0/1.0), beta=0.7, k=1.0)
	lrn0 = lrn(img0raw)
	lrn1 = lrn(img1raw)

	# gradient images
	imData0Gray = rgbToGray(img0raw)
	imData1Gray = rgbToGray(img1raw)

	imData0Grad = gradientFromGray(imData0Gray)
	imData1Grad = gradientFromGray(imData1Gray)

	# ----------expose tensors-----------

	frame0 = {
		"rgb": imData0aug,
		"rgbNorm": lrn0,
		"grad": imData0Grad
	}

	frame1 = {
		"rgb": imData1aug,
		"rgbNorm": lrn1,
		"grad": imData1Grad
	}

	return frame0, frame1, borderMask