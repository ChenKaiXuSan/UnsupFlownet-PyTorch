import torch
from components import *

def photoLoss(flow,downsampledFrame0,downsampledFrame1,alpha,beta):

	batchSize, c, height, width = flow.size()

	warpedFrame2 = flowWarp(downsampledFrame1,flow) # 1, 3, 224, 224

	# photometric subtraction
	photoDiff = downsampledFrame0 - warpedFrame2 # 1, 3, 224, 224

	# photoDist = tf.reduce_sum(tf.abs(photoDiff),axis=3,keep_dims=True)
	photoDist = torch.abs(photoDiff).sum(dim=1, keepdim=True).mean(dim=1, keepdim=True) # 1, 1, 224, 224
	robustLoss = charbonnierLoss(photoDist,alpha,beta,0.001)

	return robustLoss

# robust generalized Charbonnier penalty function
def charbonnierLoss(x,alpha,beta,epsilon):
	epsilonSq = epsilon*epsilon
	xScale = x*beta

	return torch.pow(xScale * xScale + epsilonSq, alpha)