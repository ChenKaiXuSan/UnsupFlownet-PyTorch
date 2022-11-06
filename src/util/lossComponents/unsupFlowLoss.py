import tensorflow as tf
from components import *
from photoLoss import *
from gradLoss import *
from smoothLoss import *
from smoothLoss2nd import *
from asymmetricSmoothLoss import *
from torch.utils.tensorboard import SummaryWriter

def unsupFlowLoss(flow,flowB,frame0,frame1,validPixelMask,instanceParams, summaryWriter: SummaryWriter):
	# hyperparams
	photoAlpha = instanceParams["photoParams"]["robustness"]
	photoBeta = instanceParams["photoParams"]["scale"]

	smoothReg = instanceParams["smoothParams"]["weight"]

	smooth2ndReg = instanceParams["smooth2ndParams"]["weight"]
	smooth2ndAlpha = instanceParams["smooth2ndParams"]["robustness"]
	smooth2ndBeta = instanceParams["smooth2ndParams"]["scale"]

	gradReg = instanceParams["gradParams"]["weight"]
	gradAlpha = instanceParams["gradParams"]["robustness"]
	gradBeta = instanceParams["gradParams"]["scale"]

	boundaryAlpha = instanceParams["boundaryAlpha"]
	lossComponents = instanceParams["lossComponents"]

	# helpers
	rgb0 = frame0["rgbNorm"]
	rgb1 = frame1["rgbNorm"]
	grad0 = frame0["grad"]
	grad1 = frame1["grad"]

	# masking from simple occlusion/border
	occMask = borderOcclusionMask(flow) # occ if goes off image
	occInvalidMask = validPixelMask*occMask # occluded and invalid

	# loss components
	photo = photoLoss(flow,rgb0,rgb1,photoAlpha,photoBeta)
	grad = gradLoss(flow,grad0,grad1,gradAlpha,gradBeta)
	imgGrad = None
	if lossComponents["boundaries"]:
		imgGrad = grad0

	# todo
	if lossComponents["asymmetricSmooth"]:
		smoothMasked = asymmetricSmoothLoss(flow,instanceParams,occMask,validPixelMask,imgGrad,boundaryAlpha)
	else:
		smoothMasked = smoothLoss(flow,smoothAlpha,smoothBeta,validPixelMask,imgGrad,boundaryAlpha)
	smooth2ndMasked = smoothLoss2nd(flow,smooth2ndAlpha,smooth2ndBeta,validPixelMask,imgGrad,boundaryAlpha)

	# apply masking
	photoMasked = photo * occInvalidMask
	gradMasked = grad * occInvalidMask

	# average spatially
	photoAvg = photoMasked.mean(dim=(2, 3))
	gradAvg = gradMasked.mean(dim=(2, 3))
	smoothAvg = smoothMasked.mean(dim=(2, 3))
	smooth2ndAvg = smooth2ndMasked.mean(dim=(2, 3))

	# weight loss terms
	gradAvg = gradAvg*gradReg
	smoothAvg = smoothAvg*smoothReg
	smooth2ndAvg = smooth2ndAvg*smooth2ndReg

	# summaries ----------------------------
	summaryWriter.add_scalar("photoLoss", photoAvg.item())
	summaryWriter.add_scalar("smoothLoss", smoothAvg.item())
	# tf.summary.scalar("photoLoss",photoAvg.item())
	# tf.summary.scalar("smoothLoss",smoothAvg.item())

	# final loss
	finalLoss = photoAvg + smoothAvg
	if lossComponents["smooth2nd"]:
		tf.summary.scalar("smooth2ndLoss",smooth2ndAvg.item())
		finalLoss += smooth2ndAvg
	if lossComponents["gradient"]:
		# tf.summary.scalar("gradLoss", gradAvg.item())
		summaryWriter.add_scalar("gradLoss", gradAvg.item())
		finalLoss += gradAvg
	return finalLoss

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

def gradLoss(flow,downsampledGrad0,downsampledGrad1,alpha,beta):
	"""
	like photoloss but use image gradients instead
	"""
	return photoLoss(flow,downsampledGrad0,downsampledGrad1,alpha,beta)
