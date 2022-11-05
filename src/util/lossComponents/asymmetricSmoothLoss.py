# finish to pytorch
import tensorflow as tf
import torch

from components import *
from smoothLoss import *

def asymmetricSmoothLoss(flow,instanceParams,occMask,validPixelMask,img0Grad=None,boundaryAlpha=0):
	"""
	modifies gradients so that smoothness can only go from non-occluded to occluded areas
	"""

	alpha = instanceParams["smoothParams"]["robustness"]
	beta = instanceParams["smoothParams"]["scale"]
	occAlpha = instanceParams["smoothOccParams"]["robustness"]
	occBeta = instanceParams["smoothOccParams"]["scale"]

	# occluded
	flowValid = flow*occMask
	flowInvalid = flow*(1.0-occMask)

	# block gradients to valid flow
	# flowValid = tf.stop_gradient(flowValid)
	flowValid = flowValid.detach()
	routedFlow = flowValid + flowInvalid

	occSmooth = smoothLoss(routedFlow,occAlpha,occBeta,None,img0Grad,boundaryAlpha)

	# non occluded
	nonOccSmooth = smoothLoss(flow,alpha,beta,occMask,img0Grad,boundaryAlpha)

	# final
	valid = smoothLossMaskCorrection(validPixelMask)
	smooth = nonOccSmooth + occSmooth
	return smooth*valid
