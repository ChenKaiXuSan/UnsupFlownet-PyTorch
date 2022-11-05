import tensorflow as tf
import torch
import torch.nn.functional as F
from components import *

def smoothLossMaskCorrection(validMask):
	"""
	makes correct mask for smoothness based on a valid pixel mask
	if any invalid pixel is within the inclusion kernel, ignore
	"""

	# inclusionKernel = tf.transpose(tf.constant([\
	# 	[ \
	# 		[ \
	# 			[0,0,0],\
	# 			[0,1,1],\
	# 			[0,1,0]\
	# 		] \
	# 	] \
	# ],dtype=tf.float32),perm=[3,2,1,0])

	inclusionKernel = torch.tensor([
		[ \
			[ \
				[0,0,0],\
				[0,1,1],\
				[0,1,0]\
			] \
		] \
	], dtype=torch.float32).cuda()

	maskCor = F.conv2d(validMask, inclusionKernel, stride=1, padding='same')
	# maskCor = tf.nn.conv2d(validMask,inclusionKernel,[1,1,1,1],padding="SAME")
	maskCor = torch.greater_equal(maskCor, 2.95).to(torch.float32)
	# maskCor = tf.greater_equal(maskCor,2.95)
	# maskCor = tf.cast(maskCor,tf.float32)

	return maskCor

def smoothLoss(flow,alpha,beta,validPixelMask=None,img0Grad=None,boundaryAlpha=0):
	"""
	smoothness loss, includes boundaries if img0Grad != None
	"""

	kernel = torch.tensor([
		[ \
			[ \
				[0,0,0],\
				[0,1,-1],\
				[0,0,0]\
			] \
		], \
		[ \
			[ \
				[0,0,0],\
				[0,1,0],\
				[0,-1,0]\
			] \
		] \
	], dtype=torch.float32).cuda() # 2, 1, 3, 3

	u = flow[:, 0, :].unsqueeze(dim=1) # horizontal
	v = flow[:, 1, :].unsqueeze(dim=1) # vertical

	# flowShape = flow.get_shape()

	neighborDiffU = F.conv2d(u, kernel, stride=1, padding='same')
	neighborDiffV = F.conv2d(v, kernel, stride=1, padding='same')

	diffs = torch.cat([neighborDiffU, neighborDiffV], dim=1) # 1, 4, 224, 224
	# diffs = tf.concat([neighborDiffU,neighborDiffV],3)
	# dists = tf.reduce_sum(tf.abs(diffs),axis=3,keep_dims=True)
	dists = torch.abs(diffs).sum(dim=1, keepdim=True).mean(dim=1, keepdim=True) # 1, 1, 224, 224
	# dists = torch.abs(diffs)
	robustLoss = charbonnierLoss(dists,alpha,beta,0.001)

	if not img0Grad == None:
		dMag = torch.sqrt(tf.reduce_sum(img0Grad**2, axis=3, keep_dims=True))
		# dMag = tf.sqrt(tf.reduce_sum(img0Grad**2, axis=3, keep_dims=True))
		mask = torch.exp(-boundaryAlpha*dMag)
		# mask = tf.exp(-boundaryAlpha*dMag)
		robustLoss *= mask

		# debug
		tf.summary.image("boundaryMask", mask)

	if validPixelMask is None:
		return robustLoss
	else:
		return robustLoss*smoothLossMaskCorrection(validPixelMask)
