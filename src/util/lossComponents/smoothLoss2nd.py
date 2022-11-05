import tensorflow as tf
import torch.nn.functional as F
from components import *


def smoothLoss2ndMaskCorrection(validMask):
	"""
	makes correct mask for smoothness based on a valid pixel mask
	if any invalid pixel is within the inclusion kernel, ignore
	"""

	# inclusionKernel = tf.transpose(tf.constant([\
	# 	[ \
	# 		[ \
	# 			[0,0,0,0,0],\
	# 			[0,0,0,0,0],\
	# 			[0,0,1,1,1],\
	# 			[0,0,1,0,0],\
	# 			[0,0,1,0,0]\
	# 		] \
	# 	] \
	# ],dtype=tf.float32),perm=[3,2,1,0])

	inclusionKernel = torch.tensor([\
		[ \
			[ \
				[0,0,0,0,0],\
				[0,0,0,0,0],\
				[0,0,1,1,1],\
				[0,0,1,0,0],\
				[0,0,1,0,0]\
			] \
		] \
	], dtype=torch.float32).cuda() # 1, 1, 5, 5

	# maskCor = tf.nn.conv2d(validMask,inclusionKernel,[1,1,1,1],padding="SAME")
	# maskCor = tf.greater_equal(maskCor,4.95)
	# maskCor = tf.cast(maskCor,tf.float32)

	maskCor = F.conv2d(validMask, inclusionKernel, stride=1, padding='same') # 1, 1, 224, 224
	maskCor = torch.greater_equal(maskCor, 4.95).to(torch.float32)

	return maskCor

def smoothLoss2nd(flow,alpha,beta,validPixelMask=None,img0Grad=None,boundaryAlpha=0):
	"""
	same as smoothLoss but second order
	"""
	# kernel = tf.transpose(tf.constant([\
	# 	[ \
	# 		[ \
	# 			[0,0,0],\
	# 			[0,1,-1],\
	# 			[0,0,0]\
	# 		] \
	# 	], \
	# 	[ \
	# 		[ \
	# 			[0,0,0],\
	# 			[0,1,0],\
	# 			[0,-1,0]\
	# 		] \
	# 	] \
	# ],dtype=tf.float32),perm=[3,2,1,0])

	kernel = torch.tensor([\
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

	# u = tf.slice(flow,[0,0,0,0],[-1,-1,-1,1])
	# v = tf.slice(flow,[0,0,0,1],[-1,-1,-1,-1])

	u = flow[:, 0, :].unsqueeze(dim=1) # horizontal
	v = flow[:, 1, :].unsqueeze(dim=1) # vertical

	# first order
	# neighborDiffU = tf.nn.conv2d(u,kernel,[1,1,1,1],padding="SAME")
	# neighborDiffV = tf.nn.conv2d(v,kernel,[1,1,1,1],padding="SAME")

	neighborDiffU = F.conv2d(u, kernel, stride=1, padding='same') # 1, 2, h, w
	neighborDiffV = F.conv2d(v, kernel, stride=1, padding='same') # 1, 2, h, w

	# 2nd order
	# neighborDiffU_x = tf.nn.conv2d(tf.expand_dims(neighborDiffU[:,:,:,0],-1),kernel,[1,1,1,1],padding="SAME")
	# neighborDiffU_y = tf.nn.conv2d(tf.expand_dims(neighborDiffU[:,:,:,1],-1),kernel,[1,1,1,1],padding="SAME")
	# neighborDiffV_x = tf.nn.conv2d(tf.expand_dims(neighborDiffV[:,:,:,0],-1),kernel,[1,1,1,1],padding="SAME")
	# neighborDiffV_y = tf.nn.conv2d(tf.expand_dims(neighborDiffV[:,:,:,1],-1),kernel,[1,1,1,1],padding="SAME")

	neighborDiffU_x = F.conv2d(neighborDiffU[:, 0, ...].unsqueeze(dim=1), kernel, stride=1, padding='same') # 1, 2, 224, 224
	neighborDiffU_y = F.conv2d(neighborDiffU[:, 1, ...].unsqueeze(dim=1), kernel, stride=1, padding='same') # 1, 2, 224, 224
	neighborDiffV_x = F.conv2d(neighborDiffV[:, 0, ...].unsqueeze(dim=1), kernel, stride=1, padding='same') # 1, 2, 224, 224
	neighborDiffV_y = F.conv2d(neighborDiffV[:, 1, ...].unsqueeze(dim=1), kernel, stride=1, padding='same') # 1, 2, 224, 224

	# diffs = tf.concat([neighborDiffU_x,neighborDiffU_y,neighborDiffV_x,neighborDiffV_y],3)
	diffs = torch.cat([neighborDiffU_x,neighborDiffU_y,neighborDiffV_x,neighborDiffV_y], dim=1) # 1, 8, 224, 224

	# dists = tf.reduce_sum(tf.abs(diffs),axis=3,keep_dims=True)
	dists = torch.abs(diffs).sum(dim=1, keepdim=True).mean(dim=1, keepdim=True) # 1, 1, 224, 224
	robustLoss = charbonnierLoss(dists,alpha,beta,0.001)

	if not img0Grad == None:
		dMag = tf.sqrt(tf.reduce_sum(img0Grad**2, axis=3, keep_dims=True))
		mask = tf.exp(-boundaryAlpha*dMag)
		robustLoss *= mask

		# debug
		tf.summary.image("boundaryMask", mask)

	if validPixelMask is None:
		return robustLoss
	else:
		correctedMask = smoothLoss2ndMaskCorrection(validPixelMask)
		return robustLoss*correctedMask
