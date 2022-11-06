import torch

def validPixelMask(lossShape, borderPercentH,borderPercentW):

	# batchSize = lossShape[0]
	# height = lossShape[1]
	# width = lossShape[2]
	# channels = lossShape[3]

	batchSize, channels, height, width = lossShape

	smallestDimension = torch.minimum(height, width).int()

	borderThicknessH = torch.round(borderPercentH * height).int()
	borderThicknessW = torch.round(borderPercentW * width).int()

	innerHeight = (height - 2*borderThicknessH).int()
	innerWidth = (width - 2*borderThicknessW).int()

	topBottom = torch.zeros(batchSize, channels, borderThicknessH, innerWidth) # 1, 1, 22, 180
	leftRight = torch.zeros(batchSize, channels, height, borderThicknessW) # 1, 1, 224, 224
	center = torch.ones(batchSize, channels, innerHeight, innerWidth) # 1, 1, 180, 180

	mask = torch.cat([topBottom, center, topBottom],2)
	mask = torch.cat([leftRight, mask, leftRight],3) # 1, 1, 224, 224

	#set shape
	ref = torch.zeros(lossShape.tolist())

	mask.reshape_as(ref)

	return mask.cuda()
