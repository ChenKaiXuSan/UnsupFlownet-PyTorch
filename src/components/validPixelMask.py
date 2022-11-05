import torch

def validPixelMask(lossShape, borderPercentH,borderPercentW):

	batchSize = lossShape[0]
	height = lossShape[1]
	width = lossShape[2]
	channels = lossShape[3]

	smallestDimension = torch.minimum(height, width).int()

	borderThicknessH = torch.round(borderPercentH * height).int()
	borderThicknessW = torch.round(borderPercentW * width).int()

	innerHeight = (height - 2*borderThicknessH).int()
	innerWidth = (width - 2*borderThicknessW).int()

	topBottom = torch.zeros(batchSize, borderThicknessH, innerWidth, channels)
	leftRight = torch.zeros(batchSize,height,borderThicknessW,channels)
	center = torch.ones(batchSize,innerHeight,innerWidth,channels)

	mask = torch.cat([topBottom, center, topBottom],1)
	mask = torch.cat([leftRight, mask, leftRight],2)

	#set shape
	ref = torch.zeros(lossShape.tolist())

	mask.reshape_as(ref)

	return mask.cuda()
