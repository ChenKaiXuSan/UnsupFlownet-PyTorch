import tensorflow as tf
import torch

def borderOcclusionMask(flow):
	flowShape = flow.size()

	# make grid
	x = torch.tensor(range(flowShape[3])).cuda()
	y = torch.tensor(range(flowShape[2])).cuda()

	X, Y = torch.meshgrid(x, y, indexing='xy')

	X = X.unsqueeze(dim=0).to(torch.float32) # 1, 224, 224
	Y = Y.unsqueeze(dim=0).to(torch.float32) # 1, 224, 224

	# mask flows that move off image
	grid = torch.cat([X, Y], dim=0).unsqueeze(0) # 1, 2, 224, 224

	grid = torch.tile(grid, [flowShape[0], 1, 1, 1]) # 2, 2, 224, 224

	flowPoints = grid + flow

	flowPointsU = flowPoints[:, 0, :].unsqueeze(dim=1) # b, 1, h, w
	flowPointsV = flowPoints[:, 1, :].unsqueeze(dim=1) # b, 1, h, w

	mask1 = torch.greater(flowPointsU, 0)
	mask2 = torch.greater(flowPointsV, 0)
	mask3 = torch.less(flowPointsU, flowShape[3]-1)
	mask4 = torch.less(flowPointsV, flowShape[2]-1)

	mask = torch.logical_and(mask1, mask2)
	mask = torch.logical_and(mask, mask3)
	mask = torch.logical_and(mask, mask4).to(torch.float32)

	return mask
