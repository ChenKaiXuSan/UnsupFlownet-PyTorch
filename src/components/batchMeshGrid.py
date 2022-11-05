import tensorflow as tf
import torch

def batchMeshGrid(batchSize, height, width):

	# make grid
	x = torch.tensor(range(width))
	y = torch.tensor(range(height))

	X, Y = torch.meshgrid(x, y, indexing='xy')
	# X, Y = tf.meshgrid(x, y)
	X = X.to(torch.float32)
	Y = Y.to(torch.float32)
	# X = tf.cast(X,tf.float32)
	# Y = tf.cast(Y,tf.float32)
	# grid = tf.stack([X,Y], axis=-1)
	grid = torch.stack([X, Y], dim=0)

	# tile for batch
	# batchGrid = tf.expand_dims(grid, 0)
	batchGrid = torch.unsqueeze(grid, dim=0)
	# batchGrid = tf.tile(batchGrid, [batchSize,1,1,1])
	batchGrid = torch.tile(batchGrid, dims=[batchSize, 1, 1, 1])

	return batchGrid.cuda()

def batchMeshGridLike(tensor):
	b, c, h, w = tensor.size() # 1, 2, 224, 224
	return batchMeshGrid(b, h, w)
