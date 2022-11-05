import tensorflow as tf
import torch

from flowTransformGrid import *

def flowWarp(data,flow):
	resampleGrid = flowTransformGrid(flow).permute(0, 2, 3, 1)
	#  Spatial Transformer Networks, the grid should (b, h, w, 2).
	warped = torch.nn.functional.grid_sample(data, resampleGrid)
	return warped
