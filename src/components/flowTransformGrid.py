import tensorflow as tf
import torch

from batchMeshGrid import *

def flowTransformGrid(flow):
	resampleGrid = batchMeshGridLike(flow) + flow
	return resampleGrid
