import tensorflow as tf
from components import *
from photoLoss import *

def gradLoss(flow,downsampledGrad0,downsampledGrad1,alpha,beta):
	"""
	like photoloss but use image gradients instead
	"""
	return photoLoss(flow,downsampledGrad0,downsampledGrad1,alpha,beta)
