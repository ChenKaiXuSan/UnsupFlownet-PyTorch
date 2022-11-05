import tensorflow as tf
import torch
from torchvision.transforms.functional import rgb_to_grayscale

def rgbToGray(img):
	# rgbWeights = [0.2989,0.5870,0.1140]
	# rgbWeights = tf.expand_dims(tf.expand_dims(tf.expand_dims(rgbWeights,0),0),0)

	return rgb_to_grayscale(img, num_output_channels=1)

	# weightedImg = img.permute(0, 2, 3, 1) *rgbWeights

	# return tf.reduce_sum(weightedImg,reduction_indices=[3],keep_dims=True)
