'''
change the summary writer to torch.
'''

import tensorflow as tf
from torch.utils.tensorboard import SummaryWriter

from components import *
from lossComponents import *

class TrainingLoss:
	def __init__(self,instanceParams,flows_F,flows_B, frame0, frame1, validMask, summaryWriter: SummaryWriter):
		# hyperparams
		weightDecay = instanceParams["weightDecay"]
		lossComponents = instanceParams["lossComponents"]

		# helpers
		predFlowF = flows_F
		predFlowB = flows_B
		vpm = validMask

		# unsup loss
		recLossF = unsupFlowLoss(predFlowF,predFlowB,frame0,frame1,vpm,instanceParams, summaryWriter)
		if lossComponents["backward"]:
			recLossB = unsupFlowLoss(predFlowB,predFlowF,frame1,frame0,vpm,instanceParams, summaryWriter)

		# weight decay
		# todo
		# weightLoss = tf.reduce_sum(tf.stack([tf.nn.l2_loss(i) for i in tf.get_collection("weights")]))*weightDecay
		weightLoss = weightDecay

		# final loss, schedule backward unsup loss
		recLossBWeight = tf.placeholder(tf.float32,shape=[]) #  [0,0.5]
		self.recLossBWeight = recLossBWeight # used to set weight at runtime
		if lossComponents["backward"]:
			totalLoss = \
				recLossF*(1.0 - recLossBWeight) + \
				recLossB*recLossBWeight + \
				weightLoss
			tf.summary.scalar("recLossF",tf.reduce_mean(recLossF*(1.0-recLossBWeight)))
			tf.summary.scalar("recLossB",tf.reduce_mean(recLossF*recLossBWeight))
		else:
			totalLoss = recLossF + weightLoss
			# tf.summary.scalar("recLossF",recLossF.item())
			summaryWriter.add_scalar("recLossF", recLossF.item())

		# tf.summary.scalar("weightDecay", weightLoss)
		summaryWriter.add_scalar("weightDecay", weightLoss)
		# tf.summary.scalar("totalLoss", totalLoss.item())
		summaryWriter.add_scalar("totalLoss", totalLoss.item())

		self.loss = totalLoss
