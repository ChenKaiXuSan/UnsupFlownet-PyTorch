import torch

def charbonnierLoss(x,alpha,beta,epsilon):
	epsilonSq = epsilon*epsilon
	xScale = x*beta

	return torch.pow(xScale * xScale + epsilonSq, alpha)