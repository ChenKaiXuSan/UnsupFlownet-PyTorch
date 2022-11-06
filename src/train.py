from util import checkResume
from util import TrainingLoss, solver, sessionSetup, learningRateSchedule, unsupLossBSchedule
from datasets import TrainingData
from datasets import example_dataset, image_augmentation

import sys
import time
import datetime
import os
import argparse
import json

# pytorch model
import models
import torch
from torch.utils.tensorboard import SummaryWriter

# parse command line args
argParser = argparse.ArgumentParser(description="start/resume training")
argParser.add_argument("-l", "--logDev", dest="logDev", action="store_true")
argParser.add_argument("-g", "--gpu", dest="gpu",
                       action="store", default=0, type=int)
argParser.add_argument("-i", "--iterations", dest="iterations",
                       action="store", default=0, type=int)
argParser.add_argument("-r", "--resume", dest="resume", action="store_true")
cmdArgs = argParser.parse_args()

# multi gpu management
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(cmdArgs.gpu)

# load instance params, set instance path for logs and snapshots
with open("/workspace/UnsupFlownet-PyTorch/src/hyperParams.json") as f:
    instanceParams = json.load(f)
logPath = "/workspace/UnsupFlownet-PyTorch/src/logs"
snapshotPath = "/workspace/UnsupFlownet-PyTorch/src/snapshots"

# training settings helpers
printFrequency = instanceParams["printFreq"]
snapshotFrequency = instanceParams["snapFreq"]
batchSize = instanceParams["batchSize"]

iterations = instanceParams["iterations"]
baseLearningRate = instanceParams["baseLR"]
learningRate = baseLearningRate
snapshotFrequency = instanceParams["snapshotFreq"]

# iteration override
if cmdArgs.iterations > 0:
    print("overriding max training iterations from commandline argument")
    iterations = cmdArgs.iterations

# check for resume
resume, startIteration, snapshotFiles = checkResume(
    snapshotPath, logPath, cmdArgs)

# bulid model
networkBody = models.__dict__['flownets']().cuda()

optimizer = torch.optim.Adam(networkBody.parameters())

# data loading code
trainingDataloader = example_dataset(batchSize, instanceParams)

# start summary writer
summary_writer = SummaryWriter(log_dir=logPath)

# run
lastPrint = time.time()
for i in range(startIteration, iterations):

    # scheduled values
    learningRate = learningRateSchedule(baseLearningRate, i)
    recLossBWeight = unsupLossBSchedule(i)

    for j, (imgs, _) in enumerate(trainingDataloader, 0):
		
        imgs = imgs.cuda()
        optimizer.zero_grad()

        flows_F = networkBody(imgs) # flows, b, 2, h, w
        flows_B = networkBody(imgs, flipInput=True) # flows, b, 2, h, w

        frame0, frame1, validMask = image_augmentation(imgs, batchSize, instanceParams)
        
        # calc the loss
        trainingLoss = TrainingLoss(
            instanceParams, flows_F, flows_B, frame0, frame1, validMask, summary_writer)

        trainingLoss.loss.backward()

        optimizer.step()


	# run training
    if (i+1) % printFrequency == 0:
        timeDiff = time.time() - lastPrint
        itPerSec = printFrequency/timeDiff
        remainingIt = iterations - i
        eta = remainingIt/itPerSec
        print("Iteration "+str(i+1)+": loss: "+str(trainingLoss.loss)+", iterations per second: " +
                str(itPerSec)+", ETA: "+str(datetime.timedelta(seconds=eta))+", lr: "+str(learningRate))

        lastPrint = time.time()

    sys.stdout.flush()


# loss scheduling
# recLossBWeightTensor = trainingLoss.recLossBWeight

