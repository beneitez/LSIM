"""
This script sets up and runs the training process for a distance model.
It includes data loading, model initialization, defining loss and optimizer,
and the main training and validation loop.
"""

import numpy as np
import os
import torch
from torch.utils.data import DataLoader

# Import necessary modules from the LSIM package
from LSIM.dataset_distance import *
from LSIM.distance_model import *
from LSIM.distance_model_non_siamese import *
from LSIM.loss import *
from LSIM.trainer import *


# --- SETUP FOR DATA AND MODEL ---

# Configure CUDA device visibility and GPU usage
os.environ["CUDA_VISIBLE_DEVICES"]="0" # Specifies which GPU to use (GPU 0 in this case)
useGPU = True # Flag to indicate if GPU should be used

# --- DATASET SETUP ---

# Initialize the training dataset
# It loads data from specified directories and excludes certain files
trainSet = DatasetDistance("Training", dataDirs=["Data/Smoke", "Data/BurgersEq", "Data/AdvDiff", "Data/Liquid"],
                                exclude=["plume1.", "plume2.", "plume11.", "plume12.",
                                    "burgersEq1.", "burgersEq2.", "burgersEq3.", "burgersEq4.",
                                    "burgersEq5.", "burgersEq6.", "burgersEq7.", "burgersEq8.",
                                    "burgersEq9.", "burgersEq10.", "burgersEq11.", "burgersEq12.",
                                    "advDiff1.", "advDiff2.", "advDiff3.", "advDiff4.", "advDiff5.",
                                    "advDiff6.", "advDiff7.", "advDiff8.", "advDiff9.", "advDiff10.",
                                    "drop1.", "drop2.", "drop3.", "drop21.", "drop22.", "drop23."],)
# Initialize the validation dataset
# It loads data from specified directories and includes certain files (likely the ones excluded from training)
valSet = DatasetDistance("Validation", dataDirs=["Data/Smoke", "Data/BurgersEq", "Data/AdvDiff", "Data/Liquid"],
                                include=["plume1.", "plume2.", "plume11.", "plume12.",
                                    "burgersEq1.", "burgersEq2.", "burgersEq3.", "burgersEq4.",
                                    "burgersEq5.", "burgersEq6.", "burgersEq7.", "burgersEq8.",
                                    "burgersEq9.", "burgersEq10.", "burgersEq11.", "burgersEq12.",
                                    "advDiff1.", "advDiff2.", "advDiff3.", "advDiff4.", "advDiff5.",
                                    "advDiff6.", "advDiff7.", "advDiff8.", "advDiff9.", "advDiff10.",
                                    "drop1.", "drop2.", "drop3.", "drop21.", "drop22.", "drop23."],)

# --- DATA TRANSFORMATIONS ---

# Define transformations for training data (e.g., resizing, normalization)
transTrain = TransformsTrain(224, normMin=0, normMax=255)
# Define transformations for validation data (e.g., resizing, normalization)
transVal = TransformsInference(224, 0, normMin=0, normMax=255)
# Apply the defined transformations to the datasets
trainSet.setDataTransform(transTrain)
valSet.setDataTransform(transVal)

# --- DATA LOADERS ---

# Create DataLoader for the training set
# batch_size=1 means one pair of images per batch
# shuffle=True shuffles the data for better training
# num_workers=4 uses 4 subprocesses for data loading
trainLoader = DataLoader(trainSet, batch_size=1, shuffle=True, num_workers=4)
# Create DataLoader for the validation set
# shuffle=False as shuffling is not needed for validation
valLoader = DataLoader(valSet, batch_size=1, shuffle=False, num_workers=4)

# --- MODEL INITIALIZATION ---

# Initialize the DistanceModel
# baseType="lsim": Specifies the base network architecture (LSIM)
# initBase="pretrained": Initializes the base network with pretrained weights
# initLin=0.1: Initialization scale for linear layers
# featureDistance="L2": Uses L2 distance for feature comparison
# frozenLayers=[]: No layers are frozen during training
# normMode="normDist": Normalization mode
# useNormUpdate=False: Flag to control normalization update
# isTrain=True: Indicates the model is in training mode
# useGPU=useGPU: Uses GPU if the flag is True
model = DistanceModel(baseType="lsim", initBase="pretrained", initLin=0.1, featureDistance="L2",
                frozenLayers=[], normMode="normDist", useNormUpdate=False, isTrain=True, useGPU=useGPU)
# Print the number of trainable parameters in the model
model.printNumParams()

# --- LOSS FUNCTION AND OPTIMIZER ---

# Define the criterion (loss function)
# Uses CorrelationLoss with specified weights for different components (MSE, Correlation, Cross-Correlation)
criterion = CorrelationLoss(weightMSE=0.3, weightCorr=0.7, weightCrossCorr=0.0)
# Define the optimizer
# Uses Adam optimizer with a learning rate of 0.00001 and no weight decay
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001, weight_decay=0.0)

# --- TRAINER AND VALIDATOR SETUP ---

# Initialize the Trainer object
# Handles the training loop, model updates, etc.
# printEvery=800: Prints training progress every 800 batches
# showProgressPrint=False: Disables detailed progress printing
trainer = Trainer(model, trainLoader, optimizer, criterion, 800, False)
# Initialize the Validator object
# Handles the validation process
validator = Validator(model, valLoader, criterion)


# --- ACTUAL TRAINING ---

print('Starting Training')

# Perform normalization calibration if the normalization mode is not "normUnit"
# This step might be used to compute running statistics for normalization layers
if model.normMode != "normUnit":
    trainer.normCalibration(1, stopEarly=0)

# Main training loop over epochs
for epoch in range(0, 40):
    # Perform validation every 5 epochs (starting from epoch 1)
    if epoch % 5 == 1:
        validator.validationStep()

    # Perform one training step (one epoch)
    trainer.trainingStep(epoch+1)

    # Save a temporary model checkpoint after each epoch
    model.save("Models/TrainedLSiM_tmp.pth", override=True, noPrint=True)

print('Finished Training')
# Save the final trained model
model.save("Models/TrainedLSiM.pth")

trainSet = DatasetDistance("Training", dataDirs=["Data/Smoke", "Data/BurgersEq", "Data/AdvDiff", "Data/Liquid"],
                                exclude=["plume1.", "plume2.", "plume11.", "plume12.",
                                    "burgersEq1.", "burgersEq2.", "burgersEq3.", "burgersEq4.",
                                    "burgersEq5.", "burgersEq6.", "burgersEq7.", "burgersEq8.",
                                    "burgersEq9.", "burgersEq10.", "burgersEq11.", "burgersEq12.",
                                    "advDiff1.", "advDiff2.", "advDiff3.", "advDiff4.", "advDiff5.",
                                    "advDiff6.", "advDiff7.", "advDiff8.", "advDiff9.", "advDiff10.",
                                    "drop1.", "drop2.", "drop3.", "drop21.", "drop22.", "drop23."],)
valSet = DatasetDistance("Validation", dataDirs=["Data/Smoke", "Data/BurgersEq", "Data/AdvDiff", "Data/Liquid"],
                                include=["plume1.", "plume2.", "plume11.", "plume12.",
                                    "burgersEq1.", "burgersEq2.", "burgersEq3.", "burgersEq4.",
                                    "burgersEq5.", "burgersEq6.", "burgersEq7.", "burgersEq8.",
                                    "burgersEq9.", "burgersEq10.", "burgersEq11.", "burgersEq12.",
                                    "advDiff1.", "advDiff2.", "advDiff3.", "advDiff4.", "advDiff5.",
                                    "advDiff6.", "advDiff7.", "advDiff8.", "advDiff9.", "advDiff10.",
                                    "drop1.", "drop2.", "drop3.", "drop21.", "drop22.", "drop23."],)

transTrain = TransformsTrain(224, normMin=0, normMax=255)
transVal = TransformsInference(224, 0, normMin=0, normMax=255)
trainSet.setDataTransform(transTrain)
valSet.setDataTransform(transVal)

trainLoader = DataLoader(trainSet, batch_size=1, shuffle=True, num_workers=4)
valLoader = DataLoader(valSet, batch_size=1, shuffle=False, num_workers=4)

model = DistanceModel(baseType="lsim", initBase="pretrained", initLin=0.1, featureDistance="L2",
                frozenLayers=[], normMode="normDist", useNormUpdate=False, isTrain=True, useGPU=useGPU)
model.printNumParams()

criterion = CorrelationLoss(weightMSE=0.3, weightCorr=0.7, weightCrossCorr=0.0)
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001, weight_decay=0.0)
trainer = Trainer(model, trainLoader, optimizer, criterion, 800, False)
validator = Validator(model, valLoader, criterion)


# ACTUAL TRAINING
print('Starting Training')

if model.normMode != "normUnit":
    trainer.normCalibration(1, stopEarly=0)

for epoch in range(0, 40):
    if epoch % 5 == 1:
        validator.validationStep()

    trainer.trainingStep(epoch+1)

    model.save("Models/TrainedLSiM_tmp.pth", override=True, noPrint=True)

print('Finished Training')
model.save("Models/TrainedLSiM.pth")
