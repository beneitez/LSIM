from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import numpy as np
import torch
import os
import re
import random
import scipy.ndimage
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import imageio
import math
import h5py

class DatasetDistance(Dataset):
    """
    Custom Dataset class for loading image pairs and their associated distance.
    Data is expected to be organized in subdirectories, where each subdirectory
    contains a set of images representing different levels of distortion
    relative to a reference image (implicitly the one with the lowest distance value).
    """
    def __init__(self, name, dataDirs, exclude=[], include=[], transform=None, fileType="png"):
        """
        Args:
            name (str): Name of the dataset.
            dataDirs (list): List of root directories containing the dataset subdirectories.
            exclude (list, optional): List of strings. Subdirectories containing any of these
                                      strings will be excluded. Defaults to [].
            include (list, optional): List of strings. Only subdirectories containing any of
                                      these strings will be included. Defaults to [].
            transform (callable, optional): Optional transform to be applied on a sample.
                                            Defaults to None.
            fileType (str, optional): The file extension of the image data (e.g., "png", "npz", "h5").
                                      Defaults to "png".
        """
        self.transform = transform
        self.name = name
        self.fileType = fileType
        self.dataPaths = [] # Stores paths to the individual dataset subdirectories

        print("Dataset " + name + " at " + str(dataDirs))

        # Collect paths to all relevant subdirectories based on include/exclude filters
        for dataDir in dataDirs:
            directories = os.listdir(dataDir)
            directories.sort() # Ensure consistent ordering
            for directory in directories:
                # Apply exclusion filter
                if exclude:
                    if any( item in directory for item in exclude ) :
                        continue
                # Apply inclusion filter
                if include:
                    if not any( item in directory for item in include ) :
                        continue

                currentDir = os.path.join(dataDir, directory)
                # Check if it's actually a directory before adding
                if os.path.isdir(currentDir):
                    self.dataPaths.append(currentDir)

        print("Length: %d" % len(self.dataPaths))

    def __len__(self):
        """
        Returns the total number of dataset subdirectories.
        """
        return len(self.dataPaths)

    def __getitem__(self, idx):
        """
        Loads and processes a single sample (a set of image pairs and distances)
        from a dataset subdirectory.

        Args:
            idx (int): Index of the subdirectory to load.

        Returns:
            dict: A dictionary containing:
                  - "reference": Stacked reference images (torch.Tensor).
                  - "other": Stacked distorted images (torch.Tensor).
                  - "distance": Stacked distance values (torch.Tensor).
                  - "path": Path to the loaded subdirectory (str).
        """
        directory = self.dataPaths[idx]
        fileNames = os.listdir(directory)
        fileNames.sort() # Ensure consistent file processing order

        listFrames = [] # Stores loaded image frames
        listDist = [] # Stores distance values extracted from filenames

        # Iterate through files in the subdirectory
        for fileName in fileNames:
            filePath = os.path.join(directory, fileName)
            # Skip files that don't match the fileType or are the reference file
            if not fileName.endswith(".%s" % self.fileType) or fileName == "ref.%s" % self.fileType:
                continue

            # Load frame based on file type
            if self.fileType == "png":
                frame = imageio.imread(filePath)
            elif self.fileType == "npz":
                frame = np.load(filePath)['arr_0']
            elif self.fileType == "h5":
                with h5py.File(filePath, "r+") as f:
                    # Assuming h5 contains specific fields like u, v, p, c11, etc.
                    Nx = f['x'][:].shape[0]
                    Ny = f['y'][:].shape[0]

                    # Create a multi-channel frame from h5 fields
                    frame = np.zeros((Nx, Ny, 7))
                    frame[...,0] = f['u'][:]
                    frame[...,1] = f['v'][:]
                    frame[...,2] = f['p'][:]
                    frame[...,3] = f['c11'][:]
                    frame[...,4] = f['c22'][:]
                    frame[...,5] = f['c33'][:]
                    frame[...,6] = f['c12'][:]


            # Ensure frame has a channel dimension, even for grayscale
            if frame.ndim == 2:
                frame = frame[...,None]
            # Remove alpha channel if present (assuming 4 channels is RGBA)
            if frame.shape[2] == 4:
                frame = frame[...,:-1]


            # Extract distance from filename (assuming format like 'XX.fileType')
            # This part assumes the distance is a two-digit number before the extension
            if self.fileType == "png" or self.fileType == "npz":
                start = len(fileName) - 6
                end = len(fileName) - 4
            elif self.fileType == "h5":
                start = len(fileName) - 5
                end = len(fileName) - 3
            listDist.append( float(fileName[start:end]) )
            listFrames.append(frame)

        # Assert that data was loaded from the directory
        assert(len(listDist) != 0 and len(listFrames) != 0), "%s: no data to load!" % directory

        # Normalize distances to be between 0 and 1
        distances = np.array(listDist)
        # Handle case where all distances are the same (e.g., only one file loaded)
        if np.max(distances) - np.min(distances) == 0:
             distances = np.zeros_like(distances)
        else:
            distances = (distances - np.min(distances)) / (np.max(distances) - np.min(distances))


        frames = np.array(listFrames)
        # Optional check for expected number of frames (commented out)
        #if frames.shape[0] != 11:
        #    print("%s is missing files!" % directory)

        reference = [] # List to store reference images for pairs
        other = [] # List to store distorted images for pairs
        dist = [] # List to store distance differences for pairs

        # Create pairs of images and their distance differences
        # The image with the smaller normalized distance is considered the reference
        for i in range(distances.shape[0]):
            for j in range(i+1,distances.shape[0]):
                # Calculate the absolute difference in normalized distances
                diff = np.abs(distances[j] - distances[i])
                # The image with the smaller normalized distance is the reference
                if distances[i] < distances[j]:
                    reference.append(frames[i])
                    other.append(frames[j])
                else:
                    reference.append(frames[j])
                    other.append(frames[i])
                dist.append(diff)

        # Stack the lists into numpy arrays
        sample = {"reference": np.stack(reference, 0), "other": np.stack(other, 0),
                "distance": np.stack(dist, 0), "path": directory}

        # Apply the transform if specified
        if self.transform:
            sample = self.transform(sample)
        return sample


    def setDataTransform(self, transform):
        """
        Sets or updates the transform to be applied to samples.

        Args:
            transform (callable): The transform function.
        """
        self.transform = transform

    def computeMeanAndStd(self):
        """
        Computes the mean and standard deviation of the dataset pixel values
        using an online algorithm. This is useful for normalization.

        Returns:
            tuple: A tuple containing (mean, std).
        """
        print("Computing mean and std of dataset...")
        mean = 0 # online data mean
        count = 0 # total number of pixels processed
        M2 = 0 # sum of squares of differences from the current mean (for online variance)

        # Iterate through all data paths and files to compute mean and std
        for path in self.dataPaths:
            fileNames = os.listdir(path)
            fileNames.sort()

            for fileName in fileNames:
                filePath = os.path.join(path, fileName)
                # Only process data files, skip reference and non-data files
                if not fileName.endswith(".%s" % self.fileType) or fileName == "ref.%s" % self.fileType:
                    continue

                # Load data based on file type
                if self.fileType == "png":
                    data = imageio.imread(filePath)
                elif self.fileType == "npz":
                    data = np.load(filePath)['arr_0']
                # Add h5 loading if needed for mean/std calculation

                # Remove alpha channel if present
                if data.shape[2] == 4:
                    data = data[...,:-1]

                # Update mean and M2 using Welford's online algorithm
                data_flat = data.flatten() # Flatten data for easier processing
                for x in data_flat:
                    count += 1
                    delta = x - mean
                    mean += delta / count
                    M2 += delta * (x - mean)

        # Calculate standard deviation from M2 and count
        # Handle case with only one data point to avoid division by zero
        if count < 2:
            std = 0
        else:
            std = np.sqrt(M2 / (count-1))

        self.mean, self.std = [mean, std] # Store computed mean and std

        return mean, std

# -------------------------------------------------
# TRANSFORMS TO APPLY TO THE DATA
# -------------------------------------------------

# combines randomFlip, randomRotation90, randomCrop,
# channelSwap, toTensor and normalization for efficiency
class TransformsTrain(object):
    """
    A transform for training data, applying random augmentations
    (flip, rotation, channel swap, crop) and normalization.
    Operates on numpy arrays and converts to torch.Tensor.
    """
    def __init__(self, outputSize, normMin=0, normMax=255):
        """
        Args:
            outputSize (int): The target spatial size (height and width) after cropping.
            normMin (float, optional): Minimum value for normalization. Defaults to 0.
            normMax (float, optional): Maximum value for normalization. Defaults to 255.
        """
        self.outputSize = outputSize
        self.normMin = normMin
        self.normMax = normMax
        self.angles = [0,1,2,3] # Possible 90-degree rotation angles

    def __call__(self, sample):
        """
        Applies the transformations to a sample.

        Args:
            sample (dict): A sample dictionary from DatasetDistance.__getitem__.

        Returns:
            dict: The transformed sample dictionary with torch.Tensors.
        """
        dist = sample["distance"]
        reference = sample["reference"] # numpy array (N, H, W, C)
        other = sample["other"]       # numpy array (N, H, W, C)
        path = sample["path"]

        # Initialize arrays for transformed results
        resultRef = np.zeros([reference.shape[0], self.outputSize, self.outputSize, reference.shape[3]])
        resultOther = np.zeros([other.shape[0], self.outputSize, self.outputSize, other.shape[3]])

        # Apply transformations to each image pair in the sample
        for i in range(reference.shape[0]):
            ref = reference[i]
            oth = other[i]

            # Random horizontal and vertical flips
            rand = random.randint(0, 3)
            if rand == 1: # Horizontal flip
                ref = np.fliplr(ref)
                oth = np.fliplr(oth)
            if rand == 2: # Vertical flip
                ref = np.flipud(ref)
                oth = np.flipud(oth)
            if rand == 3: # Both flips
                ref = np.flipud( np.fliplr(ref) )
                oth = np.flipud( np.fliplr(oth) )

            # Random 90-degree rotation
            angle = random.choice(self.angles)
            ref = np.rot90(ref, angle)
            oth = np.rot90(oth, angle)

            # Random channel swap (for 3-channel images)
            if ref.shape[2] == 3: # Only swap if it's an RGB-like image
                channelOrder = [0,1,2,3,4,5,6]
                random.shuffle(channelOrder)
                ref = ref[..., channelOrder]
                oth = oth[..., channelOrder]

            # Random crop
            # If image is smaller than output size, just use the original
            if ref.shape[0] <= self.outputSize or ref.shape[1] <= self.outputSize:
                # Pad if necessary to reach outputSize? Current code just assigns.
                # This might cause issues if original is smaller than outputSize.
                # Assuming original is >= outputSize or padding is handled elsewhere.
                resultRef[i] = ref
                resultOther[i] = oth
                continue

            # Calculate random top-left corner for the crop
            top = np.random.randint(0, ref.shape[0] - self.outputSize)
            left = np.random.randint(0, ref.shape[1] - self.outputSize)

            # Apply crop
            resultRef[i] = ref[top : top+self.outputSize,  left : left+self.outputSize]
            resultOther[i] = oth[top : top+self.outputSize,  left : left+self.outputSize]


        # Normalization: Scale pixel values to [normMin, normMax] based on min/max in the batch
        # Calculate min and max across all images and spatial dimensions in the batch
        dMin = np.minimum( np.min(resultRef, axis=(0,1,2)), np.min(resultOther, axis=(0,1,2)) )
        dMax = np.maximum( np.max(resultRef, axis=(0,1,2)), np.max(resultOther, axis=(0,1,2)) )


        # Apply normalization per channel
        # Handle cases where min == max for a channel
        if (dMin == dMax).all(): # All channels have constant value
            resultRef = resultRef - dMin
            resultOther = resultOther - dMin
        elif (dMin == dMax).any(): # Some channels have constant value
            for i in range(dMin.shape[0]):
                if dMin[i] == dMax[i]:
                    resultRef[..., i] = resultRef[..., i] - dMin[i]
                    resultOther[..., i] = resultOther[..., i] - dMin[i]
                else:
                    resultRef[..., i] = (self.normMax - self.normMin) * ( (resultRef[..., i] - dMin[i]) / (dMax[i] - dMin[i]) ) + self.normMin
                    resultOther[..., i] = (self.normMax - self.normMin) * ( (resultOther[..., i] - dMin[i]) / (dMax[i] - dMin[i]) ) + self.normMin
        else: # No channels have constant value
            resultRef = (self.normMax - self.normMin) * ( (resultRef - dMin) / (dMax - dMin) ) + self.normMin
            resultOther = (self.normMax - self.normMin) * ( (resultOther - dMin) / (dMax - dMin) ) + self.normMin


        # Convert numpy arrays to torch.Tensors and change channel order (HWC to CHW)
        resultRef = torch.from_numpy(resultRef.transpose(0,3,1,2)).float()
        resultOther = torch.from_numpy(resultOther.transpose(0,3,1,2)).float()
        dist = torch.from_numpy(np.array(dist)).float() # Convert distance list to tensor

        return {"reference": resultRef, "other": resultOther, "distance": dist, "path": path}


# combines resize, toTensor and normalization for efficiency
class TransformsInference(object):
    """
    A transform for inference data, applying resizing and normalization.
    Operates on numpy arrays and converts to torch.Tensor.
    """
    def __init__(self, outputSize, order, normMin = 0, normMax = 255):
        """
        Args:
            outputSize (int): The target spatial size (height and width) after resizing.
            order (int): The order of the spline interpolation (e.g., 0 for nearest, 1 for linear).
            normMin (float, optional): Minimum value for normalization. Defaults to 0.
            normMax (float, optional): Maximum value for normalization. Defaults to 255.
        """
        self.normMin = normMin
        self.normMax = normMax
        self.outputSize = outputSize
        self.order = order # Interpolation order for resizing

    def __call__(self, sample):
        """
        Applies the transformations to a sample.

        Args:
            sample (dict): A sample dictionary from DatasetDistance.__getitem__.

        Returns:
            dict: The transformed sample dictionary with torch.Tensors.
        """
        dist = sample["distance"]
        reference = sample["reference"] # numpy array (N, H, W, C)
        other = sample["other"]       # numpy array (N, H, W, C)
        path = sample["path"]

        # Repeat scalar fields across 3 channels if necessary (e.g., for grayscale input to RGB models)
        if reference.shape[reference.ndim-1] == 1:
            reference = np.repeat(reference, 3, axis=reference.ndim-1)
        if other.shape[other.ndim-1] == 1:
            other = np.repeat(other, 3, axis=other.ndim-1)

        # Resize images if outputSize is specified and different from current size
        if self.outputSize and (self.outputSize != reference.shape[1] or self.outputSize != reference.shape[2]):
            # Initialize arrays for resized results
            resultRef = np.zeros([reference.shape[0], self.outputSize, self.outputSize, reference.shape[3]])
            resultOther = np.zeros([other.shape[0], self.outputSize, self.outputSize, other.shape[3]])

            # Calculate zoom factors for scipy.ndimage.zoom
            zoom1 = [1, self.outputSize / reference.shape[1], self.outputSize / reference.shape[2], 1] # [batch, height, width, channels]
            resultRef = scipy.ndimage.zoom(reference, zoom1, order=self.order)
            zoom2 = [1, self.outputSize / other.shape[1], self.outputSize / other.shape[2], 1]
            resultOther = scipy.ndimage.zoom(other, zoom2, order=self.order)
        else:
            # If no resizing needed, just use original arrays
            resultRef = reference
            resultOther = other

        # Normalization: Scale pixel values to [normMin, normMax] based on min/max in the batch
        # Calculate min and max across all images and spatial dimensions in the batch
        dMin = np.minimum( np.min(resultRef, axis=(0,1,2)), np.min(resultOther, axis=(0,1,2)) )
        dMax = np.maximum( np.max(resultRef, axis=(0,1,2)), np.max(resultOther, axis=(0,1,2)) )

        # Apply normalization per channel
        # Handle cases where min == max for a channel
        if (dMin == dMax).all(): # All channels have constant value
            resultRef = resultRef - dMin
            resultOther = resultOther - dMin
        elif (dMin == dMax).any(): # Some channels have constant value
            for i in range(dMin.shape[0]):
                if dMin[i] == dMax[i]:
                    resultRef[..., i] = resultRef[..., i] - dMin[i]
                    resultOther[..., i] = resultOther[..., i] - dMin[i]
                else:
                    resultRef[..., i] = (self.normMax - self.normMin) * ( (resultRef[..., i] - dMin[i]) / (dMax[i] - dMin[i]) ) + self.normMin
                    resultOther[..., i] = (self.normMax - self.normMin) * ( (resultOther[..., i] - dMin[i]) / (dMax[i] - dMin[i]) ) + self.normMin
        else: # No channels have constant value
            resultRef = (self.normMax - self.normMin) * ( (resultRef - dMin) / (dMax - dMin) ) + self.normMin
            resultOther = (self.normMax - self.normMin) * ( (resultOther - dMin) / (dMax - dMin) ) + self.normMin

        # Convert numpy arrays to torch.Tensors and change channel order (HWC to CHW)
        resultRef = torch.from_numpy(resultRef.transpose(0,3,1,2)).float()
        resultOther = torch.from_numpy(resultOther.transpose(0,3,1,2)).float()
        dist = torch.from_numpy(np.array(dist)).float() # Convert distance list to tensor

        return {"reference": resultRef, "other": resultOther, "distance": dist, "path": path}

       