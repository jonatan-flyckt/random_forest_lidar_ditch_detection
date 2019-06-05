# General arr lib
import numpy as np

# Compile py code
from numba import jit
from numba import prange

# Multithreading
import dask.array as da
import dask as dk
dk.config.set(scheduler='processes')

from dask_image.ndfilters import generic_filter as d_gf

from collections import deque

from skimage.filters import gabor
from skimage.restoration import  denoise_bilateral

import datetime

import numpy as np
from numpy import random, nanmax, argmax, unravel_index
from scipy.spatial.distance import pdist, squareform
from scipy.ndimage.filters import generic_filter as gf
import scipy.stats.mstats as ms
from scipy.stats import skew
import scipy.ndimage.morphology as morph
from scipy import ndimage
from PIL import Image
import scipy
import matplotlib.pyplot as plt
import re
import os
import pandas as pd
import random
import math
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral, denoise_wavelet, estimate_sigma)
from skimage import data, img_as_float
from skimage.filters import gabor
from skimage.util import random_noise
from collections import deque
from sklearn.ensemble import RandomForestClassifier
from joblib import dump, load
from sklearn.metrics import cohen_kappa_score, accuracy_score, recall_score, confusion_matrix, precision_score
from numba import jit
from numba import prange

from Functions import general_functions

@jit(nopython=True)
def raster_to_zones(arr, zoneSize, threshold):
    """
    Converts binary pixel labels to zones with a specified zone size and threshold.
    """
    new_arr = arr.copy()
    for i in range(0, len(arr), zoneSize):
        for j in range(0, len(arr[i]), zoneSize):
            numberOfClassified = 0
            if i < len(arr) - zoneSize and j < len(arr[i]) - zoneSize:
                for k in range(zoneSize):
                    for l in range(zoneSize):
                        if arr[i + k][j + l] == 1:
                            numberOfClassified += 1
                if numberOfClassified > (zoneSize**2)/threshold:
                    for k in range(zoneSize):
                        for l in range(zoneSize):
                            new_arr[i + k][j + l] = 1
                else:
                    for k in range(zoneSize):
                        for l in range(zoneSize):
                            new_arr[i + k][j + l] = 0
    return new_arr


@jit(nopython=True)
def proba_to_zones(arr, zoneSize, threshold):
    """
    Converts continuous pixel probability values to zones with a specified zone size and threshold.
    """
    new_arr = np.zeros(arr.shape)
    for i in range(0, len(arr), zoneSize):
        for j in range(0, len(arr[i]), zoneSize):
            totalProba = 0
            if i < len(arr) - zoneSize and j < len(arr[i]) - zoneSize:
                for k in range(zoneSize):
                    for l in range(zoneSize):
                        totalProba += arr[i+k][j+l]
                if totalProba / zoneSize**2 > threshold:
                    for k in range(zoneSize):
                        for l in range(zoneSize):
                            new_arr[i + k][j + l] = 1
                else:
                    for k in range(zoneSize):
                        for l in range(zoneSize):
                            new_arr[i + k][j + l] = 0
    return new_arr


@jit(nopython=True)
def _custom_remove_noise(arr, max_arr, new_arr, threshold, selfThreshold):
    """
    Internal noise removal function of probability prediction.
    """
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            if max_arr[i][j] < threshold:
                if arr[i][j] > selfThreshold:
                    new_arr[i][j] *= 0.5
                else:
                    new_arr[i][j] *= 0.25
    return new_arr


@jit
def custom_remove_noise(arr, radius, threshold, selfThreshold):
    """
    Removes noise from a probability prediction.
    """
    max_arr = d_gf(da.from_array(arr, chunks=(800, 800)), np.nanmax,
                   footprint=general_functions.create_circular_mask(radius)).compute(scheduler='processes')
    return _custom_remove_noise(arr, max_arr, np.copy(arr), threshold, selfThreshold)

def find_max_distance(A):
    """
    Returns the maximum distance from  2x points.
    Each point is represented by an x,y coordinate.
    """
    return nanmax(squareform(pdist(A)))


def remove_clusters(arr, zoneSize, lowerIslandThreshold, upperIslandThreshold, ratioThreshold):
    """
    Removes noise in the form of small clusters or clusters of a non ditch shape from a binarized prediction.
    """
    new_arr = arr.copy()
    examinedPoints = set()
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            if arr[i][j] == 1 and (i, j) not in examinedPoints:
                cluster = _get_cluster_array(arr, (i, j), zoneSize)
                cluster_size = len(cluster)
                if cluster_size < upperIslandThreshold:
                    cluster_distance = find_max_distance(cluster)
                for k in range(cluster_size):
                    examinedPoints.add(cluster[k])
                    if cluster_size < upperIslandThreshold:
                        if cluster_size < lowerIslandThreshold:
                            new_arr[cluster[k][0]][cluster[k][1]] = 0
                        elif cluster_size / cluster_distance > ratioThreshold:
                            new_arr[cluster[k][0]][cluster[k][1]] = 0
    return new_arr      
        
@jit
def _get_cluster_array(arr, index, zoneSize):
    """
    Internal function to receive an array of all the pixels of a binary cluster.
    """
    arrayOfPoints = []
    iMax = len(arr) - 1
    jMax = len(arr[0]) - 1
    i = index[0]
    j = index[1]
    FIFOQueue = deque([(i, j)])
    examinedElements = set()
    examinedElements.add((i, j))
    while (len(FIFOQueue) > 0):
        currentIndex = FIFOQueue.popleft()
        i = currentIndex[0]
        j = currentIndex[1]
        if i >= 0 and i < iMax and j >= 0 and j < jMax and arr[i][j] == 1:
            arrayOfPoints.append((i, j))
            # add horizontally and vertically
            if (i+1, j) not in examinedElements:
                FIFOQueue.append((i+1, j))
                examinedElements.add((i+1, j))
            if (i-1, j) not in examinedElements:
                FIFOQueue.append((i-1, j))
                examinedElements.add((i-1, j))
            if (i, j+1) not in examinedElements:
                FIFOQueue.append((i, j+1))
                examinedElements.add((i, j+1))
            if (i, j-1) not in examinedElements:
                FIFOQueue.append((i, j-1))
                examinedElements.add((i, j-1))
            # add diagonally
            if (i+1, j+1) not in examinedElements:
                FIFOQueue.append((i+1, j+1))
                examinedElements.add((i+1, j+1))
            if (i-1, j+1) not in examinedElements:
                FIFOQueue.append((i-1, j+1))
                examinedElements.add((i-1, j+1))
            if (i+1, j-1) not in examinedElements:
                FIFOQueue.append((i+1, j-1))
                examinedElements.add((i+1, j-1))
            if (i-1, j-1) not in examinedElements:
                FIFOQueue.append((i-1, j-1))
                examinedElements.add((i-1, j-1))

            # Add one zone away
            # add horizontally and vertically
            if (i+1 + zoneSize, j) not in examinedElements:
                FIFOQueue.append((i+1 + zoneSize, j))
                examinedElements.add((i+1 + zoneSize, j))
            if (i-1 - zoneSize, j) not in examinedElements:
                FIFOQueue.append((i-1 - zoneSize, j))
                examinedElements.add((i-1 - zoneSize, j))
            if (i, j+1 + zoneSize) not in examinedElements:
                FIFOQueue.append((i, j+1 + zoneSize))
                examinedElements.add((i, j+1 + zoneSize))
            if (i, j-1 - zoneSize) not in examinedElements:
                FIFOQueue.append((i, j-1 - zoneSize))
                examinedElements.add((i, j-1 - zoneSize))
            # add diagonally
            if (i+1 + zoneSize, j+1 + zoneSize) not in examinedElements:
                FIFOQueue.append((i+1 + zoneSize, j+1 + zoneSize))
                examinedElements.add((i+1 + zoneSize, j+1 + zoneSize))
            if (i-1 - zoneSize, j+1 + zoneSize) not in examinedElements:
                FIFOQueue.append((i-1 - zoneSize, j+1 + zoneSize))
                examinedElements.add((i-1 - zoneSize, j+1 + zoneSize))
            if (i+1 + zoneSize, j-1 - zoneSize) not in examinedElements:
                FIFOQueue.append((i+1 + zoneSize, j-1 - zoneSize))
                examinedElements.add((i+1 + zoneSize, j-1 - zoneSize))
            if (i-1 - zoneSize, j-1 - zoneSize) not in examinedElements:
                FIFOQueue.append((i-1 - zoneSize, j-1 - zoneSize))
                examinedElements.add((i-1 - zoneSize, j-1 - zoneSize))
    return arrayOfPoints


@jit(nopython=True)
def _proba_mean_from_masks(arr, row, col, masks):
    """
    Internal function to get the mean value of a set of conic masks from a pixel.
    """
    halfMask = len(masks[0]) // 2    
    arrLenRow = len(arr)
    arrLenCol = len(arr[row])
    values = np.zeros(8)
    elementAmounts = np.zeros(8)
    for i in range(-halfMask , halfMask):
        for j in range(-halfMask , halfMask):
            if arrLenCol > col + j + 1 and col + j + 1 >= 0 and arrLenRow > row + i + 1 and row + i + 1 >= 0:
                if masks[0][i + halfMask][j + halfMask] == 1:
                    values[0] += arr[row + i][col + j]
                    elementAmounts[0] += 1
                elif masks[1][i + halfMask][j + halfMask] == 1:
                    values[1] += arr[row + i][col + j]
                    elementAmounts[1] += 1
                elif masks[2][i + halfMask][j + halfMask] == 1:
                    values[2] += arr[row + i][col + j]
                    elementAmounts[2] += 1
                elif masks[3][i + halfMask][j + halfMask] == 1:
                    values[3] += arr[row + i][col + j]
                    elementAmounts[3] += 1
                elif masks[4][i + halfMask][j + halfMask] == 1:
                    values[4] += arr[row + i][col + j]
                    elementAmounts[4] += 1
                elif masks[5][i + halfMask][j + halfMask] == 1:
                    values[5] += arr[row + i][col + j]
                    elementAmounts[5] += 1
                elif masks[6][i + halfMask][j + halfMask] == 1:
                    values[6] += arr[row + i][col + j]
                    elementAmounts[6] += 1
                elif masks[7][i + halfMask][j + halfMask] == 1:
                    values[7] += arr[row + i][col + j]
                    elementAmounts[7] += 1
    for i in range(len(values)):
        values[i] = values[i] / elementAmounts[i] if elementAmounts[i] != 0 else 0
    return values


@jit(nopython=True)
def _conic_proba_post_processing(arr, maxArr, masks, threshold):
    """
    Internal function to fill the gaps of a continuous probability prediction array.
    """
    new_arr = arr.copy()
    amountOfUpdated = 0
    examinedPoints = 0
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            if arr[i][j] < 0.5 and maxArr[i][j] > 0.6:
                examinedPoints += 1
                trueProba = _proba_mean_from_masks(arr, i, j, masks)
                
                updatePixel = 0
                if trueProba[0] > threshold and trueProba[4] > threshold:
                    updatePixel = trueProba[0] if trueProba[0] > trueProba[4] else trueProba[4]
                if trueProba[1] > threshold and trueProba[5] > threshold:
                    updatePixelAgain = trueProba[1] if trueProba[1] > trueProba[5] else trueProba[5]
                    if updatePixelAgain > updatePixel:
                        updatePixel = updatePixelAgain
                if trueProba[2] > threshold and trueProba[6] > threshold:
                    updatePixelAgain = trueProba[2] if trueProba[6] > trueProba[2] else trueProba[6]
                    if updatePixelAgain > updatePixel:
                        updatePixel = updatePixelAgain
                if trueProba[3] > threshold and trueProba[7] > threshold:
                    updatePixelAgain = trueProba[3] if trueProba[3] > trueProba[7] else trueProba[7]
                    if updatePixelAgain > updatePixel:
                        updatePixel = updatePixelAgain
                if updatePixel != 0:
                    amountOfUpdated += 1
                    if updatePixel < 0.5:
                        updatePixel *= 1.4
                    elif updatePixel < 0.55:
                        updatePixel *= 1.35
                    elif updatePixel < 0.6:
                        updatePixel *= 1.3
                    elif updatePixel < 0.65:
                        updatePixel *= 1.25
                    elif updatePixel < 0.7:
                        updatePixel *= 1.2
                    elif updatePixel < 0.75:
                        updatePixel *= 1.15
                    elif updatePixel < 0.85:
                        updatePixel *= 1.1
                    elif updatePixel < 0.9:
                        updatePixel *= 1.05
                    new_arr[i][j] = updatePixel
    return new_arr

@jit
def conic_proba_post_processing(arr, maskRadius, threshold):
    """
    Attempts to fill the gaps of a continuous probability prediction array.
    """
    masks = []
    maxArr = d_gf(da.from_array(arr,chunks = (800,800)), np.nanmax, footprint=general_functions.create_circular_mask(5))
    for i in range(0, 8):
        masks.append(create_conic_mask(maskRadius, i))

    return _conic_proba_post_processing(np.array(arr), np.array(maxArr), np.array(masks),threshold)
    
def _denoise_bilateral(arr):
    """
    Internal first noise removal step of a continuous probability prediction.
    """
    return denoise_bilateral(arr, sigma_spatial=15, multichannel=False)

def proba_noise_reduction(arr):
    """
    Removes noise from a continuous probability prediction.
    """
    d = da.from_array(arr, chunks=(800,800))
    return custom_remove_noise(d.map_overlap(_denoise_bilateral, depth=15).compute(), 10, 0.7, 0.4)
    

def proba_post_process(arr, zoneSize, probaThreshold):
    """
    Processes a continuous probability prediction in various ways to receive a final binarized ditch prediction.
    """
    deNoise = proba_noise_reduction(arr)
    gapFilled = conic_proba_post_processing(conic_proba_post_processing(deNoise, 8, 0.35), 5, 0.3)
    zonesArr = proba_to_zones(gapFilled, zoneSize, 0.4)
    noIslands = remove_clusters(zonesArr, zoneSize*2, 1500, 10000, 30)
    noIslands = remove_clusters(noIslands, zoneSize, 1000, 5000, 20)
    noIslands = remove_clusters(noIslands, 0, 500, 3000, 18)
    noIslands = remove_clusters(noIslands, 0, 500, 1200, 14)
    return noIslands

def yield_training_test_zones(list_of_files):
    folds = len(list_of_files)
    arr = list_of_files.copy()
    for i in range(folds):
      training, testing = np.delete(arr, i, 0), arr[i]
      yield (training, testing)