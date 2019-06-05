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


import general_functions


def dem_ditch_detection(arr):
    """
    DEM ditch enhancement.
    """
    newArr = arr.copy()
    maxArr = gf(arr, np.amax, footprint=create_circular_mask(30))
    minArr = gf(arr, np.amin, footprint=create_circular_mask(10))
    meanArr = gf(arr, np.median, footprint=create_circular_mask(10))
    minMaxDiff = arr.copy()
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            if minArr[i][j] < maxArr[i][j] - 3:
                minMaxDiff[i][j] = 1
            else:
                minMaxDiff[i][j] = 0
    closing = morph.binary_closing(minMaxDiff, structure=create_circular_mask(10))
    closing2 = morph.binary_closing(closing, structure=create_circular_mask(10))
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            if arr[i][j] < meanArr[i][j] - 0.1:
                newArr[i][j] = meanArr[i][j] - arr[i][j]
            else:
                newArr[i][j] = 0
            if closing2[i][j] == 1:
                newArr[i][j] = 0
    return newArr

def sky_view_hpmf_gabor_stream_removal(feature, streamAmp):
    """
    Takes a SkyViewFactor or HPMF feature and a stream amplification to create a new feature with streams weakened.
    """
    conicStreamRemoval = feature.copy()
    maxVal = np.amax(feature)
    for i in range(len(conicStreamRemoval)):
        for j in range(len(conicStreamRemoval[i])):
            print(type(streamAmp))
            print(type(streamAmp[i][j]))
            print(streamAmp)
            if streamAmp[i][j] != 0:
                conicStreamRemoval[i][j] += (streamAmp[i][j] / 2) * maxVal
                if conicStreamRemoval[i][j] > maxVal:
                    conicStreamRemoval[i][j] = maxVal
    return conicStreamRemoval

def impoundment_dem_stream_removal(impFeature, streamAmp):
    """
    Takes a DEM or Impoundment feature and a stream amplification to create a new feature with streams weakened.
    """
    impStreamRemoval = impFeature.copy()
    for i in range(len(impStreamRemoval)):
        for j in range(len(impStreamRemoval[i])):
            if streamAmp[i][j] != 0:
                impStreamRemoval[i][j] = impStreamRemoval[i][j] * (1 - (streamAmp[i][j] / 2)) if streamAmp[i][j] > 0.7 else impStreamRemoval[i][j] * 0.3
    return impStreamRemoval

def stream_amplification(arr):
    """
    Attempts to amplify only streams with impoundment index.
    """
    streamAmp = arr.copy()
    for i in range(len(streamAmp)):
        for j in range(len(streamAmp[i])):
            if streamAmp[i][j] < 14:
                streamAmp[i][j] = 0
    morphed = morph.grey_dilation(streamAmp, structure = create_circular_mask(35))
    minVal = np.amin(morphed)
    morphed -= minVal
    maxVal = np.amax(morphed)
    morphed /= maxVal if (maxVal != 0) else 1
    return morphed

@jit(nopython=True)
def _reclassify_impoundment(arr):
    """
    Internally used reclassification of impoundment index with different thresholds.
    """
    new_arr = arr.copy()
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            if new_arr[i, j] == 0:
                new_arr[i, j] = 0
            elif new_arr[i, j] < 0.002:
                new_arr[i, j] = 5
            elif arr[i, j] < 0.005:
                new_arr[i, j] = 50
            elif arr[i, j] < 0.02:
                new_arr[i, j] = 100
            elif arr[i, j] < 0.05:
                new_arr[i, j] = 1000
            elif arr[i, j] < 0.1:
                new_arr[i, j] = 10000
            elif arr[i, j] < 0.3:
                new_arr[i, j] = 100000
            else:
                new_arr[i, j] = 1000000
    return new_arr


@jit
def impoundment_amplification(arr, mask_radius=10):
    """
    Impoundment ditch enhancement.
    """
    norm_arr = da.from_array(_reclassify_impoundment(arr), chunks=(800, 800))
    mask = create_circular_mask(mask_radius)
    return d_gf(d_gf(d_gf(norm_arr, np.nanmean, footprint=mask), np.nanmean, footprint=mask), np.nanmedian, footprint=mask).compute(scheduler='processes')


@jit(nopython=True)
def _reclassify_hpmf_filter(arr):
    """
    Internally used reclassification of HPMF with different thresholds.
    """
    binary = np.copy(arr)
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            if arr[i][j] < 0.000001 and arr[i][j] > -0.000001:
                binary[i][j] = 100
            else:
                binary[i][j] = 0
    return binary


@jit(nopython=True)
def _reclassify_hpmf_filter_mean(arr):
    """
    Internally used reclassification of HPMF with different thresholds.
    """
    reclassify = np.copy(arr)
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            if arr[i][j] < 1:
                reclassify[i][j] = 0
            elif arr[i][j] < 3:
                reclassify[i][j] = 1
            elif arr[i][j] < 7:
                reclassify[i][j] = 2
            elif arr[i][j] < 10:
                reclassify[i][j] = 50
            elif arr[i][j] < 20:
                reclassify[i][j] = 75
            elif arr[i][j] < 50:
                reclassify[i][j] = 100
            elif arr[i][j] < 80:
                reclassify[i][j] = 300
            elif arr[i][j] < 100:
                reclassify[i][j] = 600
            else:
                reclassify[i][j] = 1000
    return reclassify


@jit
def hpmf_filter(arr):
    """
    HPMF ditch enhancement.
    """
    normalized_arr = da.from_array(
        _reclassify_hpmf_filter(arr), chunks=(800, 800))

    mean = d_gf(d_gf(d_gf(d_gf(normalized_arr, np.amax, footprint=create_circular_mask(1)), np.amax, footprint=create_circular_mask(
        1)), np.median, footprint=create_circular_mask(2)), np.nanmean, footprint=create_circular_mask(5)).compute(scheduler='processes')
    reclassify = da.from_array(
        _reclassify_hpmf_filter_mean(mean), chunks=(800, 800))

    return d_gf(reclassify, np.nanmean, footprint=create_circular_mask(7))


@jit(nopython=True)
def _reclassify_sky_view_non_ditch_amp(arr):
    """
    Internal non ditch amplification reclassification for SkyViewFactor.
    """
    new_arr = np.copy(arr)
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            if arr[i, j] < 0.92:
                new_arr[i, j] = 46
            elif arr[i, j] < 0.93:
                new_arr[i, j] = 37
            elif arr[i, j] < 0.94:
                new_arr[i, j] = 29
            elif arr[i, j] < 0.95:
                new_arr[i, j] = 22
            elif arr[i, j] < 0.96:
                new_arr[i, j] = 16
            elif arr[i, j] < 0.97:
                new_arr[i, j] = 11
            elif arr[i, j] < 0.98:
                new_arr[i, j] = 7
            elif arr[i, j] < 0.985:
                new_arr[i, j] = 4
            elif arr[i, j] < 0.99:
                new_arr[i, j] = 2
            else:
                new_arr[i, j] = 1
    return new_arr


@jit
def _sky_view_gabor(merged, gabors):
    """
    Internal SkyViewFactor merge of gabor filters.
    """
    for i in range(len(merged)):
        for j in range(len(merged[i])):
            merged[i][j] = 0
    for i in range(len(merged)):
        for j in range(len(merged[i])):
            for k in range(len(gabors)):
                merged[i][j] += gabors[k][i][j]
    return merged

#@jit
def sky_view_gabor(skyViewArr):
    """
    SkyViewFactor gabor filter.
    """
    delayed_gabors = []
    for i in np.arange(0.03, 0.08, 0.01):
        for j in np.arange(0, 3, 0.52):
            delayed_gabor = dk.delayed(gabor)(skyViewArr, i, j)[0]
            delayed_gabors.append(delayed_gabor)
    gabors = dk.compute(delayed_gabors)
    return _sky_view_gabor(skyViewArr.copy(), gabors[0])





@jit
def sky_view_non_ditch_amplification(arr):
    """
    Non ditch amplification from SkyViewFactor.
    """
    arr = da.from_array(arr, chunks=(800, 800))
    arr = d_gf(arr, np.nanmedian, footprint=create_circular_mask(25)
               ).compute(scheduler='processes')
    arr = da.from_array(
        _reclassify_sky_view_non_ditch_amp(arr), chunks=(800, 800))
    return d_gf(arr, np.nanmean, footprint=create_circular_mask(10))


@jit
def sky_view_conic_filter(arr, maskRadius, threshold):
    """
    Ditch amplification by taking the mean of cones in different directions from pixels.
    """
    # Standard values: maskRadius = 5, threshold = 0.975
    masks = []
    for i in range(0, 8):
        masks.append(_create_conic_mask(maskRadius, i))
    new_arr = arr.copy()
    amountOfThresholds = 0
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            values = _mean_from_masks(arr, (i, j), masks)
            dir1 = 2
            dir2 = 2
            dir3 = 2
            dir4 = 2
            if values[0] < threshold and values[4] < threshold:
                dir1 = values[0] if values[0] < values[4] else values[4]
            if values[1] < threshold and values[5] < threshold:
                dir2 = values[1] if values[0] < values[5] else values[4]
            if values[2] < threshold and values[6] < threshold:
                dir3 = values[2] if values[0] < values[6] else values[4]
            if values[3] < threshold and values[7] < threshold:
                dir4 = values[3] if values[0] < values[7] else values[4]
            dir5 = dir1 if dir1 < dir2 else dir2
            dir6 = dir3 if dir3 < dir4 else dir4
            lowest = dir5 if dir5 < dir6 else dir6
            if lowest < threshold:
                amountOfThresholds += 1
                new_arr[i][j] = 0.95 * lowest if lowest * \
                    0.95 < arr[i][j] else arr[i][j]
    return new_arr


@jit
def _mean_from_masks(arr, index, masks):
    """
    Internally used to calculate the mean of conic masks from a pixel.
    """
    row = index[0]
    col = index[1]
    halfMask = len(masks[0]) // 2
    arrLenRow = len(arr)
    arrLenCol = len(arr[row])
    values = np.zeros(8)
    elementAmounts = np.zeros(8)
    for i in range(-halfMask, halfMask):
        for j in range(-halfMask, halfMask):
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
        values[i] = values[i] / \
            elementAmounts[i] if elementAmounts[i] != 0 else 0.99
    return values


@jit
def _create_conic_mask(radius, direction):
    """
    Internally used to create conic mask for a direction and with a certain radius.
    """
    kernel = np.zeros((2*radius+1, 2*radius+1))
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]

    if direction == 0:  # topright
        mask = (x > y) & (x < abs(y)) & (x**2 + y**2 <= radius**2) & (x > 0)
    elif direction == 1:  # righttop
        mask = (x > abs(y)) & (x**2 + y**2 <= radius**2) & (y < 0)
    elif direction == 2:  # rightbottom
        mask = (x > abs(y)) & (x**2 + y**2 <= radius**2) & (y > 0)
    elif direction == 3:  # bottomright
        mask = (abs(x) < y) & (x**2 + y**2 <= radius**2) & (x > 0)
    elif direction == 4:  # bottomleft
        mask = (abs(x) < y) & (x**2 + y**2 <= radius**2) & (x < 0)
    elif direction == 5:  # leftbottom
        mask = (abs(x) > abs(y)) & (x < abs(y)) & (
            x**2 + y**2 <= radius**2) & (y > 0)
    elif direction == 6:  # lefttop
        mask = (abs(x) > abs(y)) & (x < abs(y)) & (
            x**2 + y**2 <= radius**2) & (y < 0)
    elif direction == 7:  # topleft
        mask = (x > y) & (x < abs(y)) & (x**2 + y**2 <= radius**2) & (x < 0)
    kernel[mask] = 1
    return kernel


@jit(nopython=True)
def _slope_non_ditch_amplifcation_normalize(arr, new_arr):
    """
    Internal non ditch amplification reclassification for Slope.
    """
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            if arr[i][j] < 8:
                new_arr[i][j] = 0
            elif arr[i][j] < 9:
                new_arr[i][j] = 20
            elif arr[i][j] < 10:
                new_arr[i][j] = 25
            elif arr[i][j] < 11:
                new_arr[i][j] = 30
            elif arr[i][j] < 13:
                new_arr[i][j] = 34
            elif arr[i][j] < 15:
                new_arr[i][j] = 38
            elif arr[i][j] < 17:
                new_arr[i][j] = 42
            elif arr[i][j] < 19:
                new_arr[i][j] = 46
            elif arr[i][j] < 21:
                new_arr[i][j] = 50
            else:
                new_arr[i][j] = 55
    return new_arr


@jit
def slope_non_ditch_amplification(arr):
    """
    Non ditch amplification from Slope.
    """
    new_arr = arr.copy()
    arr = d_gf(da.from_array(arr, chunks=(800, 800)), np.nanmedian,
               footprint=create_circular_mask(35)).compute(scheduler='processes')
    new_arr = _slope_non_ditch_amplifcation_normalize(arr, new_arr)
    return d_gf(da.from_array(new_arr, chunks=(800, 800)), np.nanmean, footprint=create_circular_mask(15))


